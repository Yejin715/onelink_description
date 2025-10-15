#!/usr/bin/env python3
import rclpy, csv, math, collections
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

class EffortRampLogger(Node):
    def __init__(self):
        super().__init__('effort_ramp_logger')

        # ===== 실험/모델 파라미터 =====
        self.joint    = self.declare_parameter('joint', 'slider_joint').value
        self.dt       = float(self.declare_parameter('dt', 0.001).value)           # 제어/로깅 주기 (s)
        self.kNps     = float(self.declare_parameter('kNps', 5000.0).value)        # F = k * t  [N/s]
        self.duration = float(self.declare_parameter('duration', 2.0).value)      # 실험 시간 (s)
        self.csvpath  = self.declare_parameter('csv_path','/tmp/force_velocity.csv').value

        # 마찰 소스: 'wrench' (권장) 또는 'joint_effort'
        self.friction_source = self.declare_parameter('friction_source','wrench').value
        self.use_wrench = (self.friction_source == 'wrench')
        self.invert_wrench = bool(self.declare_parameter('invert_wrench', True).value)
        self.wrench_topic  = self.declare_parameter('wrench_topic','/slider_joint/wrench').value

        # 보정 파라미터
        self.mass    = float(self.declare_parameter('mass', 280.85).value)         # kg
        self.v_th    = float(self.declare_parameter('v_th', 1e-4).value)           # 0.1 mm/s
        self.g_proj  = float(self.declare_parameter('g_proj', 0.0).value)          # 축방향 중력성분(N), 수평축이면 0
        self.k_spring= float(self.declare_parameter('k_spring', 0.0).value)        # 축방향 스프링 N/m (없으면 0)
        self.axis    = self.declare_parameter('axis', [1.0,0.0,0.0]).value         # base_link 기준 축 단위벡터
        self.bias_alpha = float(self.declare_parameter('bias_alpha', 0.02).value)  # 정지구간 바이어스 EMA

        # ===== Pub/Sub =====
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/prismatic_effort_controller/commands', 1)
        self.sub_js = self.create_subscription(JointState, '/joint_states', self.cb_js, 100)
        if self.use_wrench:
            self.sub_ft = self.create_subscription(WrenchStamped, self.wrench_topic, self.cb_ft, 100)

        # ===== 상태 =====
        self.js_ready = False
        self.ft_ready = (not self.use_wrench)
        self.started  = False
        self.sample_i = 0
        self.last_js  = None  # (pos, vel, effort)
        self.fx_axis  = float('nan')  # 축방향 힘 (N)
        self.f_cmd    = 0.0           # 현재 인가한 외력 명령 (N)
        self.bias     = 0.0           # 정지구간 바이어스(EMA)
        self.vel_buf  = collections.deque(maxlen=9)  # 속도 버퍼(가속도 추정용)
        self.t_buf    = collections.deque(maxlen=9)  # 시간 버퍼

        # ===== CSV =====
        self.csv = open(self.csvpath, 'w', newline='')
        self.w = csv.writer(self.csv)
        self.w.writerow([
            'time[s]', 'F_cmd[N]', 'F_wrench_axis[N]', 'F_fric_src[N]', 'F_fric_est[N]',
            'pos[m]', 'vel[m/s]', 'acc[m/s2]', 'bias[N]'
        ])

        # 타이머
        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info(
            f"Waiting for sensors... dt={self.dt*1000:.1f} ms, k={self.kNps} N/s, duration={self.duration}s")

    # ===== 콜백 =====
    def cb_js(self, msg: JointState):
        try:
            i = msg.name.index(self.joint)
            p = msg.position[i]
            v = msg.velocity[i]
            e = msg.effort[i]
            self.last_js = (p, v, e, msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9)
            self.vel_buf.append(v)
            self.t_buf.append(self.last_js[3])
            self.js_ready = True
        except ValueError:
            pass

    def cb_ft(self, msg: WrenchStamped):
        # base_link 기준 힘 벡터에서 축방향 성분 추출
        fx = msg.wrench.force.x * self.axis[0] + \
             msg.wrench.force.y * self.axis[1] + \
             msg.wrench.force.z * self.axis[2]
        if self.invert_wrench:
            fx = -fx
        self.fx_axis = fx
        self.ft_ready = True

    # ===== 가속도 추정 (간단 중앙차분) =====
    def estimate_acc(self):
        if len(self.vel_buf) < 3:
            return 0.0
        # 평균 샘플 주기 근사
        dt = (self.t_buf[-1] - self.t_buf[0]) / max(1, (len(self.t_buf)-1))
        v1, v2, v3 = self.vel_buf[-3], self.vel_buf[-2], self.vel_buf[-1]
        return (v3 - v1) / max(1e-9, (2*dt))

    # ===== 메인 루프 =====
    def step(self):
        # 준비 전: 힘 0 내보내며 대기, t=0 유지
        if not (self.js_ready and self.ft_ready):
            self.publish_force(0.0)
            self.sample_i = 0
            return

        # 시작 플래그
        if not self.started:
            self.started = True
            self.get_logger().info("Sensors ready. Start at t=0.000 s")

        # 시간/외력 명령
        t = self.sample_i * self.dt
        F_cmd = self.kNps * t
        self.publish_force(F_cmd)

        # 로깅 준비
        acc = self.estimate_acc()
        F_wrench_axis = self.fx_axis if self.use_wrench else float('nan')

        # 마찰 소스 선택 (원자료)
        F_fric_src = float('nan')
        if self.last_js:
            p, v, e, _ = self.last_js
            if self.use_wrench:
                F_fric_src = F_wrench_axis
            else:
                # joint_effort를 마찰 근사(부호 보정)
                F_fric_src = -e

            # 정지구간 바이어스 EMA (외력/관성/중력이 거의 0일 때의 오프셋 제거)
            if abs(v) < self.v_th:
                self.bias = (1.0 - self.bias_alpha) * self.bias + self.bias_alpha * (F_wrench_axis if self.use_wrench else 0.0)

            # 보정 마찰 추정치
            Fg = self.g_proj
            Fs = self.k_spring * p if self.k_spring != 0.0 else 0.0
            F_inert = self.mass * acc

            # wrench 기준 보정
            if self.use_wrench:
                F_fric_est = (F_wrench_axis - F_cmd - F_inert - Fg - Fs - self.bias)
            else:
                # joint_effort 기준일 때는, effort ≈ 외력 + 마찰 + 기타 → 마찰만 보려면
                #   (-e) - (-F_cmd) - F_inert - Fg - Fs - bias  형태로도 볼 수 있으나
                # 일반적으로 joint_effort는 이미 외력/관성과 분리해 쓰지 않으므로, 여기선 동일식 적용
                F_fric_est = ((-e) - (-F_cmd) - F_inert - Fg - Fs - 0.0)

            # CSV 기록
            self.w.writerow([
                f"{t:.6f}",
                f"{F_cmd:.6f}",
                f"{F_wrench_axis:.6f}" if not math.isnan(F_wrench_axis) else "",
                f"{F_fric_src:.6f}",
                f"{F_fric_est:.6f}",
                f"{p:.6f}", f"{v:.6f}", f"{acc:.6f}", f"{self.bias:.6f}"
            ])

        self.f_cmd = F_cmd
        self.sample_i += 1

        # 종료
        if t >= self.duration:
            self.finish()

    # ===== 퍼블리시/종료 =====
    def publish_force(self, F):
        self.pub.publish(Float64MultiArray(data=[float(F)]))

    def finish(self):
        self.publish_force(0.0)
        self.csv.close()
        self.get_logger().info(f"[완료] CSV 저장: {self.csvpath}")
        rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(EffortRampLogger())

if __name__ == '__main__':
    main()
