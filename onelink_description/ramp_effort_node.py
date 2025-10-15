#!/usr/bin/env python3
import rclpy, csv
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

class EffortRampLogger(Node):
    def __init__(self):
        super().__init__('effort_ramp_logger')

        # ===== 설정 =====
        self.joint   = self.declare_parameter('joint', 'slider_joint').value
        self.dt      = float(self.declare_parameter('dt', 0.001).value)      # 1kHz
        self.kNps    = float(self.declare_parameter('kNps', 5000.0).value)   # F = k * t
        self.csvpath = self.declare_parameter('csv_path','/tmp/force_velocity.csv').value

        # 마찰력 추정: 'wrench' 또는 'joint_effort'
        self.friction_source = self.declare_parameter('friction_source','wrench').value
        self.use_wrench = (self.friction_source == 'wrench')
        self.invert_wrench = bool(self.declare_parameter('invert_wrench', True).value)  #부호반대
        self.wrench_topic  = self.declare_parameter('wrench_topic','/slider_joint/wrench').value

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
        self.fx       = float('nan')

        # ===== CSV =====
        self.csv = open(self.csvpath, 'w', newline='')
        self.w = csv.writer(self.csv)
        self.w.writerow(['time [s]', 'applied force', 'sensor force', 'friction force', 'pos', 'vel', 'eff'])

        # 타이머(정확한 1kHz는 OS 지터가 있지만, 명령 F와 time은 "샘플 인덱스"로 보정)
        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info(f"Waiting for sensors... target dt={self.dt*1000:.1f} ms, k={self.kNps} N/s")

    # 콜백
    def cb_js(self, msg: JointState):
        try:
            i = msg.name.index(self.joint)
            self.last_js = (msg.position[i], msg.velocity[i], msg.effort[i])
            self.js_ready = True
        except ValueError:
            pass

    def cb_ft(self, msg: WrenchStamped):
        fx = msg.wrench.force.x
        self.fx = -fx if self.invert_wrench else fx
        self.ft_ready = True

    # 1kHz 루프
    def step(self):
        # 준비 전: 힘 0 내보내며 대기, 샘플 인덱스도 0 유지 → 시작이 항상 t=0.0
        if not (self.js_ready and self.ft_ready):
            self.publish_force(0.0)
            self.sample_i = 0
            return

        # 시작 플래그
        if not self.started:
            self.started = True
            self.get_logger().info("Sensors ready. Start at t=0.000 s")

        # 시간/외력: 샘플 인덱스 기반 (지터 무시)
        t = self.sample_i * self.dt
        F = self.kNps * t
        self.publish_force(F)

        # 로깅
        if self.last_js:
            p, v, e = self.last_js
            if self.use_wrench:
                fric = self.fx
            else:
                fric = -e  # 반작용 가정(부호 맞추기)
            new_fric = fric-F
            self.w.writerow([f"{t:.6f}", f"{F:.2f}", f"{fric:.2f}", f"{new_fric:.2f}", f"{p:.6f}", f"{v:.6f}", f"{e:.6f}"])

        self.sample_i += 1
        if t >= 2.0:  # 최대 2초
            self.finish()

    def publish_force(self, F):
        self.pub.publish(Float64MultiArray(data=[float(F)]))

    def finish(self):
        self.publish_force(0.0)
        self.csv.close()
        self.get_logger().info(f"종료 CSV: {self.csvpath}")
        rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(EffortRampLogger())

if __name__ == '__main__':
    main()
