#!/usr/bin/env python3
import rclpy, csv
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

class VelRampLogger(Node):
    def __init__(self):
        super().__init__('vel_ramp_logger')

        # === 파라미터 ===
        self.jname       = self.declare_parameter('joint', 'slider_joint').value
        self.lower       = float(self.declare_parameter('lower_limit', -1.45).value)
        self.upper       = float(self.declare_parameter('upper_limit',  +1.45).value)
        self.margin      = float(self.declare_parameter('stop_margin',   0.03).value)

        self.start_side  = self.declare_parameter('start_side', 'upper').value  # 'upper' or 'lower'
        self.home_speed  = float(self.declare_parameter('home_speed', 0.15).value)
        self.settle_wait = float(self.declare_parameter('settle_wait', 2.0).value)

        self.log_home    = bool(self.declare_parameter('log_home', False).value)
        self.log_settle  = bool(self.declare_parameter('log_settle', False).value)

        self.a           = float(self.declare_parameter('accel', 0.30).value)   # m/s^2
        self.v_max       = float(self.declare_parameter('v_max', 1.00).value)   # m/s
        self.dt          = 0.01

        # ✅ 리밋 도달 후 추가 기록 시간(초)
        self.after_limit_hold = float(self.declare_parameter('after_limit_hold', 5.0).value)

        # 퍼블리셔/서브스크라이버
        self.pub    = self.create_publisher(Float64MultiArray, '/prismatic_velocity_controller/commands', 1)
        self.sub_js = self.create_subscription(JointState, '/joint_states', self.cb_js, 10)

        self.wrench_topic = self.declare_parameter('wrench_topic', '/wrench').value
        self.sub_ft = self.create_subscription(WrenchStamped, self.wrench_topic, self.cb_ft, 10)

        # CSV
        self.csv_path = self.declare_parameter('csv_path', '/tmp/force_velocity.csv').value
        self.csvf = open(self.csv_path, 'w', newline='')
        self.w    = csv.writer(self.csvf)
        # t는 EXP 시작을 0으로, LIMIT 구간도 같은 시간축(연속)으로 기록
        self.w.writerow(['phase','t','pos','vel','effort_js','wrench_fx','cmd_v'])

        # 상태
        self.last_js = None      # (pos, vel, effort_js)
        self.last_fx = 0.0       # wrench.force.x
        self.phase   = 'home'    # 'home' -> 'settle' -> 'exp' -> 'limit'
        self._k      = 0

        # 시간 기준들
        self.home_start   = self.get_clock().now()
        self.settle_start = None
        self.exp_start    = None
        self.limit_start  = None

        # homing 목표/판정
        self.goal_pos = self.upper - self.margin if self.start_side=='upper' else self.lower + self.margin
        self.home_cmd = +abs(self.home_speed) if self.start_side=='upper' else -abs(self.home_speed)

        # 실험(램프) 방향: 끝에서 반대 방향으로
        self.sign = -1.0 if self.start_side=='upper' else +1.0

        # LIMIT 구간에서 유지할 명령 (리밋 진입 직전의 v_cmd를 유지)
        self.hold_cmd_v = 0.0

        # 타이머
        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info(
            f"[HOME] move to {self.start_side} end (goal≈{self.goal_pos:.3f} m, v={self.home_cmd:.2f} m/s)"
        )

    # 콜백들
    def cb_js(self, msg: JointState):
        try:
            i = msg.name.index(self.jname)
            self.last_js = (msg.position[i], msg.velocity[i], msg.effort[i])
        except ValueError:
            pass

    def cb_ft(self, msg: WrenchStamped):
        self.last_fx = msg.wrench.force.x

    # 유틸
    def now(self):
        return self.get_clock().now()

    def elapsed(self, start_time):
        return (self.now() - start_time).nanoseconds * 1e-9

    def publish_vel(self, v):
        self.pub.publish(Float64MultiArray(data=[float(v)]))

    def step(self):
        # ----- HOME -----
        if self.phase == 'home':
            self.publish_vel(self.home_cmd)
            if self.last_js:
                p,v,e = self.last_js
                if self.log_home:
                    t_home = self.elapsed(self.home_start)
                    self.w.writerow(['home', f'{t_home:.4f}', f'{p:.6f}', f'{v:.6f}', f'{e:.3f}', f'{self.last_fx:.6f}', f'{self.home_cmd:.4f}'])
                if (self.start_side=='upper' and p >= self.goal_pos) or (self.start_side=='lower' and p <= self.goal_pos):
                    self.publish_vel(0.0)
                    self.phase = 'settle'
                    self.settle_start = self.now()
                    self.get_logger().info(f"[SETTLE] reached {self.start_side} end (pos={p:.3f}), wait {self.settle_wait:.1f}s...")
            return

        # ----- SETTLE -----
        if self.phase == 'settle':
            self.publish_vel(0.0)
            if self.log_settle and self.last_js:
                p,v,e = self.last_js
                t_settle = self.elapsed(self.settle_start)
                self.w.writerow(['settle', f'{t_settle:.4f}', f'{p:.6f}', f'{v:.6f}', f'{e:.3f}', f'{self.last_fx:.6f}', f'{0.0:.4f}'])
            if self.elapsed(self.settle_start) >= self.settle_wait:
                self.exp_start = self.now()
                self.phase = 'exp'
                self._k = 0
                self.get_logger().info("[EXP] start experiment: t reset to 0, begin logging")
            return

        # ----- EXP -----
        if self.phase == 'exp':
            t = self.elapsed(self.exp_start) if self.exp_start else 0.0
            v_cmd = self.sign * min(self.a * t, self.v_max)
            self.publish_vel(v_cmd)

            if self.last_js:
                p,v,e = self.last_js
                self.w.writerow(['exp', f'{t:.4f}', f'{p:.6f}', f'{v:.6f}', f'{e:.3f}', f'{self.last_fx:.6f}', f'{v_cmd:.4f}'])

                if self._k % int(0.5/self.dt) == 0:
                    self.get_logger().info(f"[EXP] t={t:.2f}s cmd_v={v_cmd:.3f} pos={p:.3f} vel={v:.3f} Fx={self.last_fx:.2f}")
                self._k += 1

                # 리밋 진입 감지 → LIMIT 구간으로 전환
                if self.start_side=='upper':
                    at_limit = (p <= (self.lower + self.margin))
                else:
                    at_limit = (p >= (self.upper - self.margin))
                if at_limit:
                    self.phase = 'limit'
                    self.limit_start = self.now()
                    self.hold_cmd_v = v_cmd  # ✅ 직전 명령을 유지하며 계속 '밀기'
                    self.get_logger().info(f"[LIMIT] entered limit window, hold {self.after_limit_hold:.1f}s with cmd_v={self.hold_cmd_v:.3f}")
            # 안전 타임아웃(혹시 limit 못 찾으면)
            if t > 120.0:
                return self.finish('Timeout (no limit)')
            return

        # ----- LIMIT (리밋 도달 이후 추가 기록) -----
        if self.phase == 'limit':
            t = self.elapsed(self.exp_start)  # 시간축은 EXP 기준 연속 유지
            self.publish_vel(self.hold_cmd_v) # 계속 같은 방향으로 밀어 제한 동작 관찰
            if self.last_js:
                p,v,e = self.last_js
                self.w.writerow(['limit', f'{t:.4f}', f'{p:.6f}', f'{v:.6f}', f'{e:.3f}', f'{self.last_fx:.6f}', f'{self.hold_cmd_v:.4f}'])
            if self.elapsed(self.limit_start) >= self.after_limit_hold:
                return self.finish('Hold done')
            return

    def finish(self, reason):
        self.publish_vel(0.0)
        self.get_logger().info(f'종료({reason}). CSV: {self.csv_path}')
        self.csvf.close()
        rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(VelRampLogger())

if __name__ == '__main__':
    main()
