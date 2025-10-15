#!/usr/bin/env python3
import rclpy, csv, math
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

def lpf(prev, new, alpha):
    return alpha*new + (1.0-alpha)*prev

class EffortRampLogger(Node):
    def __init__(self):
        super().__init__('effort_ramp_logger')

        # ===== Parameters =====
        self.joint       = self.declare_parameter('joint', 'slider_joint').value
        self.dt          = float(self.declare_parameter('dt', 0.001).value)       # [s]
        self.kNps        = float(self.declare_parameter('kNps', 5000.0).value)    # F = k * t
        self.t_end       = float(self.declare_parameter('t_end', 2.0).value)      # [s]
        self.csvpath     = self.declare_parameter('csv_path','/tmp/force_velocity.csv').value
        self.cmd_topic   = self.declare_parameter('cmd_topic','/prismatic_effort_controller/commands').value
        self.cmd_scalar  = bool(self.declare_parameter('cmd_scalar', True).value)

        # friction estimation settings
        self.mass        = float(self.declare_parameter('mass', 280.85).value)    # [kg]
        self.mu_s        = float(self.declare_parameter('mu_s', 0.5).value)
        self.g           = 9.81
        self.v_thresh    = float(self.declare_parameter('v_thresh', 1e-4).value)  # [m/s] 0.1mm/s

        # wrench topics auto-detect
        self.invert_wrench = bool(self.declare_parameter('invert_wrench', False).value)
        self.wrench_candidates = self.declare_parameter(
            'wrench_candidates',
            ['/slider_joint/wrench','/joint_wrench','/wrench','/onelink/slider_joint/wrench']
        ).value

        # simple smoothing (LPF)
        self.alpha_vel = float(self.declare_parameter('alpha_vel', 0.2).value)    # 0~1 (higher=faster)
        self.alpha_acc = float(self.declare_parameter('alpha_acc', 0.2).value)

        # ===== Publishers =====
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 1)

        # ===== Subscriptions =====
        self.sub_js  = self.create_subscription(JointState, '/joint_states', self.on_js, 50)
        self.wrench_sub = None
        self.latest_wrench_x = None
        self.detect_timer = self.create_timer(0.2, self.try_connect_wrench)

        # ===== State for kinematics =====
        self.jidx = None
        self.pos  = 0.0
        self.vel  = 0.0         # filtered vel
        self.acc  = 0.0         # filtered acc
        self._last = None       # (pos, vel_raw, t)

        # ===== CSV =====
        self.csvf = open(self.csvpath, 'w', newline='')
        self.writer = csv.writer(self.csvf)
        self.writer.writerow([
            'time [s]', 'applied force [N]',
            'friction (wrench) [N]', 'friction (dyn) [N]',
            'position [m]', 'velocity [m/s]', 'accel [m/s^2]'
        ])
        self.get_logger().info(f"[EffortRampLogger] Logging → {self.csvpath}")

        # ===== Run loop =====
        self.t = 0.0
        self.breakaway_t_wrench = None
        self.breakaway_t_dyn    = None
        self.mu_d_samples_wrench = []
        self.mu_d_samples_dyn    = []
        self.timer = self.create_timer(self.dt, self.step)

        # Info
        self.get_logger().info(f"k={self.kNps} N/s, t_end={self.t_end}s, dt={self.dt}s, mass={self.mass} kg")
        self.get_logger().info("Will log BOTH friction estimates: wrench & dyn (F_cmd - m*a)")

    # ----- Auto-detect a wrench topic -----
    def try_connect_wrench(self):
        if self.wrench_sub is not None:
            return
        topics = dict(self.get_topic_names_and_types())
        for cand in self.wrench_candidates:
            if cand in topics and 'geometry_msgs/msg/WrenchStamped' in topics[cand]:
                self.wrench_sub = self.create_subscription(WrenchStamped, cand, self.on_wrench, 50)
                self.get_logger().info(f"[EffortRampLogger] Subscribed wrench: {cand}")
                self.detect_timer.cancel()
                return

    # ----- Callbacks -----
    def on_wrench(self, msg: WrenchStamped):
        x = msg.wrench.force.x
        if self.invert_wrench:
            x = -x
        self.latest_wrench_x = x

    def on_js(self, msg: JointState):
        if self.jidx is None:
            try:
                self.jidx = msg.name.index(self.joint)
            except ValueError:
                return
        if self.jidx >= len(msg.position):
            return

        pos = msg.position[self.jidx]
        vel_raw = None
        if self.jidx < len(msg.velocity):
            vel_raw = msg.velocity[self.jidx]

        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        if self._last is not None:
            last_pos, last_vel_raw, last_t = self._last
            dt = max(1e-6, t_now - last_t)
            if vel_raw is None:
                vel_raw = (pos - last_pos)/dt
            # filter vel
            self.vel = lpf(self.vel, vel_raw, self.alpha_vel) if math.isfinite(vel_raw) else self.vel
            # numerical diff for acc using filtered vel
            acc_raw = (self.vel - (lpf(0.0, last_vel_raw if last_vel_raw is not None else self.vel, 1.0))) / dt
            self.acc = lpf(self.acc, acc_raw, self.alpha_acc)
        else:
            # initialize with zeros
            self.vel = 0.0
            self.acc = 0.0

        self.pos = pos
        self._last = (pos, vel_raw, t_now)

    # ----- Main loop -----
    def step(self):
        if self.t > self.t_end:
            self.publish_effort(0.0)
            self.csvf.close()
            self.print_summary()
            self.get_logger().info("[EffortRampLogger] Done.")
            rclpy.shutdown()
            return

        # Command force ramp
        F_cmd = self.kNps * self.t
        self.publish_effort(F_cmd)

        # Two friction estimates
        F_fric_wrench = float('nan')
        if self.latest_wrench_x is not None:
            F_fric_wrench = self.latest_wrench_x

        F_fric_dyn = F_cmd - self.mass * self.acc  # dynamics-based

        # Breakaway detect (separately for each measure)
        if self.breakaway_t_wrench is None and abs(self.vel) > self.v_thresh and math.isfinite(F_fric_wrench):
            self.breakaway_t_wrench = self.t
        if self.breakaway_t_dyn is None and abs(self.vel) > self.v_thresh and math.isfinite(F_fric_dyn):
            self.breakaway_t_dyn = self.t

        # Kinetic mu estimates (when surely in slip)
        if abs(self.vel) > 3.0*self.v_thresh:
            if math.isfinite(F_fric_wrench):
                mu_w = abs(F_fric_wrench)/(self.mass*self.g)
                if mu_w < 2.0: self.mu_d_samples_wrench.append(mu_w)
            if math.isfinite(F_fric_dyn):
                mu_d = abs(F_fric_dyn)/(self.mass*self.g)
                if mu_d < 2.0: self.mu_d_samples_dyn.append(mu_d)

        # Log row
        self.writer.writerow([
            f"{self.t:.6f}",              # time [s]
            f"{F_cmd:.6f}",               # applied force [N]
            f"" if not math.isfinite(F_fric_wrench) else f"{F_fric_wrench:.6f}",   # friction (wrench) [N]
            f"{F_fric_dyn:.6f}",          # friction (dyn) [N]
            f"{self.pos:.9f}",            # position [m]
            f"{self.vel:.9f}",            # velocity [m/s]
            f"{self.acc:.9f}",            # accel [m/s^2]
        ])

        self.t += self.dt

    def publish_effort(self, effort: float):
        # Many ros2_control effort controllers accept Float64 OR Float64MultiArray.
        # Here we publish MultiArray([effort]) by default.
        msg = Float64MultiArray()
        msg.data = [effort] if self.cmd_scalar else [effort]
        self.pub_cmd.publish(msg)

    def print_summary(self):
        Fs = self.mu_s * self.mass * self.g
        ts = Fs / self.kNps
        def avg(xs): return (sum(xs)/len(xs)) if xs else float('nan')

        msg = "\n===== Summary =====\n"
        msg += f"theoretical F_s={Fs:.2f} N, t_s={ts:.4f} s\n"
        if self.breakaway_t_wrench is not None:
            msg += f"breakaway (wrench) t≈{self.breakaway_t_wrench:.4f} s (Δ={self.breakaway_t_wrench-ts:+.4f})\n"
        else:
            msg += "breakaway (wrench): n/a\n"
        if self.breakaway_t_dyn is not None:
            msg += f"breakaway (dyn)    t≈{self.breakaway_t_dyn:.4f} s (Δ={self.breakaway_t_dyn-ts:+.4f})\n"
        else:
            msg += "breakaway (dyn): n/a\n"

        msg += f"mu_d (wrench, slip avg)≈{avg(self.mu_d_samples_wrench):.3f}\n"
        msg += f"mu_d (dyn,    slip avg)≈{avg(self.mu_d_samples_dyn):.3f}\n"
        msg += f"CSV: {self.csvpath}\n"
        msg += "===================\n"
        print(msg)

def main():
    rclpy.init()
    node = EffortRampLogger()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
