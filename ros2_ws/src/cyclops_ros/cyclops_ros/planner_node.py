"""ROS 2 planning node: obstacle estimates + odometry -> PX4 offboard velocity setpoints.

Subscribes
    /cyclops/obstacles              std_msgs/Float32MultiArray (from perception_node)
    /fmu/out/vehicle_odometry       px4_msgs/VehicleOdometry
    /fmu/out/vehicle_status         px4_msgs/VehicleStatus
Publishes
    /fmu/in/offboard_control_mode   px4_msgs/OffboardControlMode   (velocity control)
    /fmu/in/trajectory_setpoint     px4_msgs/TrajectorySetpoint    (NED velocity + yaw)
    /fmu/in/vehicle_command         px4_msgs/VehicleCommand        (arm, OFFBOARD mode)

Mission: take off to `altitude`, then fly to (goal_x, goal_y) in the CYCLOPS frame
(x = north, y = west/left of start) with the APF-rollout planner, then land.
PX4 requires a steady stream (>2 Hz) of OffboardControlMode before switching to OFFBOARD.
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry,
                          VehicleStatus)

from .frames import ned_to_cyclops_pos, ned_quat_to_cyclops_R, cyclops_to_ned_vec, cyclops_yaw_to_ned


class PlannerNode(Node):
    def __init__(self):
        super().__init__('cyclops_planner')
        self.declare_parameter('cyclops_root', os.environ.get('CYCLOPS_ROOT', ''))
        self.declare_parameter('goal_x', 40.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('altitude', 2.0)
        self.declare_parameter('rate_hz', 15.0)
        root = self.get_parameter('cyclops_root').value
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from cyclops.config import Config
        from cyclops.planning.planners import APFRolloutPlanner
        from cyclops.geometry import rot_to_euler
        self.rot_to_euler = rot_to_euler
        self.cfg = Config()
        self.alt = float(self.get_parameter('altitude').value)
        self.planner = APFRolloutPlanner(self.cfg.planner, self.alt, self.cfg.camera.hfov_deg)
        self.goal = np.array([self.get_parameter('goal_x').value, self.get_parameter('goal_y').value, self.alt])

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.on_odom, qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.on_status, qos)
        self.create_subscription(Float32MultiArray, '/cyclops/obstacles', self.on_obstacles, 10)
        self.pub_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.pos = None
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.obstacles = []
        self.status = None
        self.phase = 'WAIT'
        self.counter = 0
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)
        self.create_timer(self.dt, self.loop)
        self.get_logger().info(f'CYCLOPS planner: goal {self.goal[:2]} at {self.alt} m')

    # ------------------------------------------------------------------ inputs
    def on_odom(self, msg):
        self.pos = ned_to_cyclops_pos(msg.position)
        self.vel = ned_to_cyclops_pos(msg.velocity)
        self.yaw = self.rot_to_euler(ned_quat_to_cyclops_R(msg.q))[2]

    def on_status(self, msg):
        self.status = msg

    def on_obstacles(self, msg):
        a = np.array(msg.data, float).reshape(-1, 9)
        self.obstacles = [dict(id=int(r[0]), pos=r[1:4], vel=np.array([r[4], r[5], 0.0]), radius=r[6],
                               sigma=r[7], mode=int(r[8])) for r in a]

    # ----------------------------------------------------------------- outputs
    def _now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def command(self, cmd, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.command, m.param1, m.param2 = cmd, float(p1), float(p2)
        m.target_system = m.target_component = m.source_system = m.source_component = 1
        m.from_external = True
        m.timestamp = self._now_us()
        self.pub_cmd.publish(m)

    def send_velocity(self, v_cyclops, yaw_cyclops):
        mode = OffboardControlMode()
        mode.velocity = True
        mode.timestamp = self._now_us()
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.position = [float('nan')] * 3
        sp.velocity = [float(x) for x in cyclops_to_ned_vec(v_cyclops)]
        sp.yaw = float(cyclops_yaw_to_ned(yaw_cyclops))
        sp.timestamp = self._now_us()
        self.pub_sp.publish(sp)

    # -------------------------------------------------------------------- loop
    def loop(self):
        if self.pos is None:
            return
        self.counter += 1
        if self.phase == 'WAIT':
            self.send_velocity(np.zeros(3), self.yaw)
            if self.counter > 20:                      # 1+ s of setpoints streamed
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)   # custom mode 6 = OFFBOARD
                self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
                self.phase = 'TAKEOFF'
                self.get_logger().info('arm + OFFBOARD requested')
        elif self.phase == 'TAKEOFF':
            vz = np.clip(1.2 * (self.alt - self.pos[2]), -1.0, 1.0)
            self.send_velocity(np.array([0, 0, vz]), self.yaw)
            if self.pos[2] > self.alt - 0.15:
                self.phase = 'MISSION'
                self.get_logger().info('MISSION: avoidance active')
        elif self.phase == 'MISSION':
            sp, info = self.planner.plan(self.pos, self.vel, self.yaw, self.goal, self.obstacles, self.dt)
            self.send_velocity(sp['v'], sp['yaw'])
            if np.linalg.norm(self.goal[:2] - self.pos[:2]) < self.cfg.planner.goal_tolerance:
                self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.phase = 'LAND'
                self.get_logger().info('goal reached -> LAND')


def main():
    rclpy.init()
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
