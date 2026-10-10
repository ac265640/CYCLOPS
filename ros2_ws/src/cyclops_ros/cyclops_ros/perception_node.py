"""ROS 2 perception node (report section 8.5).

Subscribes
    /camera/image_raw                sensor_msgs/Image (rgb8 / bgr8)  - forward mono camera
    /fmu/out/vehicle_odometry        px4_msgs/VehicleOdometry          - camera pose
Publishes
    /cyclops/obstacles               std_msgs/Float32MultiArray        - rows of
                                     [id, x, y, z, vx, vy, radius, sigma, mode]  (CYCLOPS world frame)
    /cyclops/markers                 visualization_msgs/MarkerArray    - for RViz
    /cyclops/latency_ms              std_msgs/Float32

The heavy lifting is the same code as the simulator: cyclops.perception.*
"""
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

from .frames import ned_to_cyclops_pos, ned_quat_to_cyclops_R

try:
    from px4_msgs.msg import VehicleOdometry
except ImportError:          # allows running without px4_msgs (pose fixed at origin)
    VehicleOdometry = None


def _import_core(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)
    from cyclops.config import Config
    from cyclops.perception.perception import Perception
    from cyclops.perception.tracker import ObstacleTracker
    return Config, Perception, ObstacleTracker


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('cyclops_perception')
        self.declare_parameter('cyclops_root', os.environ.get('CYCLOPS_ROOT', ''))
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('device', 'cpu')
        root = self.get_parameter('cyclops_root').value
        Config, Perception, ObstacleTracker = _import_core(root)
        self.cfg = Config()
        cp = self.cfg.camera
        kw = {'models_dir': os.path.join(root, 'models')} if root else {}
        self.perception = Perception(cp, device=self.get_parameter('device').value, **kw)
        self.tracker = ObstacleTracker(self.cfg.planner, cp)
        self.pos = np.array([0.0, 0.0, 2.0])
        self.R = np.eye(3)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image, qos)
        if VehicleOdometry is not None:
            self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.on_odom, qos)
        self.pub_obs = self.create_publisher(Float32MultiArray, '/cyclops/obstacles', 10)
        self.pub_mk = self.create_publisher(MarkerArray, '/cyclops/markers', 10)
        self.pub_lat = self.create_publisher(Float32, '/cyclops/latency_ms', 10)
        self.get_logger().info('CYCLOPS perception node ready')

    def on_odom(self, msg):
        self.pos = ned_to_cyclops_pos(msg.position)
        self.R = ned_quat_to_cyclops_R(msg.q)

    def on_image(self, msg):
        t0 = time.perf_counter()
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        if msg.encoding.lower().startswith('bgr'):
            img = img[..., ::-1]
        cp = self.cfg.camera
        if img.shape[0] != cp.height or img.shape[1] != cp.width:
            # nearest-neighbour resize to the network resolution (keeps dependencies minimal)
            ys = (np.arange(cp.height) * img.shape[0] / cp.height).astype(int)
            xs = (np.arange(cp.width) * img.shape[1] / cp.width).astype(int)
            img = img[ys][:, xs]
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        dets, _ = self.perception.process(np.ascontiguousarray(img), self.pos, self.R)
        self.tracker.update(stamp, dets, self.pos, self.R)
        obs = self.tracker.obstacles()

        arr = Float32MultiArray()
        arr.layout.dim = [MultiArrayDimension(label='obstacles', size=len(obs), stride=9 * len(obs)),
                          MultiArrayDimension(label='fields', size=9, stride=9)]
        arr.data = [float(v) for o in obs for v in (o['id'], *o['pos'], *o['vel'][:2], o['radius'],
                                                     o['sigma'], o['mode'])]
        self.pub_obs.publish(arr)

        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for o in obs:
            m = Marker()
            m.header.frame_id = 'cyclops_world'
            m.header.stamp = msg.header.stamp
            m.ns, m.id, m.type = 'obstacles', int(o['id']), Marker.CYLINDER
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, o['pos'])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = float(2 * o['radius'])
            m.scale.z = 3.0
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 0.85, 0.6
            ma.markers.append(m)
        self.pub_mk.publish(ma)
        self.pub_lat.publish(Float32(data=(time.perf_counter() - t0) * 1e3))


def main():
    rclpy.init()
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
