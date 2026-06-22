#!/usr/bin/env python3
"""
masternav_runner.py
Real-robot navigation node for MASt3RNav on the P3DX.

Architecture:
  RealSense /camera/color/image_raw
      ↓
  [Localization]  libs.goal_generators  →  WayPixel costmap
      ↓
  [Controller]    libs.control          →  10 BEV waypoints
      ↓
  [Velocity]      last waypoint → (v, ω)  with 5-frame smoothing
      ↓
  /RosAria/cmd_vel  (geometry_msgs/Twist)

Run OUTSIDE the pixi env is not possible — this script must be launched
from inside the mast3r-nav pixi environment. See launch file comments.

Prerequisites (run once, offline):
  1. pixi run python -m libs.mapper.create_topomap  (builds the map)
  2. Costmaps (.npz) generated alongside the map output
"""

import os
import sys
import numpy as np
from collections import deque
from pathlib import Path

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
# Path setup — mast3r-nav must be cloned somewhere accessible.
# Set MASTERNAV_ROOT env var or update the path below.
# ---------------------------------------------------------------------------
# __file__ is masternav/scripts/masternav_runner.py
# go up: scripts/ → masternav/ → repo root → mast3r-nav/
MASTERNAV_ROOT = os.environ.get(
    'MASTERNAV_ROOT',
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'mast3r-nav')
    )
)
sys.path.insert(0, MASTERNAV_ROOT)

# These imports only work inside the pixi environment
from omegaconf import OmegaConf
from libs.goal_generators import WayPixelGoalGenerator   # localize → costmap
from libs.control.gnm_controller import GNMController    # costmap → waypoints


class MASt3RNavRunner:
    def __init__(self):
        rospy.init_node('masternav_runner')

        # ---------------------------------------------------------------
        # Load config — reuse the repo's Hydra configs directly
        # ---------------------------------------------------------------
        config_path = rospy.get_param(
            '~config_path',
            os.path.join(MASTERNAV_ROOT, 'configs', 'config.yaml')
        )
        self.cfg = OmegaConf.load(config_path)

        # Override paths from ROS params so you don't edit the yaml
        map_dir      = rospy.get_param('~map_dir')       # topomap output dir
        costmap_path = rospy.get_param('~costmap_path')  # .npz file
        model_ckpt   = rospy.get_param('~model_ckpt',
                           os.path.join(MASTERNAV_ROOT,
                                        'checkpoints/gnm_mast3r_nav/latest.pth'))

        # ---------------------------------------------------------------
        # Initialise the two core components from mast3r-nav libs
        # ---------------------------------------------------------------
        rospy.loginfo("[MASt3RNavRunner] Loading map and costmaps...")
        self.goal_gen = WayPixelGoalGenerator(
            cfg=self.cfg,
            map_dir=Path(map_dir),
            costmap_path=Path(costmap_path),
        )

        rospy.loginfo("[MASt3RNavRunner] Loading controller checkpoint...")
        self.controller = GNMController(
            cfg=self.cfg.controller,
            ckpt_path=model_ckpt,
            device='cuda' if rospy.get_param('~use_gpu', True) else 'cpu',
        )

        # ---------------------------------------------------------------
        # Velocity params (tune for P3DX)
        # ---------------------------------------------------------------
        self.v_min = rospy.get_param('~v_min', 0.10)   # m/s floor
        self.v_max = rospy.get_param('~v_max', 0.30)   # m/s ceiling
        self.w_max = rospy.get_param('~w_max', 0.50)   # rad/s ceiling
        self.v_gain = rospy.get_param('~v_gain', 0.5)  # dist→v scale
        self.w_gain = rospy.get_param('~w_gain', 1.0)  # angle→ω scale

        # 5-frame smoothing window (matches paper)
        self.v_window = deque(maxlen=5)
        self.w_window = deque(maxlen=5)

        # Observation history (GNM uses last 5 frames concatenated)
        self.obs_history = deque(maxlen=6)  # current + 5 past

        # ---------------------------------------------------------------
        # ROS pub/sub
        # NOTE: rosaria publishes/subscribes on /RosAria/cmd_vel,
        #       NOT /cmd_vel — this is the most common P3DX gotcha.
        # ---------------------------------------------------------------
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher(
            '/RosAria/cmd_vel', Twist, queue_size=1
        )
        rospy.Subscriber(
            '/camera/color/image_raw', Image, self.img_callback,
            queue_size=1, buff_size=2**24
        )

        rospy.loginfo("[MASt3RNavRunner] Ready. Publishing to /RosAria/cmd_vel")
        rospy.on_shutdown(self.stop_robot)
        rospy.spin()

    # -------------------------------------------------------------------
    # Main callback — runs at camera frame rate
    # -------------------------------------------------------------------
    def img_callback(self, msg):
        # Convert ROS Image → numpy RGB (H, W, 3) uint8
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = frame_bgr[:, :, ::-1].copy()   # BGR→RGB

        # Keep observation history (GNM encoder uses past 5 frames)
        self.obs_history.append(rgb)
        if len(self.obs_history) < 2:
            return  # need at least 2 frames before running inference

        # ---- Step 1: Localize + build WayPixel costmap ----------------
        # get_goal() matches current frame against the pre-built topomap,
        # looks up path lengths, and returns a dense (H, W, 8) costmap.
        try:
            costmap = self.goal_gen.get_goal(
                rgb=rgb,
                depth=None,   # MASt3RNav uses monocular depth internally
                pts3d=None,   # set to None → uses GT/estimated depth path
            )
        except Exception as e:
            rospy.logwarn(f"[MASt3RNavRunner] get_goal failed: {e}")
            self._publish_stop()
            return

        if costmap is None:
            rospy.logwarn("[MASt3RNavRunner] No goal localized, stopping.")
            self._publish_stop()
            return

        # ---- Step 2: Controller → BEV waypoints -----------------------
        # GNMController takes obs_image (current RGB) + goal_image (costmap)
        # and returns action_pred: (10, 4) — (x, y, cos_yaw, sin_yaw)
        obs_stack = list(self.obs_history)   # list of RGB arrays
        try:
            action_pred = self.controller.predict(
                obs_images=obs_stack,
                goal_image=costmap,
            )
        except Exception as e:
            rospy.logwarn(f"[MASt3RNavRunner] predict failed: {e}")
            self._publish_stop()
            return

        # ---- Step 3: Waypoint → velocity ------------------------------
        # Use last waypoint (index -1) — this is what the paper does
        last_wp = action_pred[-1]   # (x, y, cos_yaw, sin_yaw) in robot BEV
        dx, dy = float(last_wp[0]), float(last_wp[1])

        dist  = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)   # heading to waypoint in robot frame

        v = float(np.clip(dist * self.v_gain, self.v_min, self.v_max))
        w = float(np.clip(angle * self.w_gain, -self.w_max, self.w_max))

        # ---- Step 4: 5-frame moving average ---------------------------
        self.v_window.append(v)
        self.w_window.append(w)
        v_cmd = float(np.mean(self.v_window))
        w_cmd = float(np.mean(self.w_window))

        # ---- Step 5: Publish ------------------------------------------
        twist = Twist()
        twist.linear.x  = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

        rospy.logdebug(
            f"[MASt3RNavRunner] wp=({dx:.3f},{dy:.3f}) "
            f"v={v_cmd:.3f} w={w_cmd:.3f}"
        )

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())   # zero twist = stop

    def stop_robot(self):
        rospy.loginfo("[MASt3RNavRunner] Shutting down — stopping robot.")
        self._publish_stop()


if __name__ == '__main__':
    MASt3RNavRunner()