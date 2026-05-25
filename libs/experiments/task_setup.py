import os
import sys

# IMPORTANT: Set habitat-sim env vars BEFORE importing habitat_sim
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import numpy as np
from pathlib import Path
from natsort import natsorted
import json
import pickle
import torch
import cv2
import csv
from datetime import datetime
from typing import Tuple
import torchvision.transforms as tfm
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

import habitat_sim

from libs.simulation.habitat_utils import get_sim_agent
from libs.experiments.episode_utils import (
    pick_random_start_state,
    select_trajectory_start_state,
    calculate_path_distance,
    find_shortest_path,
    initialize_results,
    write_results,
    write_final_meta_results
)
from libs.mapper.create_topomap import CostmapData
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.goal_generator.goal_gen import GoalGenerator
from libs.common.utils_sim import build_intrinsics, apply_velocity

from libs.control.learnt_controller import ObjRelLearntController

import logging
logger = logging.getLogger("[Task Setup]") # logger level is explicitly set below by LOG_LEVEL

class Episode:
    def __init__(self, cfg: DictConfig, episode_path, scene_glb_path, episode_results_path, preload_data={}):

        self.cfg = cfg
        self.steps = 0  # only used when running real in remote mode
        self.device = cfg.device
        self.H = self.cfg.sim.height
        self.W = self.cfg.sim.width

        self.resize_H = self.cfg.matcher.resize_h
        self.resize_W = self.cfg.matcher.resize_w
        if self.resize_H is not None and self.resize_W is not None:
            self.resize = (self.resize_H, self.resize_W)

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.episode_path = Path(episode_path)
        self.episode_results_path = Path(episode_results_path)
        logger.info(f"Running {self.episode_path=}...")

        self.scene_glb_path = scene_glb_path
        self.preload_data = preload_data
        self.episode_img_dir = episode_path / 'images'
        self.closest_map_img_idx = None
        
        self.results_folder_path = Path(episode_results_path)
        self.episode_vis_dir = self.results_folder_path / "vis"

        self.start_idx = cfg.start_idx
        self.loc_radius = self.cfg.localizer.loc_radius
        self.subsample_ref = self.cfg.localizer.subsample_ref

        # Resolve costmap path: {costmap_base_dir}/{episode_name}/{costmap_filename}
        # If costmap_base_dir is null, uses episode_path instead
        costmap_base_dir = cfg.get("costmap_base_dir", None)
        costmap_filename = cfg.costmap_filename
        episode_name = self.episode_path.name
        
        if costmap_base_dir:
            self.costmap_file_path = Path(costmap_base_dir) / episode_name / costmap_filename
        else:
            self.costmap_file_path = self.episode_path / costmap_filename
        
        if not self.costmap_file_path.exists():
            raise FileNotFoundError(f"Costmap file not found: {self.costmap_file_path}")
        
        logger.info(f"Loading costmap from: {self.costmap_file_path}")

        # Getting the Mapping costmap data
        self.costmap_data = CostmapData.from_npz(self.costmap_file_path)
        costmap_metadata = self.costmap_data.get_metadata()
        self.map_img_paths = costmap_metadata['image_paths']

        self.method = self.cfg.controller.name

        self.init_controller_params()

        self.setup_sim_agent()
        self.ready_agent()

        # robot intrinsics in simulator
        self.agent_intrinsics = build_intrinsics(
            image_width=self.W,
            image_height=self.H,
            field_of_view_radians_u=self.hfov_radians,
            device=self.device
        )

        # Get the goal generator
        self.get_goal_generator()

        # Set the controller
        self.set_controller()
        self.vis_img_default = np.zeros((self.H, self.W, 3)).astype(np.uint8)
    
    def init_controller_params(self):
        self.fov_deg = self.cfg.sim.hfov if 'robohop' in self.method.lower() else 79
        self.hfov_radians = np.pi * self.fov_deg / 180

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = np.nan

        self.pid_steer_values = [.25, 0, 0] if self.method.lower(
        ) == 'robohop+' else []
        self.discrete_action = -1
        self.controller_logs = None
    
    def set_controller(self):
        method_name = self.cfg.controller.name
        self.collided = None
        controller_cfg = self.cfg.controller

        if method_name == 'learnt':
            goal_controller = ObjRelLearntController(
                config=controller_cfg.config_file,
                goal_source=self.cfg.goal_source,
                boost_final_goal=controller_cfg.boost_final_goal   
            )
            goal_controller.reset_params()
            goal_controller.dirname_vis_episode = self.episode_vis_dir
        else:
            raise NotImplementedError("Other controller methods have not been implemented yet")
        
        self.goal_controller = goal_controller

    def setup_sim_agent(self):
        # Note: MAGNUM_LOG and HABITAT_SIM_LOG are set at module level before import
        sim_cfg = self.cfg.sim

        # Initialize Habitat Sim and Agent
        self.sim, self.agent, self.vel_control = get_sim_agent(
            scene_path=self.scene_glb_path,
            update_nav_mesh=sim_cfg.update_nav_mesh,
            width=sim_cfg.width,
            height=sim_cfg.height,
            hfov=sim_cfg.hfov,
            sensor_height=sim_cfg.sensor_height,
        )
        self.sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

        # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        self.vel_control = vel_control
    
    def ready_agent(self, goal_init_flag=True):

        # 1. Load agent trajectory
        agent_states_path = self.episode_path / 'agent_states.npy'
        if not agent_states_path.exists():
            raise FileNotFoundError(f"Agent states file not found: {agent_states_path}")

        self.agent_states = np.load(str(agent_states_path), allow_pickle=True)
        self.agent_positions_in_map = np.array([state.position for state in self.agent_states])

        # 2. Set goal based on task type
        self._set_goal_state()

        # 3. Select and set start state
        start_state = self._set_start_state()

        # 4. calculate distance metric
        self.distance_to_final_goal = calculate_path_distance(
            self.sim,
            self.start_position,
            self.final_goal_position,
        )
        self.agent_state_history = []

        logger.info(f"Agent ready: task_type={self.cfg.task_type}, "
            f"reverse={self.cfg.reverse}, "
            f"start_idx={self.start_idx}, "
            f"Start Position={self.start_position}, "
            f"Goal Position={self.final_goal_position}, "
            f"goal_distance={self.distance_to_final_goal:.2f}m")
    
    def _set_goal_state(self):
        """Set goal state based on task type"""
        if self.cfg.reverse:
            self._set_reverse_goal()
        elif self.cfg.task_type in ['alt_goal', 'alt_goal_v2']:
            raise NotImplementedError("Alt goal task not implemented yet.")
            # self._set_alt_goal()
        else:
            self._set_topological_goal()
    
    def _set_reverse_goal(self):
        """Set goal for reverse navigation task."""
        self.final_goal_state = self.agent_states[0]
        self.final_goal_position = self.final_goal_state.position
        self.final_goal_image_idx = len(self.agent_states) - 1

        logger.debug(f"Reverse goal set: image_idx={self.final_goal_image_idx}")
    
    # TODO: Fix this function later
    def _set_alt_goal(self):
        # TODO : This function needs to be fixed later
        metadata_path = self.episode_path / 'alt_goal_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f'Alt goal metadata not found: {metadata_path}')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.final_goal_image_idx = metadata['goal_image_idx']
        goal_instance_id = metadata['goal_instance_id']
        
        # Find instance position in scene
        instance_position = None
        for instance in self.sim.semantic_scene.objects:
            if instance.semantic_id == goal_instance_id:
                instance_position = instance.aabb.center
                break
        
        if instance_position is None:
            raise ValueError(f'Goal instance {goal_instance_id} not found in scene')
        
        # Snap to navigable surface at average floor height
        avg_floor_height = self.agent_positions_in_map[:, 1].mean()
        instance_position = np.array(instance_position, dtype=np.float32)
        instance_position[1] = avg_floor_height
        self.final_goal_position = self.sim.pathfinder.snap_point(instance_position)
        self.final_goal_state = None
        
        logger.debug(f"Alt goal set: instance_id={goal_instance_id}, "
                    f"image_idx={self.final_goal_image_idx}")

    def _set_topological_goal(self):
        """Set goal for topological graph-based navigation"""

        # Load the costmap file
        # costmap_data = CostmapData.from_npz(self.costmap_file_path)
        metadata = self.costmap_data.get_metadata()

        # get final goal position from costmap metadata
        self.final_goal_image_idx = metadata['goal_img_idx']
        self.goal_node_idx = metadata['goal_node_id']
        self.goal_px, self.goal_py = metadata['goal_pixel']
        self.goal_coord_3d = np.array(metadata['goal_coord_3d'], dtype=np.float32) # (3, )

        # Select goal position method
        goal_method = getattr(self.cfg, 'goal_position_method', 'trajectory')
        if goal_method == "trajectory":
            # Use agent position at goal image (simple, reliable)
            self.final_goal_position = np.array(
                self.agent_states[self.final_goal_image_idx].position, dtype=np.float32
            )
        else:  # "projection"
            # Project from 3D coords (may fail if not navigable)
            self.final_goal_position = self._compute_goal_position_from_3d_coords()
        
        self.final_goal_state = None
    
    def _compute_goal_position_from_3d_coords(self):
        """Project goal from image pixel + depth to navigable 3d position."""
        if self.goal_coord_3d.size < 3:
            raise ValueError(f"Invalid goal 3D coordinates for node {self.goal_node_idx}: {self.goal_coord_3d}")

        # goal_coord_3d is (x_px, y_px, z_depth). Use depth as a proxy for goal distance and
        # place the goal along the agent's forward direction at the goal image.
        depth = float(self.goal_coord_3d[2]) 

        # Get camera pose at goal image
        goal_agent_state = self.agent_states[self.final_goal_image_idx]
        camera_pos = np.array(goal_agent_state.position, dtype=np.float32)
        q = goal_agent_state.rotation

        # Extract forward direction from quaternion
        # Convert numpy-quaternion (w,x,y,z) to scipy format (x,y,z,w)
        q_scipy = np.array([q.x, q.y, q.z, q.w])
        R_mat = R.from_quat(q_scipy).as_matrix()
        forward = -R_mat[:, 2]
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Project and find navigable point
        search_depth = depth
        while search_depth > 0.1:
            candidate = camera_pos + forward * search_depth
            candidate[1] = camera_pos[1]  # Maintain floor height
            
            if self.sim.pathfinder.is_navigable(candidate):
                return self.sim.pathfinder.snap_point(candidate)
            
            search_depth -= 0.1
        
        raise ValueError(f'No navigable position found for node {self.goal_node_idx}')

    def _set_start_state(self):
        """
        Select and set the agent's starting state based on start_state_mode.
        
        Modes:
            - "random": Sample random navigable points with distance constraints
            - "trajectory": Select from recorded trajectory at target distance from goal
            - "fixed_idx": Use specific trajectory index (from start_idx config)
        """
        mode = getattr(self.cfg, 'start_state_mode', 'random')
        
        if mode == "random":
            # Random start state with distance constraints
            start_state = pick_random_start_state(
                sim=self.sim,
                cfg=self.cfg,
                final_goal_position=self.final_goal_position,
                agent_positions_in_map=self.agent_positions_in_map,
                max_tries=100
            )
            logger.debug(f"Random start state selected")
            
        elif mode == "trajectory":
            # Select from recorded trajectory at target distance from goal
            start_state = select_trajectory_start_state(
                sim=self.sim,
                cfg=self.cfg,
                agent_states=self.agent_states,
                goal_position=self.final_goal_position
            )
            logger.debug(f"Trajectory-based start state selected")
            
        elif mode == "fixed_idx":
            # Use specific trajectory index
            if self.start_idx >= len(self.agent_states):
                raise ValueError(f"start_idx {self.start_idx} out of range "
                               f"(trajectory has {len(self.agent_states)} states)")
            start_state = self.agent_states[self.start_idx]
            logger.debug(f"Fixed index start state: idx={self.start_idx}")
            
        else:
            raise ValueError(f"Unknown start_state_mode: {mode}. "
                           f"Expected: random, trajectory, fixed_idx")
        
        if start_state is None:
            raise ValueError(f'Could not find valid start state for {self.episode_path}')
        
        self.agent.set_state(start_state)
        self.start_position = start_state.position
        
        logger.debug(f"Start state set: mode={mode}, position={self.start_position}")
        
        return start_state
    
    def get_goal_generator(self):

        # setup the matcher
        if self.cfg.matcher.name == "mast3r":
            self.matcher = Mast3rMatcher(
                resize_w=self.cfg.matcher.resize_w,
                resize_h=self.cfg.matcher.resize_h,
                geometric_verification=self.cfg.matcher.geometric_verification,
                subsample_or_initxy1=self.cfg.matcher.subsample_or_initxy1,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown matcher: {self.cfg.matcher.name}")
        
        # setup the localizer
        if self.cfg.localizer.name == "topological":
            self.localizer = LocalizeTopological(
                map_img_paths=self.map_img_paths,
                H=self.resize_H,
                W=self.resize_W,
                matcher=self.matcher,
                cfg=self.cfg.localizer
            )
        else:
            raise ValueError(f"Unknown localizer: {self.cfg.localizer.name}")

        # setup the planner
        if self.cfg.planner.name == "topological":
            self.planner = PlanTopological(
                H=self.resize_H,
                W=self.resize_W,
                costmap_data=self.costmap_data,
                device=self.device,
                cfg=self.cfg.planner
            )
        else:
            raise ValueError(f"Unknown planner: {self.cfg.planner.name}")

        # setup the goal generator finally
        if self.cfg.goal_source == "topological_pixelwise":
            self.goal_generator = GoalGenerator(
                H=self.resize_H,
                W=self.resize_W,
                localizer=self.localizer,
                planner=self.planner,
                cfg=self.cfg
            )
        else:
            raise ValueError(f"Unknown goal generator: {self.cfg.goal_generator.name}")
    
    def get_goal(self, rgb, depth, pose=None, pts3d=None, return_vis_data=False):
        """
        Get goal mask for the current observation.
        
        Args:
            rgb: RGB observation (H, W, 3)
            depth: Depth map (H, W)
            pose: Agent pose (optional)
            pts3d: 3D points (H, W, 3) (optional)
            return_vis_data: If True, return (goal_mask, vis_data) tuple
            
        Returns:
            goal_mask: Distance-to-goal costmap (H, W)
            OR
            (goal_mask, vis_data): If return_vis_data=True, dict contains match data
        """
        # Getting the closest reference image to the particular query image
        if self.cfg.localizer.use_gt_localization:
            if pose is not None:
                localized_img_idxs, closest_map_img_idx = self.get_closest_map_img_from_odometry()
            else:
                localized_img_idxs, closest_map_img_idx = self.get_gt_closest_map_img()
                closest_map_img_idx = self._select_min_median_path_length(localized_img_idxs)[0]
        else:
            localized_img_idxs, closest_map_img_idx = self.get_visual_closest_map_img()
        
        # Getting the goal mask for current observation
        result = self.goal_generator.get_goal_mask(
            qry_img=rgb,
            qry_depth=depth,
            qry_pts3d=pts3d,
            intrinsics=self.agent_intrinsics,
            candidate_img_indices=[closest_map_img_idx],
            return_vis_data=return_vis_data
        )

        if return_vis_data:
            self.goal_mask, vis_data = result
            self.control_input_learnt = self.goal_mask
            self.control_input_robohop = self.goal_mask
            # Add localization metadata
            vis_data['localized_img_idxs'] = localized_img_idxs  # Candidate submap
            vis_data['closest_map_img_idx'] = closest_map_img_idx  # Best match from submap
            return self.goal_mask, vis_data
        else:
            self.goal_mask = result
            self.control_input_learnt = self.goal_mask
            self.control_input_robohop = self.goal_mask
            return self.goal_mask

    def get_gt_closest_map_img(self):
        dists = np.linalg.norm(
            self.agent_positions_in_map - self.agent.get_state().position, axis=1)
        
        top_k = 2 * self.loc_radius
        closest_idxs = np.argsort(dists)[:top_k]
        closest_idxs = sorted(closest_idxs)[::self.subsample_ref]
        logger.info(f"Top K closest idxs: {closest_idxs = }")
        closest_idx = np.argmin(dists)
        return closest_idxs, closest_idx

    def get_closest_map_img_from_odometry(self, odom_pose, episode_path, position_weight=1.0, rotation_weight=1.0):
        """
        Given a robot odometry pose (x, y, z, qx, qy, qz, qw), find the image index in poses_odom.txt
        that is closest to this pose using both translation and rotation.

        Returns:
            closest_idx: int, index of the closest image
            localized_img_inds: list of indices, sorted by combined distance (topK, subsampled)
        """
        # Path to odometry file
        odom_file = Path(episode_path) / 'poses_odom.txt'
        if not odom_file.exists():
            raise FileNotFoundError(f"{odom_file} does not exist.")

        # Load odometry poses
        poses = []
        quats = []
        with open(odom_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append([x, y, z])
                quats.append([qx, qy, qz, qw])
        poses = np.array(poses)  # shape (N, 3)
        quats = np.array(quats)  # shape (N, 4)

        pos_query = np.array(odom_pose[:3])
        quat_query = np.array(odom_pose[3:7])

        trans_dists = np.linalg.norm(poses - pos_query, axis=1)

        def quat_angle(q1, q2):
            dot = np.abs(np.sum(q1 * q2, axis=-1))
            dot = np.clip(dot, -1.0, 1.0)
            return 2 * np.arccos(dot)  # angle in radians

        rot_dists = quat_angle(quats, quat_query)
        total_dists = position_weight * trans_dists + rotation_weight * rot_dists

        closest_idx = int(np.argmin(total_dists))
        # Sort all indices by distance, take topK and subsample
        topK = 2 * self.loc_radius
        sorted_idxs = np.argsort(total_dists)[:topK]
        localized_img_idxs = sorted(sorted_idxs.tolist())[::self.subsample_ref]
        
        localized_img_idxs = self._select_min_median_path_length(localized_img_idxs)
        closest_idx = localized_img_idxs[0]
        
        return localized_img_idxs, closest_idx
    
    # TODO : clean this function up later
    def get_visual_closest_map_img(self, rgb):
        # TODO : improve this function later and add things to take from config
        current_image = rgb[:, :, :3]
        match_scores = []

        # get candidate map image indices
        candidate_imgs = os.listdir(self.episode_img_dir)
        candidate_imgs = natsorted(candidate_imgs)
        candidate_idxs = list(range(len(candidate_imgs)))
        for idx in candidate_idxs[::5]:
            img_file = candidate_imgs[idx]
            map_image = cv2.imread(str(self.episode_img_dir / img_file))[:, :, ::-1]

            # Fix: ensure images have positive strides before passing to load_image
            qry_img = self.matcher.load_image(current_image.copy())
            ref_img = self.matcher.load_image(map_image.copy())
            result = self.matcher(qry_img, ref_img)
            inlier_count = result['num_inliers']
            logger.info(f"Mast3R Found {inlier_count} inliers with Reference Image {idx}")

            match_scores.append({'idx': idx, 'inliers': inlier_count})

        # 3. Sort candidates by the number of inliers (descending)
        top_k = 2 * self.loc_radius
        sorted_matches = sorted(match_scores, key=lambda x: x['inliers'], reverse=True)[:top_k]

        # 4. Select the best goal and the list of localized images
        # The best goal is the one with the most inliers.
        best_goal_idx = sorted_matches[0]['idx']

        # The localized images are all the ones that had enough inliers, sorted by score.
        localized_img_idxs = [match['idx'] for match in sorted_matches][::self.subsample_ref]

        localized_img_idxs = self._select_min_median_path_length(localized_img_idxs)
        closest_idx = localized_img_idxs[0]
        
        return localized_img_idxs, best_goal_idx

    def _select_min_median_path_length(self, candidate_img_indices):
        min_median_path_length = 100 # Max Value

        best_ref_img_idx = None
        img_costmaps = self.costmap_data.get_costmap()
        for ref_img_idx in candidate_img_indices:
            # (H, W)
            img_pls = img_costmaps[ref_img_idx]

            median_path_length = np.median(img_pls)
            logger.info(f"FOUND Image {ref_img_idx} has {median_path_length} median path length")
            if median_path_length < min_median_path_length:
                min_median_path_length = median_path_length
                best_ref_img_idx = ref_img_idx
        
        if best_ref_img_idx is not None:
            logger.info(f"Selected Image {best_ref_img_idx} with {min_median_path_length} median path length")
            return [best_ref_img_idx]
        else:
            logger.warning(f"No good matches found, using the 0th index image: {candidate_img_indices[0] = }")
            return [candidate_img_indices[0]]
    
    def get_control_signal(self, step, rgb, depth):
        control_method = self.cfg.controller.name
        if control_method == 'learnt':
            if self.control_input_learnt[0] is None or self.control_input_learnt[1] is None:
                self.velocity_control, self.theta_control, self.vis_img = 0, 0, self.vis_img_default.copy()
            else:
                # import pdb; pdb.set_trace()
                self.velocity_control, self.theta_control, self.vis_img = self.goal_controller.predict(
                    rgb, self.control_input_learnt)
            
            self.controller_logs = self.goal_controller.controller_logs
            # NOTE: In simulation, theta is NOT negated here.
            # It's only negated for real robot (env != 'sim').
            # The negation for sim happens in execute_action with steer=-self.theta_control
        else:
            raise NotImplementedError(f"{control_method} is not available...")
    
    def execute_action(self):
        control_method = self.cfg.controller.name
        if control_method == 'learnt':
            self.agent, self.sim, self.collided = apply_velocity(
                vel_control=self.vel_control,
                agent=self.agent,
                sim=self.sim,
                velocity=self.velocity_control,
                steer=-self.theta_control,  # opposite y axis
                time_step=self.time_delta
            )  # will add velocity control once steering is working
        else:
            raise NotImplementedError("Other controller methods task not implemented yet.")
        
        self.agent_state_history.append(self.agent.get_state())
    
    def is_done(self):
        done = False
        current_robot_state = self.agent.get_state()  # world coordinates
        self.distance_to_goal = find_shortest_path(
            self.sim, p1=current_robot_state.position, p2=self.final_goal_position)[0]
        if self.distance_to_goal <= self.cfg.goal_distance_threshold:
            logger.info(
                f'\nWinner! dist to goal: {self.distance_to_goal:.6f}\n')
            self.success_status = 'success'
            done = True
        return done
    
    def setup_logging(self):
        self.episode_metadata_filepath = self.episode_results_path / 'metadata.txt'
        self.episode_results_csv = self.episode_results_path / 'results.csv'

        # Initialize results files
        initialize_results(
            metadata_file=self.episode_metadata_filepath,
            results_csv=self.episode_results_csv,
            method=self.cfg.controller.name,
            goal_source=self.cfg.goal_source,
            max_steps=self.cfg.max_steps,
            goal_distance_threshold=self.cfg.goal_distance_threshold,
            pid_steer_values=self.pid_steer_values,
            hfov_degrees=self.fov_deg,
            time_delta=self.time_delta,
            velocity_control=self.velocity_control,
            goal_position=self.final_goal_position,
        )

        # Initialize results dictionary for accumulating per-step data
        results_dict_keys = [
            "step",
            "distance_to_goal",
            "velocity_control",
            "theta_control",
            "collided",
            "discrete_action",
            "agent_states",
            "controller_logs",
        ]
        self.results_dict = {k: [] for k in results_dict_keys}

    def log_results(self, step: int, final: bool = False) -> None:
        """
        Log per-step or final results to files.
        
        Args:
            step: Current step number
            final: If True, write final metadata and save results_dict.npz
        """
        if not final:
            # Write per-step results to CSV
            write_results(
                results_csv=self.episode_results_csv,
                step=step,
                current_robot_state=self.agent.get_state() if self.agent is not None else None,
                distance_to_goal=self.distance_to_goal,
                velocity_control=self.velocity_control,
                theta_control=self.theta_control,
                collided=self.collided,
                discrete_action=self.discrete_action
            )
            
            # Accumulate results in results_dict
            results_dict_curr = {
                "step": step,
                "distance_to_goal": self.distance_to_goal,
                "velocity_control": self.velocity_control,
                "theta_control": self.theta_control,
                "collided": self.collided,
                "discrete_action": self.discrete_action,
                "agent_states": self.agent.get_state() if self.agent is not None else None,
                "controller_logs": self.controller_logs[-1] if self.controller_logs is not None and len(self.controller_logs) > 0 else None,
            }
            self.update_results_dict(results_dict_curr)
        else:
            # Write final metadata
            write_final_meta_results(
                metadata_file=self.episode_metadata_filepath,
                success_status=self.success_status,
                final_distance=self.distance_to_goal,
                step=step,
                distance_to_final_goal=self.distance_to_final_goal
            )
            
            # Save accumulated results dictionary as npz
            np.savez(
                self.episode_results_path / 'results_dict.npz',
                **self.results_dict
            )
    
    def update_results_dict(self, curr_dict: dict) -> None:
        """
        Append current step's data to the results dictionary.
        
        Args:
            curr_dict: Dictionary with current step's data
        """
        for k, v in curr_dict.items():
            self.results_dict[k].append(v)

def init_results_dir_and_save_cfg(cfg: DictConfig, default_logger=None):
    # Build results path from config
    results_path = Path(cfg.results_dirpath) if cfg.results_dirpath.startswith('/') else Path.cwd() / cfg.results_dirpath

    # Create structured folder path
    task_str = cfg.task_type
    if cfg.get('reverse', False):
        task_str += '_reverse'
    
    results_dirpath = (results_path / task_str / cfg.exp_name /
    f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{cfg.controller.name}_{cfg.goal_source}')
    results_dirpath.mkdir(exist_ok=True, parents=True)

    # Update logger file handler
    if default_logger is not None:
        default_logger.update_file_handler_root(results_dirpath / 'output.log')
    
    logger.info(f'Logging to {results_dirpath}')

    # Save config
    OmegaConf.save(cfg, results_dirpath / 'config.yaml')
    return results_dirpath