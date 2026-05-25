"""
run_nav.py - Minimal Navigation Test Script for mast3r-nav

A clean, Hydra-based script to test the navigation pipeline in simulation.
"""

import os
import sys

# Suppress habitat-sim logs - MUST be before importing habitat_sim
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# CRITICAL: Import habitat_sim FIRST before numpy, torch, scipy, etc.
import habitat_sim

# Now import everything else
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from libs.experiments import task_setup
from libs.logger import default_logger
from libs.mast3r_utils import MASt3RInference
from libs.experiments.episode_utils import check_if_stuck
from libs.visualizations import VisualizationDataCollector, VisualizationRenderer
from libs.common.gpu_memory_utils import clear_gpu_cache

# Setup logging
default_logger.setup_logging(level=logging.INFO, console=True)
logger = logging.getLogger("[RunNav]")

# ==============================================================================
# Utility Functions
# ==============================================================================

def split_observations(observations: dict) -> tuple:
    """
    Extract RGB, depth, and semantic from simulator observations.
    
    Returns:
        rgb: (H, W, 3) uint8 array
        depth: (H, W) float32 array
        semantic: (H, W) int32 array or None
    """
    rgb = observations["color_sensor"][:, :, :3]  # Drop alpha channel
    depth = observations["depth_sensor"]
    # semantic = observations.get("semantic_sensor", None)
    return rgb, depth

# ==============================================================================
# Episode Runner
# ==============================================================================

def run_episode(cfg: DictConfig, episode_path: Path, episode_results_path: Path, 
                mast3r_model=None) -> dict:
    """
    Run a single navigation episode.
    
    Args:
        cfg: Hydra config
        episode_path: Path to episode data
        episode_results_path: Path to save results
        mast3r_model: Optional pre-loaded MASt3R model (for pts3d_source="mast3r")
    
    Returns:
        dict with episode results (success_status, steps, distance_to_goal)
    """
    # Construct scene path
    hm3d_root_path = Path(cfg.hm3d_root_path)
    episode_name = str(episode_path.parts[-1].split('_')[0])
    hm3d_scene_path = sorted(hm3d_root_path.glob(f"*{episode_name}"))[0]
    scene_glb_path = str(sorted(hm3d_scene_path.glob('*basis.glb'))[0])

    logger.info(f"Scene GLB PATH: {scene_glb_path} | {os.path.exists(scene_glb_path) = }")
    
    # Create episode runner
    logger.info(f"Initializing episode: {episode_path}")
    episode = task_setup.Episode(
        cfg=cfg,
        episode_path=episode_path,
        scene_glb_path=scene_glb_path,
        episode_results_path=episode_results_path,
        preload_data={}
    )
    
    # Setup logging directories
    episode.setup_logging()
    
    # Initialize visualization system if enabled
    data_collector = None
    vis_renderer = None
    if cfg.visualization.save_raw_data.enabled:
        start_pos = episode.agent.get_state().position
        goal_pos = episode.final_goal_position
        
        data_collector = VisualizationDataCollector(
            episode_results_path=episode_results_path,
            cfg=cfg,
            start_position=start_pos,
            goal_position=goal_pos
        )
        
        vis_renderer = VisualizationRenderer(
            vis_cfg=cfg.visualization,
            episode_results_path=episode_results_path
        )
        
        # Save topdown base map and metadata for offline rendering
        meters_per_pixel = getattr(cfg.visualization.render_visualizations, 'topdown_meters_per_pixel', 0.025)
        data_collector.save_topdown_data(
            sim=episode.sim,
            start_position=np.array(start_pos),
            goal_position=np.array(goal_pos),
            meters_per_pixel=meters_per_pixel
        )
        
        logger.info("Raw data collector and visualization renderer initialized")
    
    # Navigation loop
    step = 0
    max_steps = cfg.max_steps
    stuck_threshold = cfg.get("stuck_threshold", 0.01)
    stuck_window = cfg.get("stuck_window", 15)
    pts3d_source = cfg.get("pts3d_source", "gt_depth")
    
    logger.info(f"Starting navigation loop (max_steps={max_steps}, pts3d_source={pts3d_source})")
    
    pbar = tqdm(total=max_steps, desc="Navigation", leave=False)
    
    while step < max_steps:
        step_start = time.time()
        
        # Check for stuck condition
        # if check_if_stuck(episode.agent_state_history, stuck_threshold, stuck_window):
        #     logger.warning(f"Episode failed: Agent stuck (little movement in last {stuck_window} steps)")
        #     episode.success_status = "stuck_no_movement"
        #     break
        
        # Check if goal reached
        if episode.is_done():
            logger.info(f"Episode succeeded at step {step}!")
            break
        
        # 1. Get sensor observations
        observations = episode.sim.get_sensor_observations()
        rgb, depth_gt = split_observations(observations)
        
        # 2. Get pts3d based on source
        if pts3d_source == "mast3r" and mast3r_model is not None:
            pts3d = mast3r_model.get_pts3d(rgb)
            depth = None
            logger.debug(f"Step {step}: Got pts3d from MASt3R, shape={pts3d.shape}")
        else:
            # Use GT depth, pts3d will be computed inside goal_generator from depth
            pts3d = None
            depth = depth_gt
        
        # 3. Get goal mask (localization + planning)
        if cfg.visualization.save_raw_data.enabled and data_collector is not None:
            goal_mask, vis_data = episode.get_goal(
                rgb=rgb, depth=depth, pose=None, pts3d=pts3d, return_vis_data=True
            )
        else:
            goal_mask = episode.get_goal(rgb=rgb, depth=depth, pose=None, pts3d=pts3d)
            vis_data = None
        
        # 4. Get control signal
        episode.get_control_signal(step, rgb, depth)
        
        # 5. Execute action in simulator
        episode.execute_action()
        
        # 6. Save raw data if enabled
        if cfg.visualization.save_raw_data.enabled and data_collector is not None and vis_data is not None:
            # Prepare matches data for saving
            matches_data = {
                'qry_img_idx': step,
                'closest_map_img_idx': vis_data['closest_map_img_idx'],
                'localized_img_idxs': vis_data.get('localized_img_idxs', np.array([])),
                'qry_mkpts': vis_data['qry_mkpts'],
                'ref_mkpts': vis_data['ref_mkpts'],
                'confidences': vis_data['confidences']
            }
            
            # Extract waypoints from controller if available
            waypoints = None
            if hasattr(episode, 'goal_controller') and hasattr(episode.goal_controller, 'action_pred'):
                if episode.goal_controller.action_pred is not None:
                    waypoints = episode.goal_controller.action_pred  # (N, 2) predicted waypoints
            
            # Get agent state for saving
            agent_state_obj = episode.agent.get_state()
            agent_state = {
                'position': np.array(agent_state_obj.position),
                'rotation': np.array([agent_state_obj.rotation.w, agent_state_obj.rotation.x, 
                                      agent_state_obj.rotation.y, agent_state_obj.rotation.z])
            }
            
            # Save raw data
            data_collector.save_step_data(
                step=step,
                rgb=rgb,
                depth=depth if cfg.pts3d_source != "mast3r" else None,
                pts3d=pts3d,
                costmap=episode.goal_mask,
                matches_data=matches_data,
                velocity=episode.velocity_control,
                theta=episode.theta_control,
                waypoints=waypoints,
                agent_state=agent_state,
                collided=episode.collided,
                distance_to_goal=episode.distance_to_goal
            )
            
            # Optionally render visualizations online
            if vis_renderer is not None and vis_renderer.online_render:
                ref_img_path = episode.map_img_paths[vis_data['closest_map_img_idx']]
                vis_renderer.render_step_visualizations(
                    step=step,
                    rgb=rgb,
                    costmap=episode.goal_mask,
                    matches_data=matches_data,
                    ref_img_path=ref_img_path,
                    sim=episode.sim,
                    trajectory_history=episode.agent_state_history,
                    start_position=np.array(episode.start_position),
                    goal_position=np.array(episode.final_goal_position)
                )
        
        # 7. Log step results to CSV
        episode.log_results(step, final=False)
        
        # Log progress
        step_time = time.time() - step_start
        pbar.set_postfix({
            "dist": f"{episode.distance_to_goal:.2f}m",
            "v": f"{episode.velocity_control:.3f}",
            "w": f"{episode.theta_control:.3f}",
            "t": f"{step_time:.2f}s"
        })
        pbar.update(1)
        
        step += 1
        clear_gpu_cache()
    
    pbar.close()
    
    # Finalize
    if step >= max_steps:
        episode.success_status = "exceeded_steps"
        logger.warning(f"Episode exceeded max steps ({max_steps})")
    
    # Log final results to metadata file
    episode.log_results(step, final=True)
    
    # Save episode visualization metadata if enabled
    if cfg.visualization.save_raw_data.enabled and data_collector is not None:
        data_collector.save_episode_metadata(
            success_status=episode.success_status,
            total_distance=episode.distance_to_final_goal,
            final_distance_to_goal=episode.distance_to_goal
        )
    
    # Save results
    results = {
        "success_status": episode.success_status,
        "steps": step,
        "distance_to_goal": episode.distance_to_goal,
        "distance_to_final_goal": episode.distance_to_final_goal,
    }
    
    # Close simulator
    episode.sim.close()
    
    return results


# ==============================================================================
# Episode Discovery
# ==============================================================================

def get_episode_list(cfg: DictConfig) -> list:
    """
    Get list of episode paths based on config.
    
    Priority order:
    1. If episode_list is non-empty: use only those specific episodes
    2. If multi_episode is true: get episodes from episodes_dir with start/end filtering
    3. Otherwise: use single episode_path
    
    Args:
        cfg: Hydra config with episode_path, episodes_dir, multi_episode, etc.
        
    Returns:
        List of Path objects for episodes to process
    """
    from natsort import natsorted
    
    # Single episode mode - just use episode_path
    if not cfg.multi_episode:
        episode_path = Path(cfg.episode_path)
        if not episode_path.exists():
            raise ValueError(f"Episode path does not exist: {episode_path}")
        return [episode_path]
    
    # Multi-episode mode
    episodes_dir = Path(cfg.episodes_dir) if cfg.get("episodes_dir") else None
    if not episodes_dir or not episodes_dir.exists():
        raise ValueError(f"Episodes directory does not exist: {episodes_dir}")
    
    # Priority 1: episode_list_file (txt file with episode names)
    episode_list_file = cfg.get("episode_list_file", None)
    if episode_list_file:
        file_path = Path(episode_list_file)
        if not file_path.exists():
            raise ValueError(f"Episode list file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            episode_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(episode_names)} episodes from {file_path}")
        
        episodes = []
        for ep_name in episode_names:
            ep_path = episodes_dir / ep_name
            if ep_path.exists():
                episodes.append(ep_path)
            else:
                logger.warning(f"Episode not found: {ep_path}")
        
        return natsorted(episodes, key=lambda x: x.name)
    
    # Priority 2: episode_list array (if non-empty)
    episode_list = cfg.get("episode_list", [])
    if episode_list and len(episode_list) > 0:
        logger.info(f"Using episode_list with {len(episode_list)} episodes")
        
        episodes = []
        for ep_name in episode_list:
            ep_path = episodes_dir / ep_name
            if ep_path.exists():
                episodes.append(ep_path)
            else:
                logger.warning(f"Episode not found: {ep_path}")
        
        return natsorted(episodes, key=lambda x: x.name)
    
    # Priority 3: All episodes from directory with start/end filtering
    all_episodes = [p for p in episodes_dir.iterdir() if p.is_dir()]
    all_episodes = natsorted(all_episodes, key=lambda x: x.name)
    
    logger.info(f"Found {len(all_episodes)} episodes in {episodes_dir}")
    
    # Apply start/end index filtering
    start_idx = cfg.get("episode_start_idx", 0)
    end_idx = cfg.get("episode_end_idx", -1)
    
    if start_idx > 0:
        all_episodes = all_episodes[start_idx:]
    if end_idx > 0:
        all_episodes = all_episodes[:end_idx - start_idx]
    
    logger.info(f"After filtering (start={start_idx}, end={end_idx}): {len(all_episodes)} episodes")
    
    # Apply blacklist filtering
    blacklist = cfg.get("episode_blacklist", [])
    if blacklist:
        episodes = [ep for ep in all_episodes 
                   if not any(bl in ep.name for bl in blacklist)]
        logger.info(f"After blacklist filtering: {len(episodes)} episodes")
    else:
        episodes = all_episodes
    
    return episodes


# ==============================================================================
# Main Entry Point
# ==============================================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    
    # Print config
    logger.info("=" * 60)
    logger.info("mast3r-nav Navigation Test")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load MASt3R model if needed (load once, reuse across episodes)
    mast3r_model = None
    if cfg.get("pts3d_source", "gt_depth") == "mast3r":
        mast3r_model = MASt3RInference(device=cfg.device)
    
    # Initialize results directory
    results_path = task_setup.init_results_dir_and_save_cfg(cfg, default_logger)
    logger.info(f"Results will be saved to: {results_path}")
    
    # Get episodes to process
    try:
        episodes = get_episode_list(cfg)
    except ValueError as e:
        logger.error(str(e))
        return
    
    logger.info(f"Processing {len(episodes)} episode(s)")
    
    # Results tracking
    results_summary = {
        'total_episodes': len(episodes),
        'successful_episodes': 0,
        'failed_episodes': 0,
        'exceeded_steps': 0,
        'stuck': 0,
        'success_rate': 0.0,
        'episode_results': []
    }
    
    # Process each episode
    for ei, episode_path in enumerate(tqdm(episodes, desc="Processing Episodes")):
        episode_name = episode_path.parts[-1]
        
        logger.info("=" * 60)
        logger.info(f"Episode {ei+1}/{len(episodes)}: {episode_name}")
        logger.info("=" * 60)
        
        # Create episode results directory
        episode_results_path = results_path / f"{episode_name}_{cfg.controller.name}_{cfg.goal_source}"
        episode_results_path.mkdir(exist_ok=True, parents=True)
        
        try:
            # Run episode
            results = run_episode(cfg, episode_path, episode_results_path, mast3r_model)
            
            # Track results
            results['episode_name'] = episode_name
            results_summary['episode_results'].append(results)
            
            if results['success_status'] == 'success':
                results_summary['successful_episodes'] += 1
            elif results['success_status'] == 'exceeded_steps':
                results_summary['exceeded_steps'] += 1
                results_summary['failed_episodes'] += 1
            elif 'stuck' in results['success_status']:
                results_summary['stuck'] += 1
                results_summary['failed_episodes'] += 1
            else:
                results_summary['failed_episodes'] += 1
            
            # Save individual episode results
            results_file = episode_results_path / "results_summary.txt"
            with open(results_file, "w") as f:
                f.write(f"episode_name: {episode_name}\n")
                f.write(f"success_status: {results['success_status']}\n")
                f.write(f"steps: {results['steps']}\n")
                f.write(f"distance_to_goal: {results['distance_to_goal']:.4f}\n")
                f.write(f"distance_to_final_goal: {results['distance_to_final_goal']:.4f}\n")
            
            logger.info(f"Episode {episode_name}: {results['success_status']} "
                       f"(steps={results['steps']}, dist={results['distance_to_goal']:.2f}m)")
            
        except Exception as e:
            logger.error(f"Error processing episode {episode_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary['failed_episodes'] += 1
            results_summary['episode_results'].append({
                'episode_name': episode_name,
                'success_status': f'error: {str(e)}',
                'steps': 0,
                'distance_to_goal': float('nan'),
                'distance_to_final_goal': float('nan')
            })
    
    # Calculate success rate
    results_summary['success_rate'] = (
        results_summary['successful_episodes'] / results_summary['total_episodes'] * 100
        if results_summary['total_episodes'] > 0 else 0
    )
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("Final Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total Episodes: {results_summary['total_episodes']}")
    logger.info(f"Successful: {results_summary['successful_episodes']}")
    logger.info(f"Failed: {results_summary['failed_episodes']}")
    logger.info(f"  - Exceeded Steps: {results_summary['exceeded_steps']}")
    logger.info(f"  - Stuck: {results_summary['stuck']}")
    logger.info(f"Success Rate: {results_summary['success_rate']:.2f}%")
    logger.info("=" * 60)
    
    # Save overall results summary
    summary_file = results_path / "results_summary.csv"
    import csv
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_name", "success_status", "steps", "distance_to_goal", "distance_to_final_goal"])
        for ep_result in results_summary['episode_results']:
            writer.writerow([
                ep_result['episode_name'],
                ep_result['success_status'],
                ep_result['steps'],
                f"{ep_result['distance_to_goal']:.4f}",
                f"{ep_result['distance_to_final_goal']:.4f}"
            ])
    
    logger.info(f"Results summary saved to: {summary_file}")
    
    # Save aggregated metrics
    metrics_file = results_path / "metrics_summary.txt"
    with open(metrics_file, "w") as f:
        f.write(f"total_episodes: {results_summary['total_episodes']}\n")
        f.write(f"successful_episodes: {results_summary['successful_episodes']}\n")
        f.write(f"failed_episodes: {results_summary['failed_episodes']}\n")
        f.write(f"exceeded_steps: {results_summary['exceeded_steps']}\n")
        f.write(f"stuck: {results_summary['stuck']}\n")
        f.write(f"success_rate: {results_summary['success_rate']:.2f}\n")
    
    logger.info(f"Metrics summary saved to: {metrics_file}")


if __name__ == "__main__":
    main()
