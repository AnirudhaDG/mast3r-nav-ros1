#!/usr/bin/env python3
"""
Script to analyze episode folders and generate a report of failed/blank/error cases.

Usage: python analyze_episodes.py <base_folder_path>
Example: python analyze_episodes.py /path/to/base/folder/containing/episodes
"""

import os
import sys
import argparse
from pathlib import Path


def parse_metadata(metadata_path):
    """Parse metadata.txt file and extract relevant information."""
    metadata = {}
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key] = value
        return metadata
    except Exception as e:
        print(f"Error reading {metadata_path}: {e}")
        return None


def is_failed_case(metadata):
    """Determine if this is a failed/error/blank case."""
    if metadata is None:
        return True, "missing_metadata"
    
    success_status = metadata.get('success_status', '').strip()
    
    # Consider these as failed cases
    if success_status == '':
        return True, "blank_status"
    elif success_status == 'success':
        return False, "success"
    elif success_status in ['fail', 'failure', 'error', 'exceeded_steps']:
        return True, success_status
    else:
        # Any other status we consider as potentially failed
        return True, success_status


def analyze_episodes(base_folder):
    """Analyze all episode folders in the base folder."""
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"Error: Base folder '{base_folder}' does not exist.")
        return
    
    if not base_path.is_dir():
        print(f"Error: '{base_folder}' is not a directory.")
        return
    
    failed_episodes = []
    total_episodes = 0
    
    # Get all subdirectories (episode folders)
    episode_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for episode_dir in episode_dirs:
        total_episodes += 1
        metadata_path = episode_dir / 'metadata.txt'
        
        metadata = parse_metadata(metadata_path)
        is_failed, status = is_failed_case(metadata)
        
        if is_failed:
            episode_info = {
                'path': str(episode_dir),
                'episode_name': episode_dir.name,
                'status': status,
                'final_distance': metadata.get('final_distance', 'N/A') if metadata else 'N/A',
                'distance_to_final_goal_from_start': metadata.get('distance_to_final_goal_from_start', 'N/A') if metadata else 'N/A'
            }
            failed_episodes.append(episode_info)
    
    # Generate report
    report_path = base_path / 'failed_episodes_report.txt'
    
    with open(report_path, 'w') as f:
        # Write header with base folder path
        f.write(f"FAILED EPISODES ANALYSIS REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Base folder: {base_folder}\n")
        f.write(f"Analysis date: {os.popen('date').read().strip()}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Failed episodes: {len(failed_episodes)}\n")
        f.write(f"Success rate: {((total_episodes - len(failed_episodes)) / total_episodes * 100):.1f}%\n")
        f.write(f"{'='*60}\n\n")
        
        if failed_episodes:
            f.write("FAILED EPISODES DETAILS:\n")
            f.write("-" * 60 + "\n\n")
            
            for i, episode in enumerate(failed_episodes, 1):
                f.write(f"{i}. Episode: {episode['episode_name']}\n")
                f.write(f"   Path: {episode['path']}\n")
                f.write(f"   Status: {episode['status']}\n")
                f.write(f"   Final Distance: {episode['final_distance']}\n")
                f.write(f"   Distance to Final Goal from Start: {episode['distance_to_final_goal_from_start']}\n")
                f.write("-" * 60 + "\n")
        else:
            f.write("No failed episodes found!\n")
    
    print(f"Analysis complete!")
    print(f"Total episodes analyzed: {total_episodes}")
    print(f"Failed episodes found: {len(failed_episodes)}")
    print(f"Report saved to: {report_path}")
    
    # Print summary of failure types
    if failed_episodes:
        status_counts = {}
        for episode in failed_episodes:
            status = episode['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nFailure breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze episode folders and generate a report of failed/blank/error cases."
    )
    parser.add_argument(
        'base_folder',
        help='Path to the base folder containing episode directories'
    )
    
    args = parser.parse_args()
    analyze_episodes(args.base_folder)


if __name__ == "__main__":
    main()