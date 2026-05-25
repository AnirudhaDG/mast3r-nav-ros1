"""
Mapping Data Transfer Script

Transfers mapping-related files from source directory to destination directory
for each episode. Files include:
- agent_states.npy
- action_labels.npy
- goalImg.png
- episode.npy
- obs_g.npy
- images/ directory (optional)

Can be configured to copy or symlink files.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

# =============================================================================
# CONFIGURABLE VARIABLES
# =============================================================================

# Episode selection mode
SINGLE_EPISODE = False
EPISODE_NAME = "svBbv1Pavdk_0000000_plant_7_"

# For multi-episode mode
EPISODE_LIST_FILE = "/home/onyx/work_dirs/vanshg/navigation/mast3r-nav/episodes_removing_blacklist.txt"
EPISODE_START_IDX = 0
EPISODE_END_IDX = -1
EPISODE_STEP = 1

# Directories
BASE_SOURCE_DIR = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d_navigation/hm3d_iin_val_320x240"
BASE_DEST_DIR = "/scratch2/public_scratch/vanshg/sg_habitat/hm3d_val_mapping_04ed325_commit"

# Files to transfer (relative to episode directory)
FILES_TO_TRANSFER = [
    "agent_states.npy",
    "action_labels.npy",
    "goalImg.png",
    "episode.npy",
    "obs_g.npy",
]

# Directories to transfer
DIRS_TO_TRANSFER = [
    "images",
]

# Transfer mode: "copy" or "symlink"
TRANSFER_MODE = "symlink"

# Overwrite existing files
OVERWRITE = False

# Verbose output
VERBOSE = True

# =============================================================================
# TRANSFER FUNCTIONS
# =============================================================================

def transfer_file(src_path: Path, dst_path: Path, mode: str, overwrite: bool) -> bool:
    """
    Transfer a single file using copy or symlink.
    
    Returns:
        True if transfer was successful or file already exists, False on error
    """
    if not src_path.exists():
        return False
    
    # Check if destination already exists
    if dst_path.exists() or dst_path.is_symlink():
        if not overwrite:
            return True  # Already exists, skip
        else:
            # Remove existing file/symlink
            if dst_path.is_symlink() or dst_path.is_file():
                dst_path.unlink()
            elif dst_path.is_dir():
                shutil.rmtree(dst_path)
    
    # Create parent directory if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if mode == "symlink":
            # Create relative symlink if possible, otherwise absolute
            try:
                rel_path = os.path.relpath(src_path, dst_path.parent)
                dst_path.symlink_to(rel_path)
            except ValueError:
                # On Windows or cross-device, use absolute path
                dst_path.symlink_to(src_path.resolve())
        else:  # copy
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"  Error transferring {src_path}: {e}")
        return False


def transfer_directory(src_path: Path, dst_path: Path, mode: str, overwrite: bool) -> bool:
    """
    Transfer a directory using copy or symlink.
    
    Returns:
        True if transfer was successful, False on error
    """
    if not src_path.exists() or not src_path.is_dir():
        return False
    
    # Check if destination already exists
    if dst_path.exists() or dst_path.is_symlink():
        if not overwrite:
            return True  # Already exists, skip
        else:
            # Remove existing
            if dst_path.is_symlink():
                dst_path.unlink()
            elif dst_path.is_dir():
                shutil.rmtree(dst_path)
    
    # Create parent directory if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if mode == "symlink":
            # Create symlink to directory
            try:
                rel_path = os.path.relpath(src_path, dst_path.parent)
                dst_path.symlink_to(rel_path)
            except ValueError:
                dst_path.symlink_to(src_path.resolve())
        else:  # copy
            shutil.copytree(src_path, dst_path)
        return True
    except Exception as e:
        print(f"  Error transferring directory {src_path}: {e}")
        return False


def transfer_episode(
    episode_name: str,
    base_source_dir: str,
    base_dest_dir: str,
    files_to_transfer: list,
    dirs_to_transfer: list,
    mode: str,
    overwrite: bool,
    verbose: bool
) -> dict:
    """
    Transfer all specified files and directories for a single episode.
    
    Returns:
        dict with transfer results
    """
    src_episode_dir = Path(base_source_dir) / episode_name
    dst_episode_dir = Path(base_dest_dir) / episode_name
    
    results = {
        'files_transferred': 0,
        'files_skipped': 0,
        'files_missing': 0,
        'dirs_transferred': 0,
        'dirs_skipped': 0,
        'dirs_missing': 0,
        'errors': []
    }
    
    # Check source exists
    if not src_episode_dir.exists():
        if verbose:
            print(f"  ✗ Source directory not found: {src_episode_dir}")
        results['errors'].append(f"Source directory not found")
        return results
    
    # Create destination directory
    dst_episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Transfer files
    for filename in files_to_transfer:
        src_file = src_episode_dir / filename
        dst_file = dst_episode_dir / filename
        
        if not src_file.exists():
            results['files_missing'] += 1
            if verbose:
                print(f"    - {filename}: missing")
            continue
        
        if dst_file.exists() and not overwrite:
            results['files_skipped'] += 1
            if verbose:
                print(f"    - {filename}: skipped (exists)")
            continue
        
        success = transfer_file(src_file, dst_file, mode, overwrite)
        if success:
            results['files_transferred'] += 1
            if verbose:
                print(f"    ✓ {filename}")
        else:
            results['errors'].append(f"Failed to transfer {filename}")
    
    # Transfer directories
    for dirname in dirs_to_transfer:
        src_dir = src_episode_dir / dirname
        dst_dir = dst_episode_dir / dirname
        
        if not src_dir.exists():
            results['dirs_missing'] += 1
            if verbose:
                print(f"    - {dirname}/: missing")
            continue
        
        if (dst_dir.exists() or dst_dir.is_symlink()) and not overwrite:
            results['dirs_skipped'] += 1
            if verbose:
                print(f"    - {dirname}/: skipped (exists)")
            continue
        
        success = transfer_directory(src_dir, dst_dir, mode, overwrite)
        if success:
            results['dirs_transferred'] += 1
            if verbose:
                print(f"    ✓ {dirname}/")
        else:
            results['errors'].append(f"Failed to transfer {dirname}/")
    
    return results


# =============================================================================
# EPISODE LIST HANDLING
# =============================================================================

def get_episode_list(
    single_episode: bool,
    episode_name: str,
    episode_list_file: str,
    base_source_dir: str,
    start_idx: int,
    end_idx: int,
    step: int
) -> list:
    """
    Get list of episode names based on config.
    """
    if single_episode:
        return [episode_name]
    
    # Try list file first
    if episode_list_file and Path(episode_list_file).exists():
        with open(episode_list_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(names)} episodes from {episode_list_file}")
        
        # Apply range filtering
        if end_idx == -1:
            end_idx = len(names)
        names = names[start_idx:end_idx:step]
        return names
    
    # Fall back to listing directories
    base_path = Path(base_source_dir)
    if base_path.exists():
        all_dirs = natsorted([d.name for d in base_path.iterdir() if d.is_dir()])
        
        if end_idx == -1:
            end_idx = len(all_dirs)
        
        episodes = all_dirs[start_idx:end_idx:step]
        print(f"Found {len(episodes)} episodes in {base_source_dir}")
        return episodes
    
    return []


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transfer mapping data files between directories")
    parser.add_argument("--single", action="store_true", help="Single episode mode")
    parser.add_argument("--episode", type=str, default=None, help="Episode name for single mode")
    parser.add_argument("--list-file", type=str, default=None, help="Episode list file")
    parser.add_argument("--source", type=str, default=None, help="Base source directory")
    parser.add_argument("--dest", type=str, default=None, help="Base destination directory")
    parser.add_argument("--start", type=int, default=None, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index (-1 for all)")
    parser.add_argument("--step", type=int, default=None, help="Step size")
    parser.add_argument("--mode", type=str, choices=["copy", "symlink"], default=None,
                       help="Transfer mode: copy or symlink")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                       help="List of files to transfer (overrides default)")
    parser.add_argument("--dirs", type=str, nargs="+", default=None,
                       help="List of directories to transfer (overrides default)")
    parser.add_argument("--no-dirs", action="store_true", help="Don't transfer directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be transferred without doing it")
    
    args = parser.parse_args()
    
    # Apply overrides
    single_episode = args.single if args.single else SINGLE_EPISODE
    episode_name = args.episode if args.episode else EPISODE_NAME
    episode_list_file = args.list_file if args.list_file else EPISODE_LIST_FILE
    base_source_dir = args.source if args.source else BASE_SOURCE_DIR
    base_dest_dir = args.dest if args.dest else BASE_DEST_DIR
    start_idx = args.start if args.start is not None else EPISODE_START_IDX
    end_idx = args.end if args.end is not None else EPISODE_END_IDX
    step = args.step if args.step is not None else EPISODE_STEP
    mode = args.mode if args.mode else TRANSFER_MODE
    overwrite = args.overwrite if args.overwrite else OVERWRITE
    verbose = not args.quiet if args.quiet else VERBOSE
    files_to_transfer = args.files if args.files else FILES_TO_TRANSFER
    dirs_to_transfer = [] if args.no_dirs else (args.dirs if args.dirs else DIRS_TO_TRANSFER)
    dry_run = args.dry_run
    
    print("=" * 60)
    print("MAPPING DATA TRANSFER")
    print("=" * 60)
    print(f"Source:      {base_source_dir}")
    print(f"Destination: {base_dest_dir}")
    print(f"Mode:        {mode}")
    print(f"Overwrite:   {overwrite}")
    print(f"Files:       {files_to_transfer}")
    print(f"Directories: {dirs_to_transfer}")
    if dry_run:
        print("*** DRY RUN - No files will be transferred ***")
    print("=" * 60)
    
    # Get episode list
    episodes = get_episode_list(
        single_episode, episode_name, episode_list_file,
        base_source_dir, start_idx, end_idx, step
    )
    
    if not episodes:
        print("No episodes found to process!")
        return
    
    print(f"\nProcessing {len(episodes)} episode(s)...")
    
    # Summary stats
    total_stats = {
        'files_transferred': 0,
        'files_skipped': 0,
        'files_missing': 0,
        'dirs_transferred': 0,
        'dirs_skipped': 0,
        'dirs_missing': 0,
        'episodes_success': 0,
        'episodes_failed': 0,
        'failed_episodes': []
    }
    
    # Process episodes
    for episode in tqdm(episodes, desc="Transferring"):
        if verbose:
            print(f"\n{episode}:")
        
        if dry_run:
            # Just check what would be transferred
            src_dir = Path(base_source_dir) / episode
            if src_dir.exists():
                for f in files_to_transfer:
                    exists = (src_dir / f).exists()
                    status = "exists" if exists else "missing"
                    if verbose:
                        print(f"    {f}: {status}")
                for d in dirs_to_transfer:
                    exists = (src_dir / d).exists()
                    status = "exists" if exists else "missing"
                    if verbose:
                        print(f"    {d}/: {status}")
                total_stats['episodes_success'] += 1
            else:
                if verbose:
                    print(f"    Source not found")
                total_stats['episodes_failed'] += 1
            continue
        
        results = transfer_episode(
            episode_name=episode,
            base_source_dir=base_source_dir,
            base_dest_dir=base_dest_dir,
            files_to_transfer=files_to_transfer,
            dirs_to_transfer=dirs_to_transfer,
            mode=mode,
            overwrite=overwrite,
            verbose=verbose
        )
        
        # Accumulate stats
        total_stats['files_transferred'] += results['files_transferred']
        total_stats['files_skipped'] += results['files_skipped']
        total_stats['files_missing'] += results['files_missing']
        total_stats['dirs_transferred'] += results['dirs_transferred']
        total_stats['dirs_skipped'] += results['dirs_skipped']
        total_stats['dirs_missing'] += results['dirs_missing']
        
        if results['errors']:
            total_stats['episodes_failed'] += 1
            total_stats['failed_episodes'].append(episode)
        else:
            total_stats['episodes_success'] += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER COMPLETE")
    print("=" * 60)
    print(f"Episodes: {total_stats['episodes_success']} success, {total_stats['episodes_failed']} failed")
    print(f"Files:    {total_stats['files_transferred']} transferred, "
          f"{total_stats['files_skipped']} skipped, {total_stats['files_missing']} missing")
    print(f"Dirs:     {total_stats['dirs_transferred']} transferred, "
          f"{total_stats['dirs_skipped']} skipped, {total_stats['dirs_missing']} missing")
    
    if total_stats['failed_episodes']:
        print(f"\nFailed episodes ({len(total_stats['failed_episodes'])}):")
        for ep in total_stats['failed_episodes'][:10]:
            print(f"  - {ep}")
        if len(total_stats['failed_episodes']) > 10:
            print(f"  ... and {len(total_stats['failed_episodes']) - 10} more")


if __name__ == "__main__":
    main()