"""
Graph Conversion Script: Old Format to New Sparse Format

Converts old dense topological graphs from sg_habitat to the new sparse format
used in mast3r-nav, enabling reuse of existing processed graphs without recomputation.

Key conversions:
- Dense graph (all H×W pixels) → Sparse graph (DA + goal nodes only)
- Edge attrs: edgeType → edge_type, weight_mast3r → weight
- Graph attrs: goalImgIdx → goal_img_idx, allPathLengths → all_path_lengths, etc.
- Separate costmaps file with (N_images, H, W) distances
"""

import os
import sys
import argparse
import time
import json
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
import blosc2

# Add sg_habitat to path for loading old graphs
BASE_DIR = Path(__file__).parent.parent
SG_HABITAT_DIR = BASE_DIR / "sg_habitat"
sys.path.insert(0, str(SG_HABITAT_DIR))

# =============================================================================
# CONFIGURABLE VARIABLES (set these directly or override via argparse)
# =============================================================================

# Episode selection mode
SINGLE_EPISODE = False           # True for single, False for multi-episode

# For single episode mode
EPISODE_NAME = "svBbv1Pavdk_0000000_plant_7_"     # Episode name (not full path)

# For multi-episode mode  
EPISODE_LIST_FILE = "/home/onyx/work_dirs/vanshg/navigation/mast3r-nav/episodes_removing_blacklist.txt"
EPISODE_START_IDX = 0               # Alternative: range-based selection
EPISODE_END_IDX = -1                # -1 = all
EPISODE_STEP = 1

# Directories
# BASE_IN_DIR = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d_navigation/hm3d_iin_val_320x240"
BASE_IN_DIR = "/scratch2/public_scratch/vanshg/sg_habitat/hm3d_val_mapping_04ed325_commit"
# BASE_IN_DIR = "/scratch2/public_scratch/vanshg/mast3r-nav/hm3d_sg_habitat_mapping_replicate"
BASE_OUT_DIR = "/scratch2/public_scratch/vanshg/mast3r-nav/hm3d_val_mapping_04ed325_commit_sg_habitat"

# Image directory (separate from graph directory)
# Set to None to use BASE_IN_DIR/episode_name/images
BASE_IMAGE_DIR = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d_navigation/hm3d_iin_val_320x240"

# Old graph filename pattern
OLD_GRAPH_FILENAME = "graph_with_distances_to_goalnode_fixed.pkl.b2s"

# Config values for new filename generation (match your processing params)
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
EDGE_CULLING_MODE = "NONE"
NODE_CULLING_MODE = "NONE"
NODE_CULLING_FACTOR = 10

# =============================================================================
# GRAPH LOADING (from sg_habitat format)
# =============================================================================

def load_old_graph(graph_path: str) -> nx.Graph:
    """
    Load compressed graph from old sg_habitat format.
    Handles both single and chunked blosc2 formats.
    """
    print(f"Loading old graph from: {graph_path}")
    t_start = time.time()
    
    with open(graph_path, 'rb') as f:
        # Read format marker
        format_marker = f.read(6)
        
        if format_marker == b"SINGLE":
            # Single file format
            compressed_size = int.from_bytes(f.read(8), 'little')
            compressed_data = f.read(compressed_size)
            decompressed = blosc2.decompress(compressed_data)
            graph_data = pickle.loads(decompressed)
            
        elif format_marker == b"CHUNKS":
            # Chunked format
            num_chunks = int.from_bytes(f.read(4), 'little')
            original_size = int.from_bytes(f.read(8), 'little')
            
            # Read chunk sizes
            chunk_sizes = []
            for _ in range(num_chunks):
                chunk_size = int.from_bytes(f.read(4), 'little')
                chunk_sizes.append(chunk_size)
            
            # Read and decompress chunks
            decompressed_chunks = []
            for i, chunk_size in enumerate(chunk_sizes):
                compressed_chunk = f.read(chunk_size)
                decompressed_chunk = blosc2.decompress(compressed_chunk)
                decompressed_chunks.append(decompressed_chunk)
            
            # Reconstruct original data
            decompressed = b''.join(decompressed_chunks)
            graph_data = pickle.loads(decompressed)
            
        else:
            raise ValueError(f"Unknown format marker: {format_marker}")
    
    # Reconstruct NetworkX graph
    G = nx.Graph()
    
    # Reconstruct nodes
    node_ids = graph_data['node_ids']
    node_maps = graph_data['node_maps']
    node_coords = graph_data['node_coords']
    node_pixels = graph_data['node_pixels']
    node_types = graph_data['node_types']
    
    for i, node_id in enumerate(node_ids):
        attrs = {
            'map': node_maps[i].tolist(),
            'coord_mast3r': node_coords[i],
            'pixel': node_pixels[i].tolist(),
            'type': str(node_types[i])
        }
        G.add_node(int(node_id), **attrs)
    
    # Reconstruct edges
    edge_sources = graph_data['edge_sources']
    edge_targets = graph_data['edge_targets']
    edge_margins = graph_data.get('edge_margins', np.ones(len(edge_sources)))
    edge_types = graph_data['edge_types']
    edge_weights = graph_data['edge_weights']
    
    for i in range(len(edge_sources)):
        attrs = {
            'margin': float(edge_margins[i]),
            'edgeType': str(edge_types[i]),
            'weight_mast3r': float(edge_weights[i])
        }
        G.add_edge(int(edge_sources[i]), int(edge_targets[i]), **attrs)
    
    # Reconstruct graph attributes
    G.graph.update(graph_data['graph_attrs'])
    
    # Reconstruct allPathLengths
    if 'path_nodes' in graph_data and 'path_distances' in graph_data:
        path_nodes = graph_data['path_nodes']
        path_distances = graph_data['path_distances']
        path_dict = {int(node): float(dist) for node, dist in zip(path_nodes, path_distances)}
        G.graph['allPathLengths'] = {'weight_mast3r': path_dict}
    
    print(f"Loaded graph in {time.time() - t_start:.2f}s: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# =============================================================================
# NODE AND EDGE EXTRACTION
# =============================================================================

def extract_da_and_goal_nodes(old_graph: nx.Graph) -> tuple:
    """
    Extract DA and goal nodes from old graph, preserving original node IDs.
    
    IMPORTANT: Old format stores pixel as [py, px] (row, col)
               New format stores pixel as [px, py] (col, row)
               This function converts to new format.
    
    Returns:
        da_node_ids: set of node IDs (DA + goal)
        da_nodes_dict: {node_id: converted_attrs}
    """
    da_node_ids = set()
    da_nodes_dict = {}
    
    # Get goal node ID from graph attributes
    goal_node_id = old_graph.graph.get('goalNodeId')
    
    for node_id, attrs in old_graph.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        
        # Keep DA nodes and goal node
        if node_type in ['da', 'goal']:
            da_node_ids.add(node_id)
            
            # Convert pixel format: old [py, px] → new [px, py]
            old_pixel = attrs['pixel']
            if hasattr(old_pixel, 'tolist'):
                old_pixel = old_pixel.tolist()
            new_pixel = [old_pixel[1], old_pixel[0]]  # Swap: [py, px] → [px, py]
            
            # Convert attributes
            converted_attrs = {
                'map': attrs['map'],
                'coord_mast3r': attrs['coord_mast3r'],
                'pixel': new_pixel,  # New format: [px, py]
                'type': node_type
            }
            da_nodes_dict[node_id] = converted_attrs
    
    # Ensure goal node is included even if type wasn't set correctly
    if goal_node_id is not None and goal_node_id not in da_node_ids:
        print(f"Warning: Goal node {goal_node_id} was not marked as 'goal' type, adding it")
        attrs = old_graph.nodes[goal_node_id]
        da_node_ids.add(goal_node_id)
        
        # Convert pixel format for goal node too
        old_pixel = attrs['pixel']
        if hasattr(old_pixel, 'tolist'):
            old_pixel = old_pixel.tolist()
        new_pixel = [old_pixel[1], old_pixel[0]]  # Swap: [py, px] → [px, py]
        
        da_nodes_dict[goal_node_id] = {
            'map': attrs['map'],
            'coord_mast3r': attrs['coord_mast3r'],
            'pixel': new_pixel,  # New format: [px, py]
            'type': 'goal'
        }
    
    print(f"Extracted {len(da_node_ids)} DA/goal nodes (goal_node_id={goal_node_id})")
    return da_node_ids, da_nodes_dict


def extract_da_edges(old_graph: nx.Graph, da_node_ids: set) -> list:
    """
    Extract edges where both endpoints are DA/goal nodes.
    Convert attribute names to new format.
    
    Returns:
        List of (src, dst, new_attrs) tuples
    """
    da_edges = []
    
    for src, dst, attrs in old_graph.edges(data=True):
        # Only keep edges between DA/goal nodes
        if src in da_node_ids and dst in da_node_ids:
            old_edge_type = attrs.get('edgeType', 'unknown')
            old_weight = attrs.get('weight_mast3r', 0.0)
            
            # Convert attribute names: edgeType → edge_type, weight_mast3r → weight
            new_attrs = {
                'edge_type': old_edge_type,
                'weight': old_weight
            }
            da_edges.append((src, dst, new_attrs))
    
    print(f"Extracted {len(da_edges)} edges between DA/goal nodes")
    return da_edges

# =============================================================================
# SPARSE GRAPH BUILDING
# =============================================================================

def build_sparse_graph(da_nodes_dict: dict, da_edges: list, old_graph: nx.Graph) -> nx.Graph:
    """
    Build new sparse graph with only DA/goal nodes and converted attributes.
    """
    G = nx.Graph()
    
    # Add nodes with original IDs
    for node_id, attrs in da_nodes_dict.items():
        G.add_node(node_id, **attrs)
    
    # Add edges with converted attributes
    G.add_edges_from(da_edges)
    
    # Convert graph-level attributes
    old_cfg = old_graph.graph.get('cfg', {})
    G.graph['cfg'] = old_cfg
    
    # Goal info (convert camelCase to snake_case)
    if 'goalImgIdx' in old_graph.graph:
        G.graph['goal_img_idx'] = old_graph.graph['goalImgIdx']
    if 'goalNodeCoords' in old_graph.graph:
        G.graph['goal_node_coords'] = old_graph.graph['goalNodeCoords']
    if 'goalNodeId' in old_graph.graph:
        G.graph['goal_node_id'] = old_graph.graph['goalNodeId']
    
    # Path lengths (filter to DA/goal nodes only, rename keys)
    if 'allPathLengths' in old_graph.graph:
        old_path_lengths = old_graph.graph['allPathLengths'].get('weight_mast3r', {})
        
        # Filter to only DA/goal nodes
        da_path_lengths = {}
        for node_id in da_nodes_dict.keys():
            if node_id in old_path_lengths:
                da_path_lengths[node_id] = old_path_lengths[node_id]
            else:
                da_path_lengths[node_id] = 1e6  # Unreachable
        
        G.graph['all_path_lengths'] = {'weight': da_path_lengths}
        print(f"Added path lengths for {len(da_path_lengths)} DA/goal nodes")
    
    print(f"Built sparse graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# =============================================================================
# COSTMAP BUILDING
# =============================================================================

def build_costmaps(old_graph: nx.Graph, num_images: int, H: int, W: int, 
                   episode_name: str, base_image_dir: str = None) -> tuple:
    """
    Build costmaps array from old graph's path lengths.
    
    Args:
        old_graph: The loaded old graph
        num_images: Number of images
        H: Image height
        W: Image width
        episode_name: Episode name
        base_image_dir: Base directory for images (if None, uses graph directory)
    
    Returns:
        costmaps: np.ndarray of shape (num_images, H, W)
        metadata: dict with goal info and config
    """
    max_dist = 1e6
    costmaps = np.full((num_images, H, W), max_dist, dtype=np.float32)
    
    # Get path lengths from old graph
    if 'allPathLengths' not in old_graph.graph:
        raise ValueError("Old graph missing 'allPathLengths' attribute")
    
    path_lengths = old_graph.graph['allPathLengths'].get('weight_mast3r', {})
    
    print(f"Building costmaps for {num_images} images ({H}x{W})...")
    
    # Fill costmaps from path lengths
    # Node ID formula: img_idx * (H * W) + (py * W + px)
    for node_id, dist in tqdm(path_lengths.items(), desc="Building costmaps"):
        node_id = int(node_id)
        
        # Reverse the node ID calculation
        img_idx = node_id // (H * W)
        pixel_idx = node_id % (H * W)
        py = pixel_idx // W
        px = pixel_idx % W
        
        if img_idx < num_images and 0 <= py < H and 0 <= px < W:
            costmaps[img_idx, py, px] = dist
    
    # Build metadata
    metadata = {
        'goal_img_idx': old_graph.graph.get('goalImgIdx'),
        'goal_pixel': old_graph.graph.get('goalNodeCoords'),
        'goal_node_id': old_graph.graph.get('goalNodeId'),
        'shape': list(costmaps.shape),
        'num_images': num_images,
        'height': H,
        'width': W
    }
    
    # Get image paths from the specified image directory
    image_paths = None
    if base_image_dir is not None:
        images_dir = Path(base_image_dir) / episode_name / "images"
    else:
        # Fall back to looking in graph directory
        images_dir = Path(old_graph.graph.get('cfg', {}).get('imgDir', ''))
        if not images_dir.exists():
            print(f"Warning: Could not find image directory")
    
    if images_dir and images_dir.exists():
        # Find all image files
        img_extensions = ['.jpg', '.jpeg', '.png']
        all_images = []
        for ext in img_extensions:
            all_images.extend(images_dir.glob(f"*{ext}"))
        if all_images:
            image_paths = [str(p) for p in natsorted(all_images)][:num_images]
            print(f"Found {len(image_paths)} image paths in {images_dir}")
    
    # Add image_paths to metadata
    if image_paths is not None:
        metadata['image_paths'] = image_paths
    
    # Add goal 3D coordinate if available
    goal_node_id = old_graph.graph.get('goalNodeId')
    if goal_node_id is not None and goal_node_id in old_graph.nodes:
        goal_coord = old_graph.nodes[goal_node_id].get('coord_mast3r')
        if goal_coord is not None:
            metadata['goal_coord_3d'] = goal_coord.tolist() if hasattr(goal_coord, 'tolist') else list(goal_coord)
    
    print(f"Built costmaps with shape {costmaps.shape}")
    return costmaps, metadata

# =============================================================================
# FILENAME GENERATION
# =============================================================================

def get_output_filenames(W: int, H: int, ec_mode: str, nc_mode: str, nc_factor: int) -> tuple:
    """
    Generate output filenames following new naming convention.
    
    Returns:
        (graph_filename, costmap_filename)
    """
    graph_filename = (
        f"graph_with_distances_to_goal_"
        f"{W}x{H}_"
        f"EC_{ec_mode}_"
        f"NC_{nc_mode}_"
        f"NCF_{nc_factor}.pkl"
    )
    
    costmap_filename = (
        f"costmaps_"
        f"{W}x{H}_"
        f"EC_{ec_mode}_"
        f"NC_{nc_mode}_"
        f"NCF_{nc_factor}"
    )
    
    return graph_filename, costmap_filename

# =============================================================================
# SAVING FUNCTIONS
# =============================================================================

def save_sparse_graph(graph: nx.Graph, output_path: str):
    """
    Save sparse graph using blosc2 chunked compression (same as new format).
    """
    t_start = time.time()
    
    # Decompose graph data for efficient compression
    graph_data = {
        'directed': graph.is_directed(),
        'graph_attrs': dict(graph.graph)
    }
    
    # Nodes
    nodes_list = list(graph.nodes(data=True))
    node_ids = []
    node_maps = []
    node_coords = []
    node_pixels = []
    node_types = []
    
    for node_id, attrs in nodes_list:
        node_ids.append(node_id)
        node_maps.append(attrs.get('map', [0, 0]))
        node_coords.append(attrs.get('coord_mast3r', np.array([0, 0, 0], dtype=np.float64)))
        node_pixels.append(attrs.get('pixel', [0, 0]))
        node_types.append(attrs.get('type', 'unknown'))
    
    graph_data['node_ids'] = np.array(node_ids)
    graph_data['node_maps'] = np.array(node_maps, dtype=np.int32)
    graph_data['node_coords'] = np.array(node_coords, dtype=np.float64)
    graph_data['node_pixels'] = np.array(node_pixels, dtype=np.int32)
    graph_data['node_types'] = np.array(node_types, dtype='U20')
    
    # Edges
    edges_list = list(graph.edges(data=True))
    edge_sources = []
    edge_targets = []
    edge_types = []
    edge_weights = []
    
    for source, target, attrs in edges_list:
        edge_sources.append(source)
        edge_targets.append(target)
        edge_types.append(attrs.get('edge_type', 'unknown'))
        edge_weights.append(attrs.get('weight', 0.0))
    
    graph_data['edge_sources'] = np.array(edge_sources, dtype=np.int32)
    graph_data['edge_targets'] = np.array(edge_targets, dtype=np.int32)
    graph_data['edge_types'] = np.array(edge_types, dtype='U20')
    graph_data['edge_weights'] = np.array(edge_weights, dtype=np.float64)
    
    # Handle path_lengths specially
    if 'all_path_lengths' in graph_data['graph_attrs']:
        path_lengths = graph_data['graph_attrs']['all_path_lengths']['weight']
        path_nodes = np.array(list(path_lengths.keys()))
        path_distances = np.array(list(path_lengths.values()), dtype=np.float64)
        
        graph_data['path_nodes'] = path_nodes
        graph_data['path_distances'] = path_distances
        del graph_data['graph_attrs']['all_path_lengths']
    
    # Serialize and compress
    serialized = pickle.dumps(graph_data, protocol=pickle.HIGHEST_PROTOCOL)
    original_size = len(serialized)
    
    compressed_path = f"{output_path}.b2s"
    
    # blosc2 size limit (~2GB)
    BLOSC2_MAX_SIZE = 2000000000
    
    if original_size <= BLOSC2_MAX_SIZE:
        compressed = blosc2.compress(serialized, codec=blosc2.Codec.ZSTD, clevel=9)
        
        with open(compressed_path, 'wb') as f:
            f.write(b"SINGLE")
            f.write(len(compressed).to_bytes(8, 'little'))
            f.write(compressed)
        
        compressed_size = len(compressed)
    else:
        # Chunked compression for large data
        chunk_size = 1024 * 1024 * 1024  # 1GB chunks
        num_chunks = (len(serialized) + chunk_size - 1) // chunk_size
        compressed_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(serialized))
            chunk = serialized[start_idx:end_idx]
            compressed_chunk = blosc2.compress(chunk, codec=blosc2.Codec.ZSTD, clevel=9)
            compressed_chunks.append(compressed_chunk)
        
        with open(compressed_path, 'wb') as f:
            f.write(b"CHUNKS")
            f.write(num_chunks.to_bytes(4, 'little'))
            f.write(original_size.to_bytes(8, 'little'))
            
            for chunk in compressed_chunks:
                f.write(len(chunk).to_bytes(4, 'little'))
            
            for chunk in compressed_chunks:
                f.write(chunk)
        
        compressed_size = sum(len(chunk) for chunk in compressed_chunks)
    
    compression_ratio = (1 - compressed_size / original_size) * 100
    print(f"✓ Saved sparse graph to {compressed_path} ({compressed_size / 1e6:.2f} MB, {compression_ratio:.1f}% compression) in {time.time() - t_start:.2f}s")


def save_costmaps(costmaps: np.ndarray, metadata: dict, output_path: str):
    """
    Save costmaps to compressed NPZ file.
    """
    np.savez_compressed(
        output_path,
        costmaps=costmaps,
        metadata=json.dumps(metadata)
    )
    
    file_size = os.path.getsize(f"{output_path}.npz") / 1e6
    print(f"✓ Saved costmaps to {output_path}.npz ({file_size:.2f} MB)")

# =============================================================================
# MAIN CONVERSION LOGIC
# =============================================================================

def get_num_images_from_graph(old_graph: nx.Graph) -> int:
    """
    Determine number of images from graph by finding max image index.
    """
    max_img_idx = 0
    for node_id, attrs in old_graph.nodes(data=True):
        img_idx = attrs.get('map', [0, 0])[0]
        max_img_idx = max(max_img_idx, img_idx)
    return max_img_idx + 1


def convert_single_episode(
    episode_name: str,
    base_in_dir: str,
    base_out_dir: str,
    old_graph_filename: str,
    W: int, H: int,
    ec_mode: str, nc_mode: str, nc_factor: int
) -> bool:
    """
    Convert a single episode from old format to new format.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Converting: {episode_name}")
    print(f"{'='*60}")
    
    # Paths
    episode_in_dir = Path(base_in_dir) / episode_name
    episode_out_dir = Path(base_out_dir) / episode_name
    old_graph_path = episode_in_dir / old_graph_filename
    
    # Check input exists
    if not old_graph_path.exists():
        print(f"✗ Old graph not found: {old_graph_path}")
        return False
    
    # Create output directory
    episode_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get output filenames
    graph_filename, costmap_filename = get_output_filenames(W, H, ec_mode, nc_mode, nc_factor)
    graph_out_path = episode_out_dir / graph_filename
    costmap_out_path = episode_out_dir / costmap_filename
    
    try:
        # 1. Load old graph
        old_graph = load_old_graph(str(old_graph_path))
        
        # 2. Extract DA nodes + goal node
        da_node_ids, da_nodes_dict = extract_da_and_goal_nodes(old_graph)
        
        # 3. Extract edges between DA/goal nodes
        da_edges = extract_da_edges(old_graph, da_node_ids)
        
        # 4. Build sparse graph
        sparse_graph = build_sparse_graph(da_nodes_dict, da_edges, old_graph)
        
        # 5. Build costmaps
        num_images = get_num_images_from_graph(old_graph)
        
        # Get image paths from episode directory
        images_dir = episode_in_dir / "images"
        image_paths = None
        if images_dir.exists():
            # Find all image files
            img_extensions = ['.jpg', '.jpeg', '.png']
            all_images = []
            for ext in img_extensions:
                all_images.extend(images_dir.glob(f"*{ext}"))
            if all_images:
                image_paths = [str(p) for p in natsorted(all_images)][:num_images]
                print(f"Found {len(image_paths)} image paths in {images_dir}")
        
        costmaps, metadata = build_costmaps(old_graph, num_images, H, W, episode_name, BASE_IMAGE_DIR)
        
        # 6. Save outputs
        save_sparse_graph(sparse_graph, str(graph_out_path))
        save_costmaps(costmaps, metadata, str(costmap_out_path))
        
        print(f"✓ Successfully converted {episode_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {episode_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_episode_list(
    single_episode: bool,
    episode_name: str,
    episode_list_file: str,
    base_in_dir: str,
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
    base_path = Path(base_in_dir)
    if base_path.exists():
        all_dirs = natsorted([d.name for d in base_path.iterdir() if d.is_dir()])
        
        if end_idx == -1:
            end_idx = len(all_dirs)
        
        episodes = all_dirs[start_idx:end_idx:step]
        print(f"Found {len(episodes)} episodes in {base_in_dir}")
        return episodes
    
    return []


def main():
    parser = argparse.ArgumentParser(description="Convert old dense graphs to new sparse format")
    parser.add_argument("--single", action="store_true", help="Single episode mode (overrides SINGLE_EPISODE)")
    parser.add_argument("--episode", type=str, default=None, help="Episode name for single mode")
    parser.add_argument("--list-file", type=str, default=None, help="Episode list file")
    parser.add_argument("--base-in", type=str, default=None, help="Base input directory")
    parser.add_argument("--base-out", type=str, default=None, help="Base output directory")
    parser.add_argument("--start", type=int, default=None, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index (-1 for all)")
    parser.add_argument("--step", type=int, default=None, help="Step size")
    parser.add_argument("--graph-file", type=str, default=None, help="Old graph filename")
    
    args = parser.parse_args()
    
    # Apply overrides
    single_episode = args.single if args.single else SINGLE_EPISODE
    episode_name = args.episode if args.episode else EPISODE_NAME
    episode_list_file = args.list_file if args.list_file else EPISODE_LIST_FILE
    base_in_dir = args.base_in if args.base_in else BASE_IN_DIR
    base_out_dir = args.base_out if args.base_out else BASE_OUT_DIR
    start_idx = args.start if args.start is not None else EPISODE_START_IDX
    end_idx = args.end if args.end is not None else EPISODE_END_IDX
    step = args.step if args.step is not None else EPISODE_STEP
    old_graph_filename = args.graph_file if args.graph_file else OLD_GRAPH_FILENAME
    
    print("="*60)
    print("GRAPH CONVERSION: Old Dense → New Sparse Format")
    print("="*60)
    print(f"Base input:  {base_in_dir}")
    print(f"Base output: {base_out_dir}")
    print(f"Old graph file: {old_graph_filename}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Edge culling: {EDGE_CULLING_MODE}")
    print(f"Node culling: {NODE_CULLING_MODE}, factor={NODE_CULLING_FACTOR}")
    print("="*60)
    
    # Get episode list
    episodes = get_episode_list(
        single_episode, episode_name, episode_list_file,
        base_in_dir, start_idx, end_idx, step
    )
    
    if not episodes:
        print("No episodes found to process!")
        return
    
    print(f"\nProcessing {len(episodes)} episode(s)...")
    
    # Process episodes
    results = {'success': 0, 'failed': 0, 'failed_episodes': []}
    
    for episode in tqdm(episodes, desc="Converting episodes"):
        success = convert_single_episode(
            episode_name=episode,
            base_in_dir=base_in_dir,
            base_out_dir=base_out_dir,
            old_graph_filename=old_graph_filename,
            W=IMAGE_WIDTH,
            H=IMAGE_HEIGHT,
            ec_mode=EDGE_CULLING_MODE,
            nc_mode=NODE_CULLING_MODE,
            nc_factor=NODE_CULLING_FACTOR
        )
        
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1
            results['failed_episodes'].append(episode)
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Successful: {results['success']}/{len(episodes)}")
    print(f"Failed: {results['failed']}/{len(episodes)}")
    
    if results['failed_episodes']:
        print(f"\nFailed episodes:")
        for ep in results['failed_episodes']:
            print(f"  - {ep}")


if __name__ == "__main__":
    main()
