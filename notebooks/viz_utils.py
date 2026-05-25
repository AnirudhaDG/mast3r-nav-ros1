import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_waypoints(waypoints, save_path, dpi=300, figsize=(6, 6)):
    """
    Plot and save waypoints trajectory using the exact plot_traj function.
    
    Args:
        waypoints: np.ndarray, (N, 2) or (N, 4) waypoint coordinates
        save_path: str, path to save the plot
        dpi: int, resolution for saved image (default: 300)
        figsize: tuple, figure size (default: (6, 6))
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Use your exact plot_traj function
    plot_traj(ax, waypoints)
    
    # Save the plot
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    plt.close(fig)

def plot_heatmap_with_colorbar_clean(heatmap ,step=0,
                                     cmap='turbo',
                                     cbar_tick_fontsize=10,
                                     cbar_label_fontsize=24,
                                     title=None, title_fontsize=14,
                                     save_path=None, save_path_colorbar=None, 
                                     dpi=150, pixel=False):

        heatmap_orig = heatmap.copy()

        if not pixel:
            heatmap = np.where(heatmap >= 99, np.nan, heatmap)
            valid_costs = heatmap[~np.isnan(heatmap)]
            if valid_costs.size > 0:
                vmin_rel, vmax_rel = np.percentile(valid_costs, [5, 95])
            else:
                vmin_rel, vmax_rel = 0, 1

            display_heatmap_rel = heatmap.copy()
        else:
            vmin_rel = np.min(heatmap_orig.flatten())
            vmax_rel = np.max(heatmap_orig.flatten())

            display_heatmap_rel = np.clip(heatmap, vmin_rel, vmax_rel)

        fig_rel, ax_rel = plt.subplots()
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='white')  # <-- NaNs show as white
        heat_rel = ax_rel.imshow(display_heatmap_rel, cmap=cmap_obj, vmin=vmin_rel, vmax=vmax_rel)
        ax_rel.axis('off')
        fig_rel.tight_layout()
        if save_path:
            fig_rel.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        # ax_rel.set_title(f"Step {step} | RELATIVE vmin={vmin_rel:.2f}, vmax={vmax_rel:.2f}", fontsize=12)
        cbar_rel = fig_rel.colorbar(heat_rel, ax=ax_rel, fraction=0.03, pad=0.01)
        ticks = np.linspace(vmin_rel, vmax_rel, num=5)
        cbar_rel.set_ticks(ticks)
        cbar_rel.set_ticklabels([f"{t:02.2f}" for t in ticks])
        # # cbar_rel.set_label('COST', fontsize=cbar_label_fontsize)
        cbar_rel.ax.tick_params(labelsize=cbar_tick_fontsize)
        ax_rel.axis('off')
        plt.tight_layout()
        if save_path:
            rel_path = str(save_path_colorbar).replace('.png', '_withbar.png')
            fig_rel.savefig(rel_path, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.0)
            # print(f"Saved RELATIVE heatmap to {rel_path}")
        plt.close(fig_rel)

def plot_costmap_with_info(costmap, step_id=None, save_path=None):
    """
    Plots the costmap with 'turbo' colormap, colorbar, and a title showing step id, min, and max cost.
    If save_path is provided, saves the plot to that path and does not display it.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(costmap, cmap='turbo')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    min_cost = np.nanmin(costmap)
    max_cost = np.nanmax(costmap)
    title = f"Step: {step_id if step_id is not None else '-'} | Min: {min_cost:.3f} | Max: {max_cost:.3f}"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_query_img_with_costmap(query_img, costmap, step_id=None, save_path=None, alpha=0.6):
    """
    Plots query image, overlayed costmap, and costmap side by side.
    Order: (RGB, Overlayed Costmap, Costmap) with shared colorbar.
    
    Args:
        query_img: np.ndarray, (H, W, 3) RGB image (uint8 or float32)
        costmap: np.ndarray, (H, W) cost values
        step_id: int, step identifier for title
        save_path: str, path to save the plot (if None, displays inline)
        alpha: float, transparency for overlay (0.0 = fully transparent, 1.0 = fully opaque)
    """
    # Handle different image formats (uint8 vs float32)
    if query_img.dtype == np.uint8:
        display_img = query_img / 255.0
    else:
        display_img = np.clip(query_img, 0, 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Query RGB image
    axes[0].imshow(display_img)
    axes[0].set_title(f'Query Image - Step {step_id}')
    axes[0].axis('off')
    
    # Plot 2: Overlayed costmap on RGB
    axes[1].imshow(display_img)
    im_overlay = axes[1].imshow(costmap, cmap='turbo', alpha=alpha)
    axes[1].set_title(f'RGB + Costmap Overlay - Step {step_id}')
    axes[1].axis('off')
    
    # Plot 3: Costmap only
    im_costmap = axes[2].imshow(costmap, cmap='turbo')
    min_cost = np.nanmin(costmap)
    max_cost = np.nanmax(costmap)
    axes[2].set_title(f'Costmap - Step {step_id}\nMin: {min_cost:.3f} | Max: {max_cost:.3f}')
    axes[2].axis('off')
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_costmap, cax=cbar_ax)
    cbar.set_label('Cost Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Make room for colorbar
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def plot_query_img_with_costmap_compact(query_img, costmap, step_id=None, save_path=None, alpha=0.6):
    """
    Compact version with smaller figure size and tighter layout.
    """
    # Handle different image formats
    if query_img.dtype == np.uint8:
        display_img = query_img / 255.0
    else:
        display_img = np.clip(query_img, 0, 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Query RGB image
    axes[0].imshow(display_img)
    axes[0].set_title(f'RGB\nStep {step_id}', fontsize=10)
    axes[0].axis('off')
    
    # Plot 2: Overlayed costmap on RGB
    axes[1].imshow(display_img)
    axes[1].imshow(costmap, cmap='turbo', alpha=alpha)
    axes[1].set_title(f'RGB + Overlay\nStep {step_id}', fontsize=10)
    axes[1].axis('off')
    
    # Plot 3: Costmap only
    im_costmap = axes[2].imshow(costmap, cmap='turbo')
    min_cost = np.nanmin(costmap)
    max_cost = np.nanmax(costmap)
    axes[2].set_title(f'Costmap\nMin: {min_cost:.2f} | Max: {max_cost:.2f}', fontsize=10)
    axes[2].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im_costmap, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Cost', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])

def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing

def plot_traj(ax, traj, quiver_freq=1):
    """
    Plot trajectory - exact copy from your reference
    """
    ax.plot(
        traj[:, 1],
        traj[:, 0],
        color='c',
        alpha=0.5,
        marker="o",
    )
    bearings = gen_bearings_from_waypoints(traj)
    ax.quiver(
        traj[::quiver_freq, 1],
        traj[::quiver_freq, 0],
        -bearings[::quiver_freq, 1],
        bearings[::quiver_freq, 0],
        color='y',
        scale=1.0,
    )
    # Turn off grid and axes for clean visualization
    ax.grid(False)
    ax.axis('off')
    ax.set_ylim(-1, 12)
    ax.set_xlim(-4, 4)
    ax.invert_xaxis()
    ax.set_aspect("equal", "box")

def plot_query_img_costmap_waypoints(query_img, costmap, waypoints, step_id=None, save_path=None, alpha=0.6, dpi=150, 
                                   nan_threshold=None, use_percentile=False):
    """
    Plots query image, costmap, and waypoints side by side.
    Order: (RGB, Costmap, Waypoints)
    
    Args:
        query_img: np.ndarray, (H, W, 3) RGB image (uint8 or float32)
        costmap: np.ndarray, (H, W) cost values
        waypoints: np.ndarray, (N, 2) waypoint coordinates
        step_id: int, step identifier for title
        save_path: str, path to save the plot (if None, displays inline)
        alpha: float, not used in this function but kept for consistency
        dpi: int, resolution for saved image
        nan_threshold: float or None, values >= this threshold will be set to NaN (default: None, no filtering)
        use_percentile: bool, if True use 5th/95th percentile for vmin/vmax, else use actual min/max (default: False)
    """
    # Handle different image formats (uint8 vs float32)
    if query_img.dtype == np.uint8:
        display_img = query_img / 255.0
    else:
        display_img = np.clip(query_img, 0, 1)
    
    # Process costmap for display
    costmap_display = costmap.copy()
    
    # Apply NaN threshold if specified
    if nan_threshold is not None:
        costmap_display = np.where(costmap_display >= nan_threshold, np.nan, costmap_display)
    
    # Calculate original min/max for title
    original_min = np.nanmin(costmap)
    original_max = np.nanmax(costmap)
    
    # Calculate vmin/vmax for display
    valid_costs = costmap_display[~np.isnan(costmap_display)]
    if valid_costs.size > 0:
        if use_percentile:
            vmin_display, vmax_display = np.percentile(valid_costs, [5, 95])
        else:
            vmin_display, vmax_display = np.nanmin(costmap_display), np.nanmax(costmap_display)
    else:
        vmin_display, vmax_display = 0, 1
    
    # Setup colormap to handle NaN values
    cmap_obj = plt.get_cmap('turbo').copy()
    cmap_obj.set_bad(color='white')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Query RGB image
    axes[0].imshow(display_img)
    axes[0].set_title(f'Query Image - Step {step_id}')
    axes[0].axis('off')
    
    # Plot 2: Costmap
    im_costmap = axes[1].imshow(costmap_display, cmap=cmap_obj, vmin=vmin_display, vmax=vmax_display)
    
    # Create detailed title with all relevant information
    title_parts = [f'Costmap - Step {step_id}']
    title_parts.append(f'Original: [{original_min:.3f}, {original_max:.3f}]')
    title_parts.append(f'Display: [{vmin_display:.3f}, {vmax_display:.3f}]')
    
    if nan_threshold is not None:
        title_parts.append(f'NaN≥{nan_threshold}')
    
    if use_percentile:
        title_parts.append('(5th-95th %tile)')
    
    axes[1].set_title('\n'.join(title_parts))
    axes[1].axis('off')
    
    # Add colorbar for costmap
    cbar = plt.colorbar(im_costmap, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Cost Value', rotation=270, labelpad=15)
    
    # Plot 3: Waypoints trajectory using your exact plot_traj function
    plot_traj(axes[2], waypoints)
    axes[2].set_title(f'Waypoints - Step {step_id}\nPoints: {len(waypoints)}')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()

def plot_query_img_costmap_waypoints_compact(query_img, costmap, waypoints, step_id=None, save_path=None, dpi=150):
    """
    Compact version with smaller figure size and tighter layout.
    """
    # Handle different image formats
    if query_img.dtype == np.uint8:
        display_img = query_img / 255.0
    else:
        display_img = np.clip(query_img, 0, 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Query RGB image
    axes[0].imshow(display_img)
    axes[0].set_title(f'RGB\nStep {step_id}', fontsize=10)
    axes[0].axis('off')
    
    # Plot 2: Costmap
    im_costmap = axes[1].imshow(costmap, cmap='turbo')
    min_cost = np.nanmin(costmap)
    max_cost = np.nanmax(costmap)
    axes[1].set_title(f'Costmap\nMin: {min_cost:.2f} | Max: {max_cost:.2f}', fontsize=10)
    axes[1].axis('off')
    
    # Add colorbar for costmap
    cbar = plt.colorbar(im_costmap, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Cost', rotation=270, labelpad=10)
    
    # Plot 3: Waypoints trajectory using your exact plot_traj function
    plot_traj(axes[2], waypoints)
    axes[2].set_title(f'Waypoints\nPts: {len(waypoints)}', fontsize=10)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()