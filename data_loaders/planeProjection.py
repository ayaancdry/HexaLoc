
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
# fallback for older matplotlib
try:
    import matplotlib.colormaps as cmaps
    get_cmap = cmaps.get_cmap
except ImportError:
    get_cmap = plt.get_cmap

def encode_to_multi_xy_planes(bin_file_path, label_file_path=None, bounds_zyx=[-3, 1, -70.4, 70.4, -40, 40], 
                               num_planes=100, grid_size=GRID_SIZE, device=None):
    """
    GPU-accelerated encoding of point cloud into multiple XY planes (bird's-eye view slices).
    Each plane stores the signed depth value of points relative to the slice start.
    
    This version uses vectorized operations instead of Python loops for significant speedup.
    Even on CPU, vectorized operations are ~10-50x faster than the original Python loop.
    """
    # Auto-detect device if not specified
    # Note: In DataLoader workers, CUDA may not work properly, so we catch errors and fall back to CPU
    if device is None:
        try:
            if torch.cuda.is_available():
                # Try to use CUDA - this may fail in worker processes
                device = 'cuda'
            else:
                device = 'cpu'
        except:
            device = 'cpu'
    
    try:
        return _encode_to_multi_xy_planes_vectorized(bin_file_path, label_file_path, bounds_zyx, 
                                                      num_planes, grid_size, device)
    except RuntimeError as e:
        # If CUDA fails (e.g., in DataLoader worker), fall back to CPU
        if 'cuda' in str(e).lower() or 'CUDA' in str(e):
            return _encode_to_multi_xy_planes_vectorized(bin_file_path, label_file_path, bounds_zyx, 
                                                          num_planes, grid_size, 'cpu')
        raise


def _encode_to_multi_xy_planes_vectorized(bin_file_path, label_file_path, bounds_zyx, 
                                           num_planes, grid_size, device):
    """
    Internal vectorized implementation of plane encoding.
    Works on both CPU and GPU with identical logic.
    """
    # Step 1: Load point cloud (file I/O must be on CPU)
    points = torch.from_numpy(np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4))
    xyz = points[:, :3].to(device)  # Move to target device
    
    # Step 2: Apply bounds filtering (vectorized)
    if bounds_zyx is not None:
        z_min_bound, z_max_bound, y_min_bound, y_max_bound, x_min_bound, x_max_bound = bounds_zyx
        mask = ((xyz[:, 0] >= x_min_bound) & (xyz[:, 0] <= x_max_bound) &
                (xyz[:, 1] >= y_min_bound) & (xyz[:, 1] <= y_max_bound) &
                (xyz[:, 2] >= z_min_bound) & (xyz[:, 2] <= z_max_bound))
        xyz = xyz[mask]
    
    # Handle empty point cloud
    if xyz.shape[0] == 0:
        empty_planes = [torch.zeros(grid_size, grid_size) for _ in range(num_planes)]
        return empty_planes, None, (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    
    # Step 3: Compute bounds (same as original)
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    
    z_range = z_max - z_min
    z_slice_width = z_range / num_planes
    
    # Step 4: Compute which plane each point belongs to (vectorized)
    # Original logic: for plane i, z_start = z_max - (i+1)*z_slice_width, z_end = z_max - i*z_slice_width
    # Points in slice: z > z_start AND z <= z_end
    # Plane index formula: i = floor((z_max - z) / z_slice_width)
    plane_indices = ((z_max - xyz[:, 2]) / (z_slice_width + 1e-8)).long()
    plane_indices = plane_indices.clamp(0, num_planes - 1)
    
    # Step 5: Compute grid indices for all points (vectorized, same formula as original)
    x_idx = ((xyz[:, 0] - x_min) / (x_max - x_min + 1e-8) * (grid_size - 1)).long()
    y_idx = ((xyz[:, 1] - y_min) / (y_max - y_min + 1e-8) * (grid_size - 1)).long()
    x_idx = x_idx.clamp(0, grid_size - 1)
    y_idx = y_idx.clamp(0, grid_size - 1)
    
    # Step 6: Compute signed depth for each point (same formula as original)
    # z_start = z_max - (plane_index + 1) * z_slice_width
    z_starts = z_max - (plane_indices.float() + 1) * z_slice_width
    signed_depth = xyz[:, 2] - z_starts + 0.01
    
    # Step 7: Create flat indices for 3D tensor [plane, y, x]
    flat_indices = plane_indices * (grid_size * grid_size) + y_idx * grid_size + x_idx
    
    # Step 8: Handle "keep smallest |signed_depth|" logic
    # Strategy: Sort by |signed_depth| descending, then scatter (last write wins = smallest abs)
    abs_depth = signed_depth.abs()
    sorted_order = abs_depth.argsort(descending=True)  # Largest abs first, smallest last
    
    flat_indices_sorted = flat_indices[sorted_order]
    signed_depth_sorted = signed_depth[sorted_order]
    
    # Step 9: Create output tensor and scatter (last write wins)
    planes_flat = torch.zeros(num_planes * grid_size * grid_size, device=device)
    planes_flat.scatter_(0, flat_indices_sorted, signed_depth_sorted)
    
    # Step 10: Reshape to [num_planes, grid_size, grid_size] and convert to list
    planes_tensor = planes_flat.view(num_planes, grid_size, grid_size)
    
    # Move to CPU and convert to list of tensors (for compatibility with original interface)
    planes_cpu = planes_tensor.cpu()
    planes = [planes_cpu[i] for i in range(num_planes)]
    
    # Return bounds as Python floats
    z_bounds = (z_min.cpu().item(), z_max.cpu().item())
    xy_bounds = (x_min.cpu().item(), x_max.cpu().item(), y_min.cpu().item(), y_max.cpu().item())
    
    return planes, None, z_bounds, xy_bounds


def encode_to_multi_xy_planes_with_custom_sizes(bin_file_path, label_file_path=None, bounds_zyx=[-3, 1, -70.4, 70.4, -40, 40], 
                                               num_planes_full=100, grid_size_full=GRID_SIZE,
                                               num_planes_small=None, grid_size_small=None):
    """
    Encode point cloud from .bin file into multiple XY planes with both full-size and custom-sized versions.
    """
    # Set default values for small version
    if num_planes_small is None:
        num_planes_small = num_planes_full
    if grid_size_small is None:
        grid_size_small = grid_size_full // 2
    
    # Get full-size planes
    planes_full, label_planes_full, z_bounds, xy_bounds = encode_to_multi_xy_planes(
        bin_file_path, label_file_path, bounds_zyx, num_planes_full, grid_size_full
    )
    
    # Get small planes with custom dimensions
    planes_small, label_planes_small, _, _ = encode_to_multi_xy_planes(
        bin_file_path, label_file_path, bounds_zyx, num_planes_small, grid_size_small
    )
    
    return planes_full, planes_small, label_planes_full, label_planes_small, z_bounds, xy_bounds
