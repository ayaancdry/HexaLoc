#!/usr/bin/env python3
"""
PyTorch dataloader for the NCLT dataset. 

It will:
- Discover sequences under the NCLT root. 
- For each sequence, generate or load interpolated poses and valid timestamps
  following the logic in SGLoc's NCLTVelodyne_datagenerator.py
- On-the-fly encode Velodyne scans to multiple XY planes via planeProjection.encode_to_multi_xy_planes
"""

import os
import os.path as osp
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader

from data_loaders.planeProjection import encode_to_multi_xy_planes


#
# Pose utilities import: prefer local plane_project_cbam3/pose_util.py; fallback to SGLoc's file by absolute path
try:
    # Try importing from parent directory (plane_project_cbam3 root)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import pose_util
except ImportError:
    # Fallback to absolute path
    try:
        sys.path.append("/csehome/pydah/anirudh/SGLoc/code/utils")
        import pose_util
    except ImportError:
        # Try another absolute path
        try:
            sys.path.append("/home/ayaan/Desktop/plane_project_cbam3")
            import pose_util
        except ImportError:
            raise ImportError(
            "Could not find pose_util.py in plane_project_cbam3 or SGLoc. Expected at /csehome/pydah/anirudh/SGLoc/code/utils/pose_util.py"
            )

# Explicitly import required functions from pose_util
from pose_util import (
    filter_overflow_nclt,
    interpolate_pose_nclt,
    so3_to_euler_nclt,
    process_poses,
    qexp,
)

class NCLTDataset(Dataset):
    """
    NCLT dataset that provides plane-encoded frames and corresponding 6DOF poses.

    - Sequences are directories named by date (e.g., 2012-01-08) under nclt_root
    - Each sequence must contain 'velodyne_left' with .bin scans and a groundtruth CSV
      named 'groundtruth_<seq>.csv'
    - We mirror SGLoc's flow: compute/load valid timestamps and interpolated poses,
      compute normalization stats (mean/std of translation), and return normalized 6DOF poses
      in log-quaternion format [tx, ty, tz, log_qx, log_qy, log_qz]
    """

    def __init__(
        self,
        nclt_root: str,
        sequence_ids: Optional[List[str]] = None,
        train: bool = True,
        valid: bool = False,
        augmentation: Optional["PlaneTransform"] = None,
        bounds_zyx: List[float] = [-7, 1, -55, 60, -40, 40],
        num_planes: int = 30,
        grid_size: int = 512,
        real: bool = False,
    ) -> None:
        self.nclt_root = Path(nclt_root).expanduser().resolve()
        self.train = train
        self.valid = valid
        self.augmentation = augmentation
        self.bounds_zyx = bounds_zyx
        self.num_planes = num_planes
        self.grid_size = grid_size
        self.real = real

        if sequence_ids is None:
            # Auto-discover sequences: directories that contain 'velodyne_left'
            sequence_ids = [
                p.name
                for p in sorted(self.nclt_root.iterdir())
                if p.is_dir() and (p / "velodyne_left").exists()
            ]

        self.sequence_ids = sequence_ids

        # Per-sequence data
        self.sequence_data: Dict[str, Dict] = {}
        self.pcs: List[str] = []  # full paths to .bin files aligned with valid timestamps
        self.poses_all = np.empty((0, 12))  # concatenated 12D pose matrices rows for stats

        # Load or generate per-sequence timestamp and pose data
        ts_map: Dict[str, np.ndarray] = {}
        ps_map: Dict[str, np.ndarray] = {}
        vo_stats: Dict[str, Dict[str, np.ndarray]] = {}

        for seq in self.sequence_ids:
            seq_dir = self.nclt_root / seq
            lidar_dir = seq_dir / "velodyne_left"
            if not lidar_dir.exists():
                print(f"Warning: velodyne_left not found for sequence {seq}")
                continue

            h5_path = seq_dir / f"velodyne_left_{str(self.real)}.h5"

            if not osp.isfile(h5_path):
                # Build timestamp list from velodyne_left/*.bin
                vel_files = [f for f in os.listdir(str(lidar_dir)) if f.endswith('.bin')]
                if len(vel_files) == 0:
                    print(f"Warning: no .bin files in {lidar_dir}")
                    continue
                ts_raw = sorted([int(f[:-4]) for f in vel_files])

                gt_csv = seq_dir / f"groundtruth_{seq}.csv"
                if not gt_csv.exists():
                    print(f"Warning: groundtruth CSV not found for sequence {seq}: {gt_csv}")
                    continue

                # Generate valid timestamps and interpolated poses
                ts_map[seq] = filter_overflow_nclt(str(gt_csv), ts_raw)
                p = interpolate_pose_nclt(str(gt_csv), ts_map[seq])  # (n, 6) -> actually 4x4 then converted
                p = so3_to_euler_nclt(p)  # (n, 4, 4)
                ps_map[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)

                # Persist to h5 for speed next time
                with h5py.File(str(h5_path), 'w') as h5_file:
                    h5_file.create_dataset('valid_timestamps', data=np.asarray(ts_map[seq], dtype=np.int64))
                    h5_file.create_dataset('poses', data=ps_map[seq])
            else:
                with h5py.File(str(h5_path), 'r') as h5_file:
                    ts_map[seq] = h5_file['valid_timestamps'][...]
                    ps_map[seq] = h5_file['poses'][...]

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            # Build point cloud file list aligned with valid timestamps
            for t in ts_map[seq]:
                self.pcs.append(str(lidar_dir / f"{int(t)}.bin"))

            # Accumulate for stats
            self.poses_all = np.vstack((self.poses_all, ps_map[seq]))

        # Pose normalization stats (translation mean/std) stored at root
        pose_stats_file = self.nclt_root / 'NCLT_pose_stats.txt'
        if self.train:
            if self.poses_all.size == 0:
                raise RuntimeError("No poses collected; check NCLT root and sequences")
            mean_t = np.mean(self.poses_all[:, [3, 7, 11]], axis=0)
            std_t = np.std(self.poses_all[:, [3, 7, 11]], axis=0)
            np.savetxt(str(pose_stats_file), np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(str(pose_stats_file))

        # Convert per-sequence 12D poses to normalized translation + log-quaternion (6DOF) and store
        # process_poses already returns log-quaternions, so we keep them as-is
        self.poses_6dof = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
            
        for seq in self.sequence_ids:
            if seq not in ps_map:
                continue
            pss, rotation, _, _ = process_poses(
                poses_in=ps_map[seq],
                mean_t=mean_t,
                std_t=std_t,
                align_R=vo_stats[seq]['R'],
                align_t=vo_stats[seq]['t'],
                align_s=vo_stats[seq]['s'],
            )
            # process_poses returns pss as (n, 6): tx, ty, tz, log_qx, log_qy, log_qz
            # Keep as-is since it's already in log-quaternion format
            self.poses_6dof = np.vstack((self.poses_6dof, pss))
            self.rots = np.vstack((self.rots, rotation))
        
        # Set total number of samples (should match len(self.pcs) and len(self.poses_6dof))
        self.total_samples = len(self.pcs)
        if len(self.poses_6dof) != self.total_samples:
            raise RuntimeError(f"Mismatch: {len(self.pcs)} point clouds but {len(self.poses_6dof)} poses")
            
    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        scan_path = self.pcs[index]

        # Load raw point cloud (matching NCLTVelodyne_datagenerator.py)
        ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        #ptcld[:, 2] = -1 * ptcld[:, 2]  # Flip z coordinate
        scan = ptcld[:, :3]  # (N, 3) - raw points
        scan = np.ascontiguousarray(scan)

        # Get pose and rotation matrix
        pose_6dof = self.poses_6dof[index]  # [6] (tx, ty, tz, log_qx, log_qy, log_qz)
        rot = self.rots[index]  # (3, 3) rotation matrix

        # Transform points: scan_gt = (rot @ scan.T).T + pose[:3]
        # This applies rotation then translation (matching NCLTVelodyne_datagenerator.py line 131)
        pose_translation = pose_6dof[:3].reshape(1, 3)
        scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose_translation

        # Encode to XY planes on-the-fly
        try:
            planes, label_planes, z_bounds, xy_bounds = encode_to_multi_xy_planes(
                scan_path,
                label_file_path=None,
                bounds_zyx=self.bounds_zyx,
                num_planes=self.num_planes,
                grid_size=self.grid_size,
            )
            planes_tensor = torch.stack(planes).float()
        except Exception as e:
            print(f"Error processing scan {scan_path}: {e}")
            planes_tensor = torch.zeros(self.num_planes, self.grid_size, self.grid_size)

        # Convert to tensors
        scan_tensor = torch.from_numpy(scan).float()  # (N, 3)
        scan_gt_tensor = torch.from_numpy(scan_gt).float()  # (N, 3)
        pose_6dof_tensor = torch.from_numpy(pose_6dof).float()  # [6]

        if self.augmentation is not None:
            planes_tensor = self.augmentation(planes_tensor.unsqueeze(0)).squeeze(0)

        return {
            'planes': planes_tensor,           # [num_planes, grid_size, grid_size]
            'poses': pose_6dof_tensor,         # [6] (normalized tx, ty, tz, log_qx, log_qy, log_qz)
            'points': scan_tensor,             # [N, 3] raw point cloud
            'points_gt': scan_gt_tensor,       # [N, 3] transformed points (rot @ points + translation)
            'frame_id': index,
            'scan_path': scan_path,
        }


class PlaneTransform:
    def __init__(
        self,
        rotation_range: float = 0.0,
        translation_range: float = 0.0,
        noise_std: float = 0.0,
        flip_prob: float = 0.0,
    ) -> None:
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.flip_prob = flip_prob

    def __call__(self, planes: torch.Tensor) -> torch.Tensor:
        transformed = planes.clone()
        if self.noise_std > 0:
            transformed = transformed + torch.randn_like(transformed) * self.noise_std
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            transformed = torch.flip(transformed, dims=[2])
        if self.rotation_range > 0 and random.random() < 0.5:
            k = random.randint(1, 3)
            transformed = torch.rot90(transformed, k=k, dims=[1, 2])
        return transformed


def collate_nclt_batch(batch, max_points_per_cloud=50000):
    """
    Custom collate function for NCLT dataset that handles variable-sized point clouds.
    Pads or truncates point clouds to a fixed maximum size to prevent OOM.
    """
    import torch
    
    # Use fixed maximum instead of batch max to prevent OOM
    # Limit to prevent memory issues
    max_points = max_points_per_cloud
    
    batch_size = len(batch)
    
    # Collate all fields
    planes = torch.stack([item['planes'] for item in batch])  # [B, num_planes, grid_size, grid_size]
    poses = torch.stack([item['poses'] for item in batch])  # [B, 6] (log-quaternion)
    
    # Handle variable-sized point clouds by padding
    points_list = []
    points_gt_list = []
    
    for item in batch:
        num_points = item['points'].shape[0]
        
        # Pad or truncate to max_points
        if num_points < max_points:
            # Pad with zeros
            pad_size = max_points - num_points
            points_padded = torch.cat([
                item['points'],
                torch.zeros(pad_size, 3, dtype=item['points'].dtype)
            ], dim=0)
            points_gt_padded = torch.cat([
                item['points_gt'],
                torch.zeros(pad_size, 3, dtype=item['points_gt'].dtype)
            ], dim=0)
        else:
            # Truncate if too large - randomly sample to max_points
            if num_points > max_points:
                # Random sampling for better coverage
                indices = torch.randperm(num_points)[:max_points]
                points_padded = item['points'][indices]
                points_gt_padded = item['points_gt'][indices]
            else:
                points_padded = item['points']
                points_gt_padded = item['points_gt']
        
        points_list.append(points_padded)
        points_gt_list.append(points_gt_padded)
    
    points = torch.stack(points_list)  # [B, max_points, 3]
    points_gt = torch.stack(points_gt_list)  # [B, max_points, 3]
    
    # Optional: return frame_id and scan_path as lists
    frame_ids = [item['frame_id'] for item in batch]
    scan_paths = [item['scan_path'] for item in batch]
    
    return {
        'planes': planes,
        'poses': poses,
        'points': points,
        'points_gt': points_gt,
        'frame_id': frame_ids,
        'scan_path': scan_paths,
    }

def create_nclt_dataloaders(
    nclt_root: str = "/csehome/pydah/anirudh/SGLoc/NCLT",
    train_sequences: Optional[List[str]] = None,
    val_sequences: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    use_augmentation: bool = True,
    bounds_zyx: List[float] = [-11, 1, -55, 60, -40, 40],
    num_planes: int = 6,
    grid_size: int = 512,
    max_points_per_cloud: int = 30000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders for NCLT.
    
    Args:
        max_points_per_cloud: Maximum number of points per point cloud (default: 50000)
                              This prevents OOM by limiting point cloud size.
    """
    train_transform = None
    if use_augmentation:
        train_transform = PlaneTransform(
            rotation_range=0.0,
            translation_range=0.0,
            noise_std=0.01,
            flip_prob=0.5,
        )

    train_dataset = NCLTDataset(
        nclt_root=nclt_root,
        sequence_ids=train_sequences,
        train=True,
        valid=False,
        augmentation=train_transform,
        bounds_zyx=bounds_zyx,
        num_planes=num_planes,
        grid_size=grid_size,
    )

    val_dataset = NCLTDataset(
        nclt_root=nclt_root,
        sequence_ids=val_sequences,
        train=False,
        valid=True,
        augmentation=None,
        bounds_zyx=bounds_zyx,
        num_planes=num_planes,
        grid_size=grid_size,
    )

    # Create partial function with max_points_per_cloud parameter
    from functools import partial
    collate_fn_train = partial(collate_nclt_batch, max_points_per_cloud=max_points_per_cloud)
    collate_fn_val = partial(collate_nclt_batch, max_points_per_cloud=max_points_per_cloud)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_train,  # Use custom collate function with fixed max points
        prefetch_factor=2 if num_workers > 0 else None,  # Reduced from 4 to save memory
        persistent_workers=False,  # Disabled to free memory between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_val,  # Use custom collate function with fixed max points
        prefetch_factor=2 if num_workers > 0 else None,  # Reduced from 4 to save memory
        persistent_workers=False,  # Disabled to free memory between epochs
    )

    return train_loader, val_loader


