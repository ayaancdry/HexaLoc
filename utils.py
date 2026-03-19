#!/usr/bin/env python3
"""
Utility functions for pose processing, loss computation, and evaluation metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import math
import os
import sys
import importlib.util

# Import pose_util functions
current_dir = os.path.dirname(os.path.abspath(__file__))
pose_util_path = os.path.join(current_dir, 'pose_util.py')
if os.path.isfile(pose_util_path):
    spec = importlib.util.spec_from_file_location("pose_util", pose_util_path)
    pose_util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose_util)
    qexp = pose_util.qexp
    qexp_t = pose_util.qexp_t
    quaternion_to_matrix = pose_util.quaternion_to_matrix
else:
    raise ImportError(f"Could not find pose_util.py at {pose_util_path}")

def chamfer_distance_numpy(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point sets.
    """
    # Import locally to avoid hard dependency unless used
    try:
        from scipy.spatial import cKDTree as KDTree  # faster KDTree implementation
    except Exception:
        from scipy.spatial import KDTree  # fallback if cKDTree unavailable

    tree_B = KDTree(B)
    dist_A, _ = tree_B.query(A)  # distances from A to nearest in B

    tree_A = KDTree(A)
    dist_B, _ = tree_A.query(B)  # distances from B to nearest in A

    return float(np.mean(dist_A) + np.mean(dist_B))

def chamfer_distance_torch(points_src: torch.Tensor, 
                           points_tgt: torch.Tensor, 
                           squared: bool = False) -> torch.Tensor:
    """
    Compute differentiable Chamfer Distance between two point clouds using PyTorch.
    """
    if points_src.dim() == 2:
        points_src = points_src.unsqueeze(0)
    if points_tgt.dim() == 2:
        points_tgt = points_tgt.unsqueeze(0)

    # [B, Ns, Nt] pairwise distances
    dists = torch.cdist(points_src, points_tgt, p=2)  # Euclidean distances

    # For each point, distance to nearest neighbor in the other set
    min_src, _ = dists.min(dim=2)  # [B, Ns]
    min_tgt, _ = dists.min(dim=1)  # [B, Nt]

    if squared:
        min_src = min_src.pow(2)
        min_tgt = min_tgt.pow(2)

    # Mean over points, then average over batch
    cd_batch = min_src.mean(dim=1) + min_tgt.mean(dim=1)  # [B]
    return cd_batch.mean()


def quaternion_xyzw_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion in (x, y, z, w) format to rotation matrix.
    """
    single = False
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        single = True
    
    # Extract quaternion components (x, y, z, w)
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Normalize quaternion
    norm = torch.sqrt(qx**2 + qy**2 + qz**2 + qw**2 + 1e-12)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    
    # Convert to rotation matrix
    # R = [1-2(y^2+z^2)   2(xy-wz)     2(xz+wy)    ]
    #     [2(xy+wz)       1-2(x^2+z^2) 2(yz-wx)    ]
    #     [2(xz-wy)       2(yz+wx)     1-2(x^2+y^2)]
    
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw
    
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=-1),
        torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=-1),
        torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)
    
    if single:
        return R[0]
    return R

def point_cloud_transformation_loss(
    points_src: torch.Tensor,
    points_tgt: torch.Tensor,
    pose_7dof: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Transform source point cloud using 7DOF pose (translation + quaternion) and compute
    Euclidean distance loss between corresponding points.
    """
    # Handle batching
    batch_mode = points_src.dim() == 3
    if not batch_mode:
        points_src = points_src.unsqueeze(0)
        points_tgt = points_tgt.unsqueeze(0)
        pose_7dof = pose_7dof.unsqueeze(0)
    
    batch_size = points_src.shape[0]
    device = points_src.device
    
    # Extract translation and quaternion from 7DOF pose
    translation = pose_7dof[:, :3]  # [B, 3] (tx, ty, tz)
    quaternion = pose_7dof[:, 3:]  # [B, 4] (qx, qy, qz, qw)
    
    # Convert quaternion to rotation matrix
    # quaternion_xyzw_to_rotation_matrix expects [..., 4]
    R = quaternion_xyzw_to_rotation_matrix(quaternion)  # [B, 3, 3]
    
    # Transform source point cloud: points_transformed = R @ points_src.T + t
    # points_src: [B, N, 3]
    # R: [B, 3, 3]
    # Translation: [B, 3] -> [B, 1, 3] for broadcasting
    points_src_T = points_src.transpose(-1, -2)  # [B, 3, N]
    points_transformed_T = torch.bmm(R, points_src_T)  # [B, 3, N]
    points_transformed = points_transformed_T.transpose(-1, -2)  # [B, N, 3]
    points_transformed = points_transformed + translation.unsqueeze(1)  # [B, N, 3]
    
    # Compute Euclidean distance between corresponding points
    # points_transformed: [B, N, 3]
    # points_tgt: [B, N, 3]
    distances = torch.norm(points_transformed - points_tgt, dim=-1)  # [B, N]
    
    # Mask out zero-padded points (points that are exactly at origin are likely padding)
    # Check if point is non-zero in any dimension
    valid_mask_src = (torch.norm(points_src, dim=-1) > 1e-6)  # [B, N]
    valid_mask_tgt = (torch.norm(points_tgt, dim=-1) > 1e-6)  # [B, N]
    valid_mask = valid_mask_src & valid_mask_tgt  # Both should be valid
    
    # Mask distances to only include valid points
    distances = distances * valid_mask.float()  # [B, N]
    num_valid = valid_mask.sum().float()  # Total number of valid points
    
    # Reduce loss
    if reduction == 'mean':
        if num_valid > 0:
            loss = distances.sum() / num_valid  # Mean over valid points only
        else:
            loss = distances.mean()  # Fallback if all points are invalid
    elif reduction == 'sum':
        loss = distances.sum()
    elif reduction == 'none':
        loss = distances
        # Remove batch dimension if input wasn't batched
        if not batch_mode:
            loss = loss.squeeze(0)
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")
    
    return loss

class PointCloudTransformationLoss(nn.Module):
    """
    PyTorch nn.Module for point cloud transformation loss using 7DOF poses.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize point cloud transformation loss.
        """
        super(PointCloudTransformationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        points_src: torch.Tensor,
        points_tgt: torch.Tensor,
        pose_7dof: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the loss function.
        """
        return point_cloud_transformation_loss(
            points_src=points_src,
            points_tgt=points_tgt,
            pose_7dof=pose_7dof,
            reduction=self.reduction
        )

class AtLocCriterion(nn.Module):
    """
    AtLoc loss function for pose prediction.
    Combines translation and quaternion losses with learnable weighting parameters.
    
    The loss uses exponential weighting: exp(-sax) * translation_loss + sax + 
    exp(-saq) * quaternion_loss + saq, where sax and saq are learnable parameters.
    """
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        """
        Initialize AtLoc criterion.
        """
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        Compute AtLoc loss.
        """
        loss = (torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq)#/(torch.exp(-self.sax) + torch.exp(-self.saq))
        return loss

def log_quaternion_point_cloud_loss(
    points_gt: torch.Tensor,
    points_gt_transformed: torch.Tensor,
    translation: torch.Tensor,
    log_quaternion: torch.Tensor,
    reduction: str = 'mean',
    loss_type: str = 'l2'
) -> torch.Tensor:
    """
    Compute L1 or L2 loss between transformed point cloud and ground truth transformed point cloud.
    Uses log-quaternion representation and functions from pose_util.py.
    Uses qexp_t (PyTorch version) to maintain gradients.
    """
    # Handle batching
    batch_mode = points_gt.dim() == 3
    if not batch_mode:
        points_gt = points_gt.unsqueeze(0)
        points_gt_transformed = points_gt_transformed.unsqueeze(0)
        translation = translation.unsqueeze(0)
        log_quaternion = log_quaternion.unsqueeze(0)
    
    batch_size = points_gt.shape[0]
    device = points_gt.device
    
    # Convert log-quaternion to quaternion using qexp_t from pose_util (PyTorch version, maintains gradients)
    # log_quaternion: [B, 3]
    # qexp_t expects [B, 3] and returns [B, 4] with format [qw, qx, qy, qz] (real part first)
    quaternion = qexp_t(log_quaternion)  # [B, 4] (qw, qx, qy, qz)
    
    # Convert quaternion to rotation matrix using quaternion_to_matrix from pose_util
    # quaternion_to_matrix expects quaternions with real part last (..., 4)
    # qexp_t returns [B, 4] with format [qw, qx, qy, qz] (real part first)
    # But quaternion_to_matrix expects [..., 4] with format [qx, qy, qz, qw] (real part last)
    # We need to reorder to [qx, qy, qz, qw] for quaternion_to_matrix
    quaternion_reordered = torch.cat([quaternion[:, 1:], quaternion[:, :1]], dim=1)  # [B, 4] -> [qx, qy, qz, qw]
    R = quaternion_to_matrix(quaternion_reordered)  # [B, 3, 3]
    
    # Apply transformation: R @ points_gt.T + T
    # points_gt: [B, N, 3]
    # R: [B, 3, 3]
    # translation: [B, 3] -> [B, 1, 3] for broadcasting
    points_gt_T = points_gt.transpose(-1, -2)  # [B, 3, N]
    points_transformed_T = torch.bmm(R, points_gt_T)  # [B, 3, N]
    points_transformed = points_transformed_T.transpose(-1, -2)  # [B, N, 3]
    points_transformed = points_transformed + translation.unsqueeze(1)  # [B, N, 3]
    
    # Compute loss between transformed points and ground truth transformed points
    # points_transformed: [B, N, 3]
    # points_gt_transformed: [B, N, 3]
    if loss_type.lower() == 'l1':
        # L1 loss (Mean Absolute Error)
        abs_diff = torch.abs(points_transformed - points_gt_transformed)  # [B, N, 3]
        error_per_point = torch.sum(abs_diff, dim=-1)  # [B, N] - sum over x, y, z
        # Scale by 1e-2 to prevent gradient explosion (mean over 30k points already applied)
        error_per_point = error_per_point 
    elif loss_type.lower() == 'l2':
        # L2 loss (Mean Squared Error)
        squared_diff = (points_transformed - points_gt_transformed) ** 2  # [B, N, 3]
        error_per_point = torch.sum(squared_diff, dim=-1)  # [B, N] - sum over x, y, z
        # Scale by 1e-3 to prevent gradient explosion (squared values are larger, mean over 30k already applied)
        error_per_point = error_per_point 
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'l1' or 'l2'")
    
    # Mask out zero-padded points (points that are exactly at origin are likely padding)
    valid_mask_gt = (torch.norm(points_gt, dim=-1) > 1e-6)  # [B, N]
    valid_mask_gt_transformed = (torch.norm(points_gt_transformed, dim=-1) > 1e-6)  # [B, N]
    valid_mask = valid_mask_gt & valid_mask_gt_transformed  # Both should be valid
    
    # Mask error to only include valid points
    error_per_point = error_per_point * valid_mask.float()  # [B, N]
    num_valid = valid_mask.sum().float()  # Total number of valid points
    
    # Reduce loss
    if reduction == 'mean':
        if num_valid > 0:
            # Mean over all valid points: sum all errors and divide by number of valid points
            loss = (error_per_point.sum() / num_valid) 
        else:
            loss = error_per_point.mean()  # Fallback if all points are invalid
    elif reduction == 'sum':
        loss = error_per_point.sum()
    elif reduction == 'none':
        loss = error_per_point
        # Remove batch dimension if input wasn't batched
        if not batch_mode:
            loss = loss.squeeze(0)
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")
    
    return loss

class LogQuaternionPointCloudLoss(nn.Module):
    """
    PyTorch nn.Module for point cloud transformation loss using log-quaternion representation.
    """
    def __init__(self, reduction: str = 'mean', loss_type: str = 'l2'):
        """
        Initialize log-quaternion point cloud transformation loss.
        """
        super(LogQuaternionPointCloudLoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type.lower()
        if self.loss_type not in ['l1', 'l2']:
            raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'l1' or 'l2'")
    
    def forward(
        self,
        points_gt: torch.Tensor,
        points_gt_transformed: torch.Tensor,
        translation: torch.Tensor,
        log_quaternion: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the loss function.
        """
        return log_quaternion_point_cloud_loss(
            points_gt=points_gt,
            points_gt_transformed=points_gt_transformed,
            translation=translation,
            log_quaternion=log_quaternion,
            reduction=self.reduction,
            loss_type=self.loss_type
        )

def pose_matrix_to_6dof(pose_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x4 pose matrix to 6DOF representation.
    """
    if pose_matrix.dim() == 2:
        # Single pose
        translation = pose_matrix[:, 3]  # [3]
        rotation_matrix = pose_matrix[:, :3]  # [3, 3]
        
        # Convert rotation matrix to Euler angles (ZYX order)
        # This is a simplified conversion - for production use, consider using proper quaternion conversion
        sy = torch.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = torch.atan2(-rotation_matrix[2, 0], sy)
            z = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = torch.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = torch.atan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        rotation = torch.stack([x, y, z])
        return torch.cat([translation, rotation])
    else:
        # Batch of poses
        batch_size = pose_matrix.shape[0]
        translations = pose_matrix[:, :, 3]  # [batch_size, 3]
        rotation_matrices = pose_matrix[:, :, :3]  # [batch_size, 3, 3]
        
        # Convert rotation matrices to Euler angles
        rotations = []
        for i in range(batch_size):
            R = rotation_matrices[i]
            sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                x = torch.atan2(R[2, 1], R[2, 2])
                y = torch.atan2(-R[2, 0], sy)
                z = torch.atan2(R[1, 0], R[0, 0])
            else:
                x = torch.atan2(-R[1, 2], R[1, 1])
                y = torch.atan2(-R[2, 0], sy)
                z = 0
            
            rotations.append(torch.stack([x, y, z]))
        
        rotations = torch.stack(rotations)  # [batch_size, 3]
        return torch.cat([translations, rotations], dim=1)  # [batch_size, 6]

def pose_6dof_to_matrix(pose_6dof: torch.Tensor) -> torch.Tensor:
    """
    Convert 6DOF pose to 3x4 pose matrix.
    """
    if pose_6dof.dim() == 1:
        # Single pose
        translation = pose_6dof[:3]  # [3]
        rotation_euler = pose_6dof[3:]  # [3] (roll, pitch, yaw)
        
        # Convert Euler angles to rotation matrix (ZYX order)
        x, y, z = rotation_euler[0], rotation_euler[1], rotation_euler[2]
        
        # Rotation matrices for each axis
        Rx = torch.tensor([[1, 0, 0],
                          [0, torch.cos(x), -torch.sin(x)],
                          [0, torch.sin(x), torch.cos(x)]], device=pose_6dof.device, dtype=pose_6dof.dtype)
        
        Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)],
                          [0, 1, 0],
                          [-torch.sin(y), 0, torch.cos(y)]], device=pose_6dof.device, dtype=pose_6dof.dtype)
        
        Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                          [torch.sin(z), torch.cos(z), 0],
                          [0, 0, 1]], device=pose_6dof.device, dtype=pose_6dof.dtype)
        
        # Combined rotation matrix
        R = torch.mm(torch.mm(Rz, Ry), Rx)
        
        # Create 3x4 pose matrix
        pose_matrix = torch.cat([R, translation.unsqueeze(1)], dim=1)
        return pose_matrix
    else:
        # Batch of poses
        batch_size = pose_6dof.shape[0]
        translations = pose_6dof[:, :3]  # [batch_size, 3]
        rotation_eulers = pose_6dof[:, 3:]  # [batch_size, 3]
        
        pose_matrices = []
        for i in range(batch_size):
            translation = translations[i]
            x, y, z = rotation_eulers[i, 0], rotation_eulers[i, 1], rotation_eulers[i, 2]
            
            # Rotation matrices for each axis
            Rx = torch.tensor([[1, 0, 0],
                              [0, torch.cos(x), -torch.sin(x)],
                              [0, torch.sin(x), torch.cos(x)]], device=pose_6dof.device, dtype=pose_6dof.dtype)
            
            Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)],
                              [0, 1, 0],
                              [-torch.sin(y), 0, torch.cos(y)]], device=pose_6dof.device, dtype=pose_6dof.dtype)
            
            Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                              [torch.sin(z), torch.cos(z), 0],
                              [0, 0, 1]], device=pose_6dof.device, dtype=pose_6dof.dtype)
            
            # Combined rotation matrix
            R = torch.mm(torch.mm(Rz, Ry), Rx)
            
            # Create 3x4 pose matrix
            pose_matrix = torch.cat([R, translation.unsqueeze(1)], dim=1)
            pose_matrices.append(pose_matrix)
        
        return torch.stack(pose_matrices)  # [batch_size, 3, 4]

def pose_to_transformation_matrix(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert pose to 4x4 transformation matrix.
    """
    if pose.dim() == 2:
        # Single pose
        T = torch.eye(4, device=pose.device, dtype=pose.dtype)
        T[:3, :] = pose
        return T
    else:
        # Batch of poses
        batch_size = pose.shape[0]
        T = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        T[:, :3, :] = pose
        return T


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix/matrices to unit quaternions (w, x, y, z).
    """
    single = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        single = True

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    qw = torch.empty_like(trace)
    qx = torch.empty_like(trace)
    qy = torch.empty_like(trace)
    qz = torch.empty_like(trace)

    cond0 = trace > 0

    # Case 0: trace positive
    s0 = torch.sqrt(trace[cond0] + 1.0) * 2  # s = 4*qw
    qw[cond0] = 0.25 * s0
    qx[cond0] = (R[cond0, 2, 1] - R[cond0, 1, 2]) / s0
    qy[cond0] = (R[cond0, 0, 2] - R[cond0, 2, 0]) / s0
    qz[cond0] = (R[cond0, 1, 0] - R[cond0, 0, 1]) / s0

    # Remaining cases where trace <= 0
    cond1 = (~cond0) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s1 = torch.sqrt(1.0 + R[cond1, 0, 0] - R[cond1, 1, 1] - R[cond1, 2, 2]) * 2
    qw[cond1] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s1
    qx[cond1] = 0.25 * s1
    qy[cond1] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / s1
    qz[cond1] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / s1

    cond2 = (~cond0) & (~cond1) & (R[..., 1, 1] > R[..., 2, 2])
    s2 = torch.sqrt(1.0 + R[cond2, 1, 1] - R[cond2, 0, 0] - R[cond2, 2, 2]) * 2
    qw[cond2] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / s2
    qx[cond2] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s2
    qy[cond2] = 0.25 * s2
    qz[cond2] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / s2

    cond3 = (~cond0) & (~cond1) & (~cond2)
    s3 = torch.sqrt(1.0 + R[cond3, 2, 2] - R[cond3, 0, 0] - R[cond3, 1, 1]) * 2
    qw[cond3] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / s3
    qx[cond3] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / s3
    qy[cond3] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s3
    qz[cond3] = 0.25 * s3

    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    # Normalize to guard against numerical drift
    quat = quat / torch.clamp(quat.norm(dim=-1, keepdim=True), min=1e-12)

    if single:
        return quat[0]
    return quat

def transformation_matrix_to_tq(T: torch.Tensor) -> torch.Tensor:
    """
    Convert a transformation matrix to [tx, ty, tz, qw, qx, qy, qz].
    """
    if T.dim() == 2:
        # Single matrix
        if T.shape == (4, 4):
            R = T[:3, :3]
            t = T[:3, 3]
        elif T.shape == (3, 4):
            R = T[:, :3]
            t = T[:, 3]
        else:
            raise ValueError("Expected T of shape [3,4] or [4,4]")
        q = rotation_matrix_to_quaternion(R)
        return torch.cat([t, q], dim=0)
    else:
        # Batched
        if T.shape[-2:] == (4, 4):
            R = T[..., :3, :3]
            t = T[..., :3, 3]
        elif T.shape[-2:] == (3, 4):
            R = T[..., :3, :3]
            t = T[..., :3, 3]
        else:
            raise ValueError("Expected T of shape [...,3,4] or [...,4,4]")
        q = rotation_matrix_to_quaternion(R)
        return torch.cat([t, q], dim=-1)


def save_checkpoint(filepath: str, 
                   model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int, 
                   val_loss: float, 
                   config: Dict,
                   best_val_loss: float = None,
                   train_losses: List[float] = None,
                   val_losses: List[float] = None,
                   atloc_criterion: nn.Module = None):
    """
    Save training checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss if best_val_loss is not None else val_loss,
        'train_losses': train_losses if train_losses is not None else [],
        'val_losses': val_losses if val_losses is not None else [],
        'config': config
    }
    
    # Save AtLocCriterion state if provided
    if atloc_criterion is not None:
        checkpoint['atloc_criterion_state_dict'] = atloc_criterion.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str) -> Dict:
    """
    Load training checkpoint.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint



