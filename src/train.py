#!/usr/bin/env python3
"""
Training script for plane-based localization neural network.
Handles training with on-the-fly data processing from KITTI dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import json
import os
import sys
import math
import importlib.util
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # plane_project root
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import model from src directory
from src.model import create_model, count_parameters

from data_loaders.dataloader_nclt_logq import create_nclt_dataloaders

# Import utils from parent directory (plane_project root)
spec = importlib.util.spec_from_file_location("plane_utils", os.path.join(parent_dir, "utils.py"))
plane_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plane_utils)

PointCloudTransformationLoss = plane_utils.PointCloudTransformationLoss
AtLocCriterion = plane_utils.AtLocCriterion
LogQuaternionPointCloudLoss = plane_utils.LogQuaternionPointCloudLoss
save_checkpoint = plane_utils.save_checkpoint
load_checkpoint = plane_utils.load_checkpoint


class Trainer:
    """
    Trainer class for plane-based localization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # plane_project root
        
        # Create model
        self.model = create_model(config['model']).to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Create L1 loss function for direct pose prediction (kept for compatibility, not used)
        self.criterion = nn.L1Loss(reduction='mean')
        print(f"Using AtLoc criterion for pose prediction")
        
        # Create AtLoc criterion as fallback/alternative loss for pose prediction
        # Uses learnable weighting between translation and quaternion losses
        self.atloc_criterion = AtLocCriterion(
            t_loss_fn=nn.L1Loss(),
            q_loss_fn=nn.L1Loss(),
            sax=config['training'].get('atloc_sax', 0.0),
            saq=config['training'].get('atloc_saq', 0.0),
            learn_beta=config['training'].get('learn_beta', False)
        ).to(self.device)
        
        # Create optimizer
        # Include AtLocCriterion parameters if learn_beta is enabled
        optimizer_params = list(self.model.parameters())
        if config['training'].get('learn_beta', False):
            optimizer_params += list(self.atloc_criterion.parameters())
        
        self.optimizer = optim.Adam(
            optimizer_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_decay_factor']
        )
        
        # Create dataloaders - check if using NCLT or KITTI
        dataset_type = config['data'].get('dataset_type', 'nclt')  # Default to NCLT
        
            # Use NCLT dataloader with point clouds
        train_sequences = config['data'].get('train_sequences', None)
        val_sequences = config['data'].get('val_sequences', None)
        # Get max_points_per_cloud from config to prevent OOM
        max_points = config['data'].get('max_points_per_cloud', 50000)
        
        self.train_loader, self.val_loader = create_nclt_dataloaders(
            nclt_root=nclt_root,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            use_augmentation=config['training']['use_augmentation'],
            bounds_zyx=config['data']['bounds_zyx'],
            num_planes=config['data']['num_planes'],
            grid_size=config['data']['grid_size'],
            max_points_per_cloud=max_points
        )
        
        # Create output directory (relative to project root)
        output_dir = config['training']['output_dir']
        if not os.path.isabs(output_dir):
            # If relative path, make it relative to project root
            output_dir = os.path.join(parent_dir, output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tensorboard writer
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Gradient statistics log file
        self.grad_stats_file = self.output_dir / 'gradient_stats.txt'
        self._init_gradient_stats_file()
        
        # Load checkpoint if specified
        if config['training']['resume_from']:
            self.load_checkpoint(config['training']['resume_from'])
    
    def _init_gradient_stats_file(self):
        """Initialize the gradient statistics log file with header."""
        with open(self.grad_stats_file, 'w') as f:
            f.write("GRADIENT STATISTICS LOG\n")
            # Get model name from config if available, otherwise use model class name
            model_name = self.config['model'].get('name', self.model.__class__.__name__)
            f.write(f"Model: {model_name}\n")
            f.write(f"Total Parameters: {count_parameters(self.model):,}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
    
    def _save_gradient_stats(self, epoch: int, batch_idx: int, loss: float, 
                            grad_stats: Dict, learning_rate: float):
        """
        Save detailed gradient statistics to file.
        """
        with open(self.grad_stats_file, 'a') as f:
            f.write(f"Epoch: {epoch:4d} | Batch: {batch_idx:4d} | Loss: {loss:.6f} | LR: {learning_rate:.8f}\n")
            
            # Summary statistics
            f.write(f"\nSUMMARY STATISTICS:\n")
            f.write(f"Total Gradient Norm:     {grad_stats['total_norm']:.8f}\n")
            f.write(f"Average Gradient Norm:   {grad_stats['avg_norm']:.8f}\n")
            f.write(f"Max Gradient Norm:       {grad_stats['max_norm']:.8f} (Layer: {grad_stats['max_norm_layer']})\n")
            f.write(f"Min Gradient Norm:       {grad_stats['min_norm']:.8f} (Layer: {grad_stats['min_norm_layer']})\n")
            
            if 'total_norm_after_clip' in grad_stats:
                f.write(f"  Gradient Norm After Clip: {grad_stats['total_norm_after_clip']:.8f}\n")
            
            # Detailed layer-by-layer statistics
            f.write(f"\nLAYER-BY-LAYER GRADIENT NORMS:\n")
            f.write(f"{'Layer Name':<60} {'Grad Norm':<15} {'Param Shape':<20}\n")
            
            # Get all layers with their parameter shapes
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    param_shape = str(list(param.shape))
                    f.write(f"{name:<60} {grad_norm:<15.8f} {param_shape:<20}\n")
                else:
                    param_shape = str(list(param.shape))
                    f.write(f"{name:<60} {'NO_GRAD':<15} {param_shape:<20}\n")
            
            # Top 20 layers with largest gradients
            f.write(f"\nTOP 20 LAYERS BY GRADIENT NORM:\n")
            f.write(f"{'Rank':<6} {'Layer Name':<60} {'Grad Norm':<15}\n")
            for i, (name, norm) in enumerate(grad_stats['top_layers'], 1):
                f.write(f"{i:<6} {name:<60} {norm:<15.8f}\n")
            
            # Bottom 20 layers with smallest gradients (non-zero)
            f.write(f"\nBOTTOM 20 LAYERS BY GRADIENT NORM (non-zero):\n")
            f.write(f"{'Rank':<6} {'Layer Name':<60} {'Grad Norm':<15}\n")
            for i, (name, norm) in enumerate(grad_stats['bottom_layers'], 1):
                f.write(f"{i:<6} {name:<60} {norm:<15.8f}\n")
            
            f.write(f"\n")
    
    def _compute_gradient_stats(self) -> Dict:
        """
        Compute gradient statistics for all model parameters.
        """
        grad_norms = []
        grad_info = []
        
        # Compute gradient norm for each parameter
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
                grad_info.append((name, grad_norm))
            else:
                # Parameters without gradients (shouldn't happen, but handle it)
                grad_info.append((name, 0.0))
        
        if len(grad_norms) == 0:
            # No gradients computed yet
            return {
                'total_norm': 0.0,
                'avg_norm': 0.0,
                'max_norm': 0.0,
                'min_norm': 0.0,
                'max_norm_layer': 'none',
                'min_norm_layer': 'none',
                'top_layers': [],
                'bottom_layers': []
            }
        
        # Compute total norm (L2 norm of all gradients concatenated)
        # This computes sqrt(sum(grad_norm^2 for all parameters))
        total_norm = math.sqrt(sum(gn ** 2 for gn in grad_norms))
        
        # Compute statistics
        avg_norm = sum(grad_norms) / len(grad_norms)
        max_norm = max(grad_norms)
        min_norm = min(grad_norms)
        
        # Find layers with max and min norms
        max_norm_layer = max(grad_info, key=lambda x: x[1])[0]
        min_norm_layer = min(grad_info, key=lambda x: x[1] if x[1] > 0 else float('inf'))[0]
        
        # Sort by gradient norm
        top_layers = sorted(grad_info, key=lambda x: x[1], reverse=True)
        bottom_layers = sorted([(n, v) for n, v in grad_info if v > 0], key=lambda x: x[1])
        
        return {
            'total_norm': total_norm,
            'avg_norm': avg_norm,
            'max_norm': max_norm,
            'min_norm': min_norm,
            'max_norm_layer': max_norm_layer,
            'min_norm_layer': min_norm_layer,
            'top_layers': top_layers,
            'bottom_layers': bottom_layers
        }

    def _compute_weight_stats(self) -> Dict:
        """
        Compute statistics for all model weights.
        """
        weight_norms = []
        weight_info = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                norm = param.data.norm(2).item()
                weight_norms.append(norm)
                weight_info.append((name, norm))
        
        if not weight_norms:
            return {}
            
        total_norm = math.sqrt(sum(n ** 2 for n in weight_norms))
        avg_norm = sum(weight_norms) / len(weight_norms)
        max_norm = max(weight_norms)
        min_norm = min(weight_norms)
        
        max_norm_layer = max(weight_info, key=lambda x: x[1])[0]
        min_norm_layer = min(weight_info, key=lambda x: x[1])[0]
        
        top_layers = sorted(weight_info, key=lambda x: x[1], reverse=True)
        bottom_layers = sorted(weight_info, key=lambda x: x[1])
        
        return {
            'total_norm': total_norm,
            'avg_norm': avg_norm,
            'max_norm': max_norm,
            'min_norm': min_norm,
            'max_norm_layer': max_norm_layer,
            'min_norm_layer': min_norm_layer,
            'top_layers': top_layers,
            'bottom_layers': bottom_layers
        }

    def _print_weight_stats(self, header: str = "[Model Weights] Statistics"):
        """Print detailed weight statistics."""
        stats = self._compute_weight_stats()
        if not stats:
            return

        print(f"\n{header}")
        print(f"  Total Weight Norm: {stats['total_norm']:.6f}")
        print(f"  Average Weight Norm: {stats['avg_norm']:.6f}")
        print(f"  Max Weight Norm: {stats['max_norm']:.6f} (layer: {stats['max_norm_layer']})")
        print(f"  Min Weight Norm: {stats['min_norm']:.6f} (layer: {stats['min_norm_layer']})")
        
        print(f"  Top 10 Layers by Weight Norm:")
        for i, (name, norm) in enumerate(stats['top_layers'][:10], 1):
            print(f"    {i}. {name}: {norm:.6f}")
            
        print(f"  Bottom 10 Layers by Weight Norm:")
        for i, (name, norm) in enumerate(stats['bottom_layers'][:10], 1):
            print(f"    {i}. {name}: {norm:.6f}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        # Reset used samples for new epoch (if dataset supports it)
        if hasattr(self.train_loader.dataset, 'reset_epoch'):
            self.train_loader.dataset.reset_epoch()
        
        self.model.train()
        total_loss = 0.0
        total_trans_error = 0.0
        total_rot_error = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            planes = batch['planes'].to(self.device)
            poses = batch['poses'].to(self.device)  # [B, 6] 6DOF poses (log-quaternion)
            
            # Get point clouds for loss computation (if available)
            points_src = batch.get('points', None)  # [B, N, 3] raw point cloud
            points_tgt = batch.get('points_gt', None)  # [B, N, 3] transformed target points
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_poses = self.model(planes)  # [B, 6] predicted 6DOF poses (log-quaternion)
            
            # Compute loss using AtLoc criterion
            loss = self.atloc_criterion(pred_poses, poses)
            
            # Compute translation and rotation errors for progress bar
            pred_translation = pred_poses[:, :3]  # [B, 3]
            target_translation = poses[:, :3]  # [B, 3]
            trans_error = torch.mean(torch.norm(pred_translation - target_translation, dim=1)).item()
            
            pred_log_q = pred_poses[:, 3:]  # [B, 3] log-quaternion
            target_log_q = poses[:, 3:]  # [B, 3] log-quaternion
            # Rotation error: L2 norm of log-quaternion difference (approximation, in degrees)
            rot_error = torch.mean(torch.norm(pred_log_q - target_log_q, dim=1)).item() * 180.0 / math.pi
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (this computes total norm efficiently)
            # clip_grad_norm_ returns the total norm BEFORE clipping
            if self.config['training']['grad_clip'] > 0:
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            else:
                # Compute total norm for progress bar even without clipping
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    float('inf')
                )
            
            # Compute norm after clipping and detailed stats only when logging (before optimizer.step())
            grad_stats = None
            grad_norm_after_clip = None
            if batch_idx % self.config['training']['log_interval'] == 0:
                # Compute detailed gradient statistics only when logging
                grad_stats = self._compute_gradient_stats()
                # Compute norm after clipping if gradient clipping is enabled
                if self.config['training']['grad_clip'] > 0:
                    grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        float('inf')
                    )
                    grad_stats['total_norm_after_clip'] = grad_norm_after_clip.item()
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_trans_error += trans_error
            total_rot_error += rot_error
            avg_loss = total_loss / (batch_idx + 1)
            avg_trans_error = total_trans_error / (batch_idx + 1)
            avg_rot_error = total_rot_error / (batch_idx + 1)
            
            # Update progress bar with loss, grad norm, translation error, and rotation error
            pbar.set_postfix({
                'loss': f'{avg_loss:.6f}',
                'grad_norm': f'{grad_norm_before_clip.item():.4f}',
                'trans_err': f'{avg_trans_error:.4f}',
                'rot_err': f'{avg_rot_error:.4f}°'
            })
            
            # Log to tensorboard and print gradient info
            # Only compute expensive gradient stats when logging
            if batch_idx % self.config['training']['log_interval'] == 0:
                global_step = self.epoch * num_batches + batch_idx
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/LR', current_lr, global_step)
                
                # Log gradient statistics to tensorboard
                self.writer.add_scalar('Gradients/TotalNorm', grad_stats['total_norm'], global_step)
                self.writer.add_scalar('Gradients/AvgNorm', grad_stats['avg_norm'], global_step)
                self.writer.add_scalar('Gradients/MaxNorm', grad_stats['max_norm'], global_step)
                self.writer.add_scalar('Gradients/MinNorm', grad_stats['min_norm'], global_step)
                self.writer.add_scalar('Gradients/TotalNormBeforeClip', grad_norm_before_clip.item(), global_step)
                if 'total_norm_after_clip' in grad_stats:
                    self.writer.add_scalar('Gradients/TotalNormAfterClip', grad_stats['total_norm_after_clip'], global_step)
                
                # Save detailed gradient statistics to file
                self._save_gradient_stats(
                    epoch=self.epoch,
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    grad_stats=grad_stats,
                    learning_rate=current_lr
                )
                
                # Print detailed gradient information
                print(f"\n[Epoch {self.epoch}, Batch {batch_idx}] Gradient Statistics:")
                print(f"  Total Gradient Norm: {grad_stats['total_norm']:.6f}")
                print(f"  Average Gradient Norm: {grad_stats['avg_norm']:.6f}")
                print(f"  Max Gradient Norm: {grad_stats['max_norm']:.6f} (layer: {grad_stats['max_norm_layer']})")
                print(f"  Min Gradient Norm: {grad_stats['min_norm']:.6f} (layer: {grad_stats['min_norm_layer']})")
                if 'total_norm_after_clip' in grad_stats:
                    print(f"  Gradient Norm After Clip: {grad_stats['total_norm_after_clip']:.6f}")
                print(f"  Gradient stats saved to: {self.grad_stats_file}")
                
                # Print top 10 layers with largest gradients
                print(f"  Top 10 Layers by Gradient Norm:")
                for i, (name, norm) in enumerate(grad_stats['top_layers'][:10], 1):
                    print(f"    {i}. {name}: {norm:.6f}")
                
                # Print bottom 10 layers with smallest gradients
                print(f"  Bottom 10 Layers by Gradient Norm:")
                for i, (name, norm) in enumerate(grad_stats['bottom_layers'][:10], 1):
                    print(f"    {i}. {name}: {norm:.6f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss and metrics dictionary
        """
        self.model.eval()
        #self.model.train()
        total_loss = 0.0
        all_metrics = {
            'translation_error': [],
            'rotation_error': []
        }
        
        # Import qexp from pose_util for metrics computation (once, outside the loop)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir_metrics = os.path.dirname(current_dir)  # plane_project root
        pose_util_path = os.path.join(parent_dir_metrics, 'pose_util.py')
        qexp_metrics = None
        if os.path.isfile(pose_util_path):
            spec_pose = importlib.util.spec_from_file_location("pose_util_metrics", pose_util_path)
            pose_util_metrics = importlib.util.module_from_spec(spec_pose)
            spec_pose.loader.exec_module(pose_util_metrics)
            qexp_metrics = pose_util_metrics.qexp
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                planes = batch['planes'].to(self.device)
                poses = batch['poses'].to(self.device)  # [B, 6] 6DOF poses (log-quaternion)
                
                # Get point clouds for loss computation (if available)
                points_src = batch.get('points', None)
                points_tgt = batch.get('points_gt', None)
                
                # Forward pass
                pred_poses = self.model(planes)  # [B, 6] predicted 6DOF poses (log-quaternion)
                
                # Compute loss using AtLoc criterion
                loss = self.atloc_criterion(pred_poses, poses)
                
                
                total_loss += loss.item()
                
                # Compute simple metrics (translation and quaternion errors)
                # Translation error
                pred_trans = pred_poses[:, :3]  # [B, 3]
                target_trans = poses[:, :3]  # [B, 3]
                
                # Ensure same dtype for computation
                if pred_trans.dtype != target_trans.dtype:
                    print(f"WARNING: dtype mismatch! pred_trans: {pred_trans.dtype}, target_trans: {target_trans.dtype}")
                    target_trans = target_trans.to(pred_trans.dtype)
                
                trans_error = torch.norm(pred_trans - target_trans, dim=1).cpu().numpy()
                all_metrics['translation_error'].extend(trans_error.tolist())
                
                # Debug: Print first batch stats to compare with test.py
                if batch_idx == 0:
                    # Get frame IDs and scan paths from batch to verify we're loading the same samples
                    frame_ids = batch.get('frame_id', None)
                    scan_paths = batch.get('scan_path', None)
                    
                    print(f"\nDEBUG - First batch translation error stats (train.py):")
                    print(f"  Batch size: {pred_trans.shape[0]}")
                    if frame_ids is not None:
                        if isinstance(frame_ids, list):
                            print(f"  Frame IDs (first 3): {frame_ids[:3]}")
                        else:
                            print(f"  Frame IDs (first 3): {list(frame_ids)[:3] if hasattr(frame_ids, '__iter__') else [frame_ids]}")
                    if scan_paths is not None:
                        if isinstance(scan_paths, list):
                            print(f"  Scan paths (first 1): {scan_paths[0] if len(scan_paths) > 0 else 'N/A'}")
                        else:
                            print(f"  Scan paths (first 1): {list(scan_paths)[0] if hasattr(scan_paths, '__iter__') else scan_paths}")
                    print(f"  pred_trans dtype: {pred_trans.dtype}, device: {pred_trans.device}")
                    print(f"  target_trans dtype: {target_trans.dtype}, device: {target_trans.device}")
                    print(f"  pred_poses dtype: {pred_poses.dtype}, device: {pred_poses.device}")
                    print(f"  poses dtype: {poses.dtype}, device: {poses.device}")
                    print(f"  planes dtype: {planes.dtype}, device: {planes.device}")
                    print(f"  pred_trans sample: {pred_trans[0].cpu().numpy()}")
                    print(f"  target_trans sample: {target_trans[0].cpu().numpy()}")
                    print(f"  trans_error sample: {trans_error[0]}")
                    print(f"  trans_error dtype: {trans_error.dtype}")
                    print(f"  batch_avg_trans_error: {np.mean(trans_error)}")
                    print(f"  pred_trans norm: {torch.norm(pred_trans[0]).item():.6f}")
                    print(f"  target_trans norm: {torch.norm(target_trans[0]).item():.6f}")
                    print(f"  Model training mode: {self.model.training}")
                    sys.stdout.flush()
                
                # Rotation error (convert log-quaternions to quaternions, then compute distance)
                pred_log_q = pred_poses[:, 3:]  # [B, 3] log-quaternion
                target_log_q = poses[:, 3:]  # [B, 3] log-quaternion
                
                if qexp_metrics is not None:
                    # Convert log-quaternions to quaternions using qexp
                    pred_log_q_np = pred_log_q.detach().cpu().numpy()
                    target_log_q_np = target_log_q.detach().cpu().numpy()
                    pred_quat_list = []
                    target_quat_list = []
                    for i in range(pred_log_q.shape[0]):
                        q_pred = qexp_metrics(pred_log_q_np[i])  # (4,) numpy array (qw, qx, qy, qz)
                        q_target = qexp_metrics(target_log_q_np[i])
                        pred_quat_list.append(q_pred)
                        target_quat_list.append(q_target)
                    
                    pred_quat = torch.from_numpy(np.stack(pred_quat_list)).to(pred_log_q.device).float()  # [B, 4] (qw, qx, qy, qz)
                    target_quat = torch.from_numpy(np.stack(target_quat_list)).to(target_log_q.device).float()  # [B, 4] (qw, qx, qy, qz)
                    
                    # Reorder to (qx, qy, qz, qw) format for quaternion distance computation
                    pred_quat_reordered = torch.cat([pred_quat[:, 1:], pred_quat[:, :1]], dim=1)  # [B, 4] -> [qx, qy, qz, qw]
                    target_quat_reordered = torch.cat([target_quat[:, 1:], target_quat[:, :1]], dim=1)  # [B, 4] -> [qx, qy, qz, qw]
                    
                    # Normalize quaternions
                    pred_quat_norm = pred_quat_reordered / (pred_quat_reordered.norm(dim=1, keepdim=True) + 1e-12)
                    target_quat_norm = target_quat_reordered / (target_quat_reordered.norm(dim=1, keepdim=True) + 1e-12)
                    # Quaternion distance: 1 - |q1 · q2|
                    quat_dot = torch.abs(torch.sum(pred_quat_norm * target_quat_norm, dim=1))
                    rot_error = torch.acos(torch.clamp(quat_dot, -1.0, 1.0)) * 2.0  # In radians
                    rot_error_deg = rot_error * 180.0 / math.pi  # Convert to degrees
                    all_metrics['rotation_error'].extend(rot_error_deg.cpu().numpy().tolist())
                else:
                    # Fallback: use L2 norm of log-quaternion difference as approximation
                    rot_error_deg = torch.norm(pred_log_q - target_log_q, dim=1).cpu().numpy() * 180.0 / math.pi
                    all_metrics['rotation_error'].extend(rot_error_deg.tolist())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = np.mean(values)
            # Debug: Print metric computation info
            if key == 'translation_error':
                print(f"\nDEBUG - Translation error computation (train.py):")
                print(f"  Total samples: {len(values)}")
                print(f"  Number of batches: {len(self.val_loader)}")
                print(f"  Batch size: {len(values) / len(self.val_loader) if len(self.val_loader) > 0 else 0:.1f}")
                print(f"  Min error: {np.min(values):.6f}")
                print(f"  Max error: {np.max(values):.6f}")
                print(f"  Mean error: {avg_metrics[key]:.6f}")
                print(f"  Std error: {np.std(values):.6f}")
                print(f"  Median error: {np.median(values):.6f}")
                sys.stdout.flush()
        
        return avg_loss, avg_metrics
    
    def train(self):
        """
        Main training loop.
        """
        print("Starting training...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Print weight statistics at the start of each epoch
            self._print_weight_stats(header=f"[Epoch {epoch} Start] Weight Statistics:")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate only every 5 epochs
            run_validation = (epoch % 5 == 0)
            
            if run_validation:
                # Validate
                val_loss, val_metrics = self.validate_epoch()
                
                # Update best validation loss
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                # Skip validation, use previous metrics
                val_loss = self.val_losses[-1] if self.val_losses else float('inf')
                val_metrics = {
                    'translation_error': 0.0,
                    'rotation_error': 0.0
                }
                is_best = False
            
            # Update learning rate (StepLR doesn't need validation loss)
            self.scheduler.step()
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.6f}")
            if run_validation:
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Learning Rate: {current_lr:.6f}")
                print(f"  Val Translation Error: {val_metrics['translation_error']:.4f}")
                print(f"  Val Rotation Error: {val_metrics['rotation_error']:.4f}")
            else:
                print(f"  Val Loss: (skipped - validation every 5 epochs)")
                print(f"  Learning Rate: {current_lr:.6f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            if run_validation:
                self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
                self.writer.add_scalar('Epoch/ValTranslationError', val_metrics['translation_error'], epoch)
                self.writer.add_scalar('Epoch/ValRotationError', val_metrics['rotation_error'], epoch)
            
            # Additional useful metrics
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            if run_validation:
                self.writer.add_scalar('Loss_Ratio/Val_Train', val_loss / train_loss if train_loss > 0 else 0, epoch)
            
            
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                val_loss,
                self.config,
                self.best_val_loss,
                self.train_losses,
                self.val_losses,
                atloc_criterion=self.atloc_criterion
            )
            
            if is_best:
                best_path = self.output_dir / 'best_model.pth'
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    self.config,
                    self.best_val_loss,
                    self.train_losses,
                    self.val_losses,
                    atloc_criterion=self.atloc_criterion
                )
                print(f"  New best model saved!")

        print("Training completed!")
        self.writer.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Use strict=True to match test.py behavior and catch any mismatches
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load AtLocCriterion state if available (for backward compatibility with older checkpoints)
        if 'atloc_criterion_state_dict' in checkpoint:
            self.atloc_criterion.load_state_dict(checkpoint['atloc_criterion_state_dict'])
            print(f"AtLocCriterion parameters loaded from checkpoint")
            if self.config['training'].get('learn_beta', False):
                print(f"  sax: {self.atloc_criterion.sax.item():.6f}, saq: {self.atloc_criterion.saq.item():.6f}")
        else:
            print(f"Warning: AtLocCriterion state not found in checkpoint, using initialized values")
        
        # Override scheduler step_size and gamma from config
        self.scheduler.step_size = self.config['training']['lr_step_size']
        self.scheduler.gamma = self.config['training']['lr_decay_factor']
        
        self.epoch = checkpoint['epoch'] + 1
        
        # Handle missing keys in older checkpoints
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        # Override learning rate with config value or use the lower one
        config_lr = self.config['training']['learning_rate']
        checkpoint_lr = self.optimizer.param_groups[0]['lr']
        
        # Check if we should force use config LR or use the lower one
        force_config_lr = self.config['training'].get('force_config_lr', False)
        
        if force_config_lr:
            # Always use config learning rate
            final_lr = config_lr
            print(f"Force using config learning rate: {final_lr:.6f}")
        else:
            # Use the lower learning rate for safety
            final_lr = min(config_lr, checkpoint_lr)
            print(f"Using lower learning rate for safety: {final_lr:.6f}")
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = final_lr
        
        # Override weight decay with config value
        config_weight_decay = self.config['training']['weight_decay']
        checkpoint_weight_decay = self.optimizer.param_groups[0].get('weight_decay', 0.0)
        
        # Update optimizer weight_decay to use config value
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = config_weight_decay
        
        print(f"Resumed from epoch {self.epoch}, best val loss: {self.best_val_loss:.6f}")
        print(f"Learning rate: Config={config_lr:.6f}, Checkpoint={checkpoint_lr:.6f}, Using={final_lr:.6f}")
        print(f"Weight decay: Config={config_weight_decay:.6f}, Checkpoint={checkpoint_weight_decay:.6f}, Using={config_weight_decay:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Train plane-based localization model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # plane_project_cbam3 root
    
    # Load configuration - check if path is relative or absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        # If relative, try config_files/ directory first, then current directory
        config_files_path = os.path.join(parent_dir, "config_files", config_path)
        if os.path.exists(config_files_path):
            config_path = config_files_path
        elif os.path.exists(os.path.join(parent_dir, config_path)):
            config_path = os.path.join(parent_dir, config_path)
        else:
            config_path = os.path.join(current_dir, config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override resume path if provided
    if args.resume:
        config['training']['resume_from'] = args.resume
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
