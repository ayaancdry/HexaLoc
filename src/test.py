#!/usr/bin/env python3
"""
Test script for running inference on NCLT dataset sequentially.
Loads a checkpoint, runs inference on all samples in order, and saves results to a .txt file.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
import sys
import math
import importlib.util
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # plane_project_cbam3 root
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import utils module explicitly from parent directory to avoid conflicts
utils_path = os.path.join(parent_dir, 'utils.py')
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
load_checkpoint = utils_module.load_checkpoint

# Import model from src directory
from src.model import create_model

# Import dataloaders from data_loaders directory
from data_loaders.dataloader_nclt_logq import NCLTDataset, collate_nclt_batch

# Import loss functions from utils
AtLocCriterion = utils_module.AtLocCriterion
LogQuaternionPointCloudLoss = utils_module.LogQuaternionPointCloudLoss

def quaternion_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion distance between two quaternions.
    """
    # Normalize quaternions
    q1_norm = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-12)
    q2_norm = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-12)
    
    # Quaternion dot product
    if q1.dim() == 1:
        dot = torch.abs(torch.sum(q1_norm * q2_norm))
    else:
        dot = torch.abs(torch.sum(q1_norm * q2_norm, dim=1))
    
    # Clamp to avoid numerical issues
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Angle between quaternions: 2 * arccos(|q1 · q2|)
    angle = 2.0 * torch.acos(dot)
    
    return angle

def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
    atloc_criterion: nn.Module = None,
) -> Tuple[float, Dict, List]:
    """
    Validate for one epoch (standalone function matching train.py validation).
    """
    model.eval()
    total_loss = 0.0
    num_batches_with_loss = 0
    all_metrics = {
        'translation_error': [],
        'rotation_error': []
    }
    per_frame_details = []  # Store per-frame details for output file
    
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
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            planes = batch['planes'].to(device)
            poses = batch['poses'].to(device)  # [B, 6] 6DOF poses (log-quaternion)
            
            # Forward pass
            pred_poses = model(planes)  # [B, 6] predicted 6DOF poses (log-quaternion)
            
            # Compute loss matching train.py validation: L1 loss for translation and rotation
            # Split into translation and log-quaternion components
            pred_translation = pred_poses[:, :3]  # [B, 3] translation
            pred_log_quaternion = pred_poses[:, 3:]  # [B, 3] log-quaternion (in radians)
            target_translation = poses[:, :3]  # [B, 3] translation
            target_log_quaternion = poses[:, 3:]  # [B, 3] log-quaternion (in radians)
            
            # Compute L1 loss separately for translation and rotation, then combine
            translation_loss = torch.mean(torch.abs(pred_translation - target_translation))
            rotation_loss = torch.mean(torch.abs(pred_log_quaternion - target_log_quaternion))
            loss = translation_loss + 60.0 * rotation_loss
            
            total_loss += loss.item()
            num_batches_with_loss += 1
            
            # Compute simple metrics (translation and quaternion errors) for current batch
            # Translation error (per sample in batch)
            pred_trans = pred_poses[:, :3]  # [B, 3]
            target_trans = poses[:, :3]  # [B, 3]
            
            # Ensure same dtype for computation
            if pred_trans.dtype != target_trans.dtype:
                print(f"WARNING: dtype mismatch! pred_trans: {pred_trans.dtype}, target_trans: {target_trans.dtype}")
                target_trans = target_trans.to(pred_trans.dtype)
            
            trans_error = torch.norm(pred_trans - target_trans, dim=1).cpu().numpy()  # [B] per-sample errors
            batch_avg_trans_error = np.mean(trans_error)  # Average for current batch only
            all_metrics['translation_error'].extend(trans_error.tolist())  # Store all for final average
            
            # Get frame IDs and scan paths from batch
            batch_size = batch['planes'].shape[0]
            frame_ids = batch.get('frame_id', [batch_idx * batch_size + i for i in range(batch_size)])
            scan_paths = batch.get('scan_path', [None] * batch_size)
            # Ensure they are lists
            if not isinstance(frame_ids, list):
                if hasattr(frame_ids, 'tolist'):
                    frame_ids = frame_ids.tolist()
                else:
                    frame_ids = [frame_ids] if not hasattr(frame_ids, '__iter__') else list(frame_ids)
            if not isinstance(scan_paths, list):
                if hasattr(scan_paths, 'tolist'):
                    scan_paths = scan_paths.tolist()
                else:
                    scan_paths = [scan_paths] if not hasattr(scan_paths, '__iter__') else list(scan_paths)
            
            # Debug: Print first batch stats to compare with train.py
            if batch_idx == 0:
                print(f"\nDEBUG - First batch translation error stats (test.py):")
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
                print(f"  batch_avg_trans_error: {batch_avg_trans_error}")
                print(f"  pred_trans norm: {torch.norm(pred_trans[0]).item():.6f}")
                print(f"  target_trans norm: {torch.norm(target_trans[0]).item():.6f}")
                print(f"  Model training mode: {model.training}")
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
                rot_error_deg = rot_error * 180.0 / math.pi  # Convert to degrees [B] per-sample errors
                batch_avg_rot_error = rot_error_deg.mean().item()  # Average for current batch only
                all_metrics['rotation_error'].extend(rot_error_deg.cpu().numpy().tolist())  # Store all for final average
                
                # Store per-frame details
                pred_trans_np = pred_trans.detach().cpu().numpy()
                target_trans_np = target_trans.detach().cpu().numpy()
                pred_quat_np = pred_quat_reordered.detach().cpu().numpy()  # [B, 4] (qx, qy, qz, qw)
                target_quat_np = target_quat_reordered.detach().cpu().numpy()  # [B, 4] (qx, qy, qz, qw)
                
                rot_error_np = rot_error.detach().cpu().numpy()  # [B] in radians
                for i in range(pred_log_q.shape[0]):
                    per_frame_details.append({
                        'frame_id': frame_ids[i] if i < len(frame_ids) else batch_idx * batch['planes'].shape[0] + i,
                        'scan_path': scan_paths[i] if i < len(scan_paths) else None,
                        'trans_error': float(trans_error[i]),
                        'rot_error_deg': float(rot_error_deg[i].item()),
                        'rot_error_rad': float(rot_error_np[i]),
                        'pred_trans': pred_trans_np[i].tolist(),
                        'target_trans': target_trans_np[i].tolist(),
                        'pred_quat': pred_quat_np[i].tolist(),  # (qx, qy, qz, qw)
                        'target_quat': target_quat_np[i].tolist(),  # (qx, qy, qz, qw)
                    })
            else:
                # Fallback: use L2 norm of log-quaternion difference as approximation
                # But still compute quaternions for output using numpy qexp if available
                rot_error_deg = torch.norm(pred_log_q - target_log_q, dim=1).cpu().numpy() * 180.0 / math.pi  # [B] per-sample errors
                batch_avg_rot_error = np.mean(rot_error_deg)  # Average for current batch only
                all_metrics['rotation_error'].extend(rot_error_deg.tolist())  # Store all for final average
                
                # Store per-frame details (with approximate rotation error)
                pred_trans_np = pred_trans.detach().cpu().numpy()
                target_trans_np = target_trans.detach().cpu().numpy()
                rot_error_rad = rot_error_deg * math.pi / 180.0
                
                # Try to compute quaternions even in fallback case using numpy qexp
                pred_log_q_np = pred_log_q.detach().cpu().numpy()
                target_log_q_np = target_log_q.detach().cpu().numpy()
                pred_quat_list = []
                target_quat_list = []
                
                # Use numpy qexp if available, otherwise set to None
                if qexp_metrics is not None:
                    for i in range(pred_log_q.shape[0]):
                        q_pred = qexp_metrics(pred_log_q_np[i])  # (4,) numpy array (qw, qx, qy, qz)
                        q_target = qexp_metrics(target_log_q_np[i])
                        # Reorder to (qx, qy, qz, qw) format
                        pred_quat_list.append([q_pred[1], q_pred[2], q_pred[3], q_pred[0]])
                        target_quat_list.append([q_target[1], q_target[2], q_target[3], q_target[0]])
                else:
                    # If qexp is not available, we can't compute quaternions
                    pred_quat_list = [None] * pred_log_q.shape[0]
                    target_quat_list = [None] * pred_log_q.shape[0]
                
                for i in range(pred_log_q.shape[0]):
                    per_frame_details.append({
                        'frame_id': frame_ids[i] if i < len(frame_ids) else batch_idx * batch['planes'].shape[0] + i,
                        'scan_path': scan_paths[i] if i < len(scan_paths) else None,
                        'trans_error': float(trans_error[i]),
                        'rot_error_deg': float(rot_error_deg[i]),
                        'rot_error_rad': float(rot_error_rad[i]),
                        'pred_trans': pred_trans_np[i].tolist(),
                        'target_trans': target_trans_np[i].tolist(),
                        'pred_quat': pred_quat_list[i],
                        'target_quat': target_quat_list[i],
                    })
            
            # Update progress bar with batch-specific metrics (not running averages)
            # Final averages will be computed at the end from all stored values
            if qexp_metrics is not None:
                pbar.set_postfix({
                    'L1_loss': f'{loss.item():.6f}',
                    'trans_err': f'{batch_avg_trans_error:.4f}',
                    'rot_err_deg': f'{batch_avg_rot_error:.4f}'
                })
            else:
                pbar.set_postfix({
                    'L1_loss': f'{loss.item():.6f}',
                    'trans_err': f'{batch_avg_trans_error:.4f}',
                    'rot_err_deg': f'{batch_avg_rot_error:.4f} (approx)'
                })
    
    # Compute average loss
    avg_loss = total_loss / num_batches_with_loss if num_batches_with_loss > 0 else 0.0
    
    # Compute average metrics
    avg_metrics = {}
    for key, values in all_metrics.items():
        avg_metrics[key] = np.mean(values)
        # Debug: Print metric computation info
        if key == 'translation_error':
            print(f"\nDEBUG - Translation error computation (test.py):")
            print(f"  Total samples: {len(values)}")
            print(f"  Number of batches: {num_batches_with_loss}")
            print(f"  Batch size: {len(values) / num_batches_with_loss if num_batches_with_loss > 0 else 0:.1f}")
            print(f"  Min error: {np.min(values):.6f}")
            print(f"  Max error: {np.max(values):.6f}")
            print(f"  Mean error: {avg_metrics[key]:.6f}")
            print(f"  Std error: {np.std(values):.6f}")
            print(f"  Median error: {np.median(values):.6f}")
            sys.stdout.flush()
    
    return avg_loss, avg_metrics, per_frame_details
        
def run_inference(
    checkpoint_path: str,
    nclt_root: str = "/csehome/pydah/anirudh/SGLoc/NCLT",
    sequence_ids: List[str] = None,
    output_file: str = "inference_results.txt",
    batch_size: int = 1,
    bounds_zyx: List[float] = [-11, 1, -55, 60, -40, 40],
    num_planes: int = 8,
    grid_size: int = 512,
    max_points_per_cloud: int = 30000,
):
    """
    Run inference on NCLT dataset sequentially and save results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        
        # Use validation sequences from checkpoint config if sequence_ids not provided
        if sequence_ids is None:
            sequence_ids = data_config.get('val_sequences', None)
            if sequence_ids:
                print(f"Using validation sequences from checkpoint config: {sequence_ids}")
        
        # Use batch size from checkpoint config to match training
        training_config = config.get('training', {})
        if batch_size == 1:  # Only override if using default
            config_batch_size = training_config.get('batch_size', 12)
            print(f"Using batch size from checkpoint config: {config_batch_size} (to match training)")
            batch_size = config_batch_size
        
        # Use NCLT root from checkpoint config to match training
        config_nclt_root = data_config.get('nclt_root', None)
        if config_nclt_root:
            print(f"Using NCLT root from checkpoint config: {config_nclt_root} (to match training)")
            nclt_root = config_nclt_root
        else:
            print(f"Warning: No nclt_root in checkpoint config, using provided/default: {nclt_root}")
        
        # Use config values if parameters are None
        if num_planes is None:
            num_planes = model_config.get('num_planes', 8)
        if grid_size is None:
            grid_size = model_config.get('grid_size', 512)
        if bounds_zyx is None:
            bounds_zyx = data_config.get('bounds_zyx', [-11, 1, -55, 60, -40, 40])
        
        # Ensure model_config has all required keys
        model_config = model_config.copy()
        model_config.setdefault('num_planes', num_planes)
        model_config.setdefault('grid_size', grid_size)
        model_config.setdefault('feature_dim', 1024)
        model_config.setdefault('hidden_dim', 1024)
        model_config.setdefault('block_type', 'basic')
        model_config.setdefault('dropout_rate', 0.1)
        model_config.setdefault('num_layers', 4)
    else:
        # Fallback to defaults if config not in checkpoint
        print("Warning: No config found in checkpoint, using defaults")
        if num_planes is None:
            num_planes = 8
        if grid_size is None:
            grid_size = 512
        if bounds_zyx is None:
            bounds_zyx = [-11, 1, -55, 60, -40, 40]
        
        model_config = {
            'num_planes': num_planes,
            'grid_size': grid_size,
            'feature_dim': 128,
            'hidden_dim': 256,
            'block_type': 'basic',
            'dropout_rate': 0.1,
            'num_layers': 4
        }
    
    # Create model
    print("Creating model...")
    print(f"Model config: {model_config}")
    model = create_model(model_config).to(device)
    
    # Load model weights
    print("Loading model weights...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        print("This might indicate a model architecture mismatch!")
        raise
    
    model.eval()  # Ensure model is in eval mode (important for batch norm, dropout, etc.)
    
    # Verify model is in eval mode
    print(f"Model training mode: {model.training} (should be False)")
    
    # Debug: Check if model has batch normalization layers
    bn_layers = [name for name, module in model.named_modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if bn_layers:
        print(f"Model has {len(bn_layers)} batch normalization layers")
        print(f"  First 5 BN layers: {bn_layers[:5]}")
        print(f"  Note: Batch norm behavior may differ with different batch sizes even in eval mode")
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Create dataset (sequential, no augmentation)
    print("Creating dataset...")
    print(f"Using sequences: {sequence_ids if sequence_ids else 'auto-discover'}")
    dataset = NCLTDataset(
        nclt_root=nclt_root,
        sequence_ids=sequence_ids,
        train=False,  # Use validation mode (loads stats from training)
        valid=True,
        augmentation=None,  # No augmentation for testing
        bounds_zyx=bounds_zyx,
        num_planes=num_planes,
        grid_size=grid_size,
        real=False,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dataset sequences: {dataset.sequence_ids}")
    print(f"Note: Ensure these sequences match train.py validation sequences for fair comparison")
    
    # Debug: Check normalization stats if available
    pose_stats_file = Path(nclt_root) / 'NCLT_pose_stats.txt'
    if pose_stats_file.exists():
        stats = np.loadtxt(str(pose_stats_file))
        mean_t = stats[0]
        std_t = stats[1]
        print(f"\nDEBUG - Normalization stats from {pose_stats_file}:")
        print(f"  mean_t: {mean_t}")
        print(f"  std_t: {std_t}")
        print(f"  Note: Translation errors are computed in normalized space.")
        print(f"  To convert to unnormalized space, multiply by std_t: {std_t}")
    else:
        print(f"\nWARNING: Normalization stats file not found at {pose_stats_file}")
    
    # Create dataloader (sequential, no shuffle)
    # Use num_workers=0 to ensure deterministic ordering and match train.py validation behavior
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential processing
        num_workers=12,  # Set to 0 for deterministic ordering (parallel workers can cause non-deterministic order)
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: collate_nclt_batch(batch, max_points_per_cloud=max_points_per_cloud),
    )
    
    # Run validation using validate_epoch function (matching train.py)
    # Loss computation matches train.py: translation_loss + 60.0 * rotation_loss (both L1)
    print("Running validation using validate_epoch function...")
    avg_loss, avg_metrics, per_frame_details = validate_epoch(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=None,  # Not used - loss computed directly matching train.py
        atloc_criterion=None,  # Not used - loss computed directly matching train.py
    )
    
    # Extract metrics
    avg_trans_error = avg_metrics['translation_error']
    avg_rot_error = avg_metrics['rotation_error']
    num_samples = len(dataset)
    
    print(f"\nValidation completed!")
    print(f"Total samples: {num_samples}")
    print(f"Average loss: {avg_loss:.6f}")
    print(f"Average translation error: {avg_trans_error:.6f} (in normalized coordinate space)")
    print(f"Average rotation error: {avg_rot_error:.4f} degrees")
    print(f"\nNote: Translation and rotation errors are computed on normalized poses.")
    print(f"Ensure you're comparing the same validation sequences as train.py.")
    print(f"Train.py uses: {sequence_ids if sequence_ids else 'check config val_sequences'}")
    
    # Save results to file
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        # Write header with summary
        f.write("NCLT Dataset Inference Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {nclt_root}\n")
        f.write(f"Sequences: {sequence_ids if sequence_ids else 'auto-discovered'}\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Average translation error: {avg_trans_error:.6f}\n")
        f.write(f"Average rotation error: {avg_rot_error:.4f} degrees\n")
        f.write("\n")
        
        # Write detailed results table
        f.write("Detailed Results:\n")
        f.write(f"{'Frame ID':<10} {'Scan Path':<50} {'Trans Error':<15} {'Rot Error (deg)':<15}\n")
        
        for detail in per_frame_details:
            scan_path = detail['scan_path'] if detail['scan_path'] else "N/A"
            # Truncate scan path if too long
            if len(scan_path) > 48:
                scan_path = "..." + scan_path[-45:]
            f.write(f"{detail['frame_id']:<10} {scan_path:<50} {detail['trans_error']:<15.6f} {detail['rot_error_deg']:<15.4f}\n")
        f.write("\n")
        f.write("\n")
        
        # Write detailed per-frame information
        for detail in per_frame_details:
            scan_path = detail['scan_path'] if detail['scan_path'] else "N/A"
            f.write(f"Frame {detail['frame_id']} ({scan_path}):\n")
            f.write(f"  Translation Error: {detail['trans_error']:.6f}\n")
            f.write(f"  Rotation Error: {detail['rot_error_deg']:.4f} degrees ({detail['rot_error_rad']:.6f} radians)\n")
            f.write(f"  Predicted Translation: [{detail['pred_trans'][0]:.6f}, {detail['pred_trans'][1]:.6f}, {detail['pred_trans'][2]:.6f}]\n")
            f.write(f"  Ground Truth Translation: [{detail['target_trans'][0]:.6f}, {detail['target_trans'][1]:.6f}, {detail['target_trans'][2]:.6f}]\n")
            if detail['pred_quat'] is not None and detail['target_quat'] is not None:
                f.write(f"Predicted Quaternion: [{detail['pred_quat'][0]:.6f}, {detail['pred_quat'][1]:.6f}, {detail['pred_quat'][2]:.6f}, {detail['pred_quat'][3]:.6f}]\n")
                f.write(f"Ground Truth Quaternion: [{detail['target_quat'][0]:.6f}, {detail['target_quat'][1]:.6f}, {detail['target_quat'][2]:.6f}, {detail['target_quat'][3]:.6f}]\n")
            else:
                # If quaternions are not available, write a note
                f.write(f"Predicted Quaternion: [Not available - qexp function not found]\n")
                f.write(f"Ground Truth Quaternion: [Not available - qexp function not found]\n")
            f.write("\n")
    
    print(f"Results saved to: {output_file}")
    
    return {
        'avg_loss': avg_loss,
        'avg_trans_error': avg_trans_error,
        'avg_rot_error': avg_rot_error,
        'num_samples': num_samples,
        'metrics': avg_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Run inference on NCLT dataset sequentially")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoint_epoch_40.pth",
        help="Path to checkpoint file (default: outputs/checkpoint_epoch_40.pth)"
    )
    parser.add_argument(
        "--nclt_root",
        type=str,
        default="/csehome/pydah/anirudh/SGLoc/NCLT",
        help="Root directory of NCLT dataset"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequence IDs to test (e.g., 2012-03-31). If not specified, auto-discovers all sequences."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.txt",
        help="Output file path (default: inference_results.txt)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1 for sequential processing)"
    )
    parser.add_argument(
        "--num_planes",
        type=int,
        default=None,
        help="Number of XY planes (default: from checkpoint config)"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Grid size for each plane (default: from checkpoint config)"
    )
    
    args = parser.parse_args()
    
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # plane_project root
    
    # Resolve checkpoint path - check if relative or absolute
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        # If relative, try outputs/ directory in project root
        outputs_path = os.path.join(parent_dir, "outputs", checkpoint_path)
        if os.path.exists(outputs_path):
            checkpoint_path = outputs_path
        elif os.path.exists(os.path.join(parent_dir, checkpoint_path)):
            checkpoint_path = os.path.join(parent_dir, checkpoint_path)
        else:
            checkpoint_path = os.path.join(current_dir, checkpoint_path)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print(f"Available checkpoints in outputs/:")
        outputs_dir = os.path.join(parent_dir, "outputs")
        if os.path.exists(outputs_dir):
            for f in os.listdir(outputs_dir):
                if f.endswith(".pth"):
                    print(f"  - outputs/{f}")
        return
    
    # Update checkpoint path in args
    args.checkpoint = checkpoint_path
    
    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        nclt_root=args.nclt_root,
        sequence_ids=args.sequences,
        output_file=args.output,
        batch_size=args.batch_size,
        num_planes=args.num_planes,
        grid_size=args.grid_size,
    )

if __name__ == "__main__":
    main()

