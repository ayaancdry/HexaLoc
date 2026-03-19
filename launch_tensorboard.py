#!/usr/bin/env python3
"""
Script to launch TensorBoard for visualizing training progress.
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

def launch_tensorboard(log_dir="outputs/logs", port=6006, start_idx=None, end_idx=None):
    """
    Launch TensorBoard with the specified log directory and port.
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Error: Log directory {log_path} does not exist!")
        print("Make sure you have run training first to generate logs.")
        return
    
    # Find all log files
    log_files = sorted(
        [f for f in log_path.iterdir() if f.name.startswith('events.out.tfevents')],
        key=lambda x: x.stat().st_mtime,
        reverse=True  # Most recent first
    )
    
    if not log_files:
        print(f"Error: No log files found in {log_path}")
        return
    
    # Filter by range if specified
    temp_dir = None
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(log_files) - 1
        
        # Validate range
        if start < 0:
            start = 0
        if end >= len(log_files):
            end = len(log_files) - 1
        if start > end:
            print(f"Error: Invalid range: start ({start}) > end ({end})")
            return
        
        selected_files = log_files[start:end+1]
        print(f"\nFound {len(log_files)} total log files")
        print(f"Selecting files {start} to {end} (most recent = 0):")
        for i, f in enumerate(selected_files, start=start):
            mtime = os.path.getmtime(f)
            print(f"  [{i}] {f.name} (modified: {os.path.getctime(f):.0f})")
        
        # Create temporary directory with symlinks to selected files
        temp_dir = tempfile.mkdtemp(prefix="tensorboard_logs_")
        try:
            for f in selected_files:
                os.symlink(f.absolute(), os.path.join(temp_dir, f.name))
            
            actual_log_dir = temp_dir
            print(f"\nUsing temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error creating temporary directory: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
    else:
        actual_log_dir = log_path
        print(f"\nFound {len(log_files)} log files (using all)")
        print("Most recent files:")
        for i, f in enumerate(log_files[:5]):  # Show first 5
            print(f"  [{i}] {f.name}")
        if len(log_files) > 5:
            print(f"  ... and {len(log_files) - 5} more")
    
    print(f"\nLaunching TensorBoard...")
    print(f"Log directory: {Path(actual_log_dir).absolute()}")
    print(f"Port: {port}")
    print(f"URL: http://localhost:{port}")
    print("\nPress Ctrl+C to stop TensorBoard")

    
    try:
        # Launch TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(actual_log_dir),
            "--port", str(port),
            "--host", "0.0.0.0"  # Allow external access
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")
        print("Make sure TensorBoard is installed: pip install tensorboard")
    finally:
        # Clean up temporary directory if created
        if temp_dir is not None and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for training visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_tensorboard.py
        """
    )
    parser.add_argument("--logdir", default="outputs/logs", help="Path to log directory")
    parser.add_argument("--port", type=int, default=6006, help="Port number")
    parser.add_argument(
        "--range", 
        nargs=2, 
        type=int, 
        metavar=('START', 'END'),
        help="Index range of log files to use (most recent = 0). Example: --range 0 4 uses 5 most recent files"
    )
    
    args = parser.parse_args()
    
    start_idx = None
    end_idx = None
    if args.range:
        start_idx, end_idx = args.range
    
    launch_tensorboard(args.logdir, args.port, start_idx, end_idx)
