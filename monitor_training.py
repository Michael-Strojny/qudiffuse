#!/usr/bin/env python3
"""
Training Monitor Script

Monitors the binary autoencoder training progress by checking:
- Process status
- Output files
- Log parsing
- Progress tracking
"""

import os
import time
import subprocess
import glob
from datetime import datetime

def check_process_status():
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        train_processes = [
            line for line in result.stdout.split('\n') 
            if 'train_binary_autoencoder.py' in line and 'grep' not in line
        ]
        
        if train_processes:
            print(f"✅ Training process active:")
            for proc in train_processes:
                parts = proc.split()
                pid = parts[1]
                cpu_time = parts[9]
                mem_usage = f"{float(parts[5])/1024:.1f}MB"
                print(f"   PID: {pid}, CPU Time: {cpu_time}, Memory: {mem_usage}")
            return True
        else:
            print("❌ No training process found")
            return False
            
    except Exception as e:
        print(f"⚠️ Error checking process: {e}")
        return False

def check_output_files():
    """Check for training output files."""
    print("\n📁 Output Files:")
    
    # Check for model checkpoints
    checkpoints = glob.glob("*.pth")
    if checkpoints:
        print(f"   Model checkpoints: {len(checkpoints)}")
        for ckpt in sorted(checkpoints):
            size = os.path.getsize(ckpt) / (1024*1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(ckpt))
            print(f"      {ckpt}: {size:.1f}MB, {mtime.strftime('%H:%M:%S')}")
    else:
        print("   No model checkpoints yet")
    
    # Check for visualizations
    images = glob.glob("reconstruction_*.png")
    if images:
        print(f"   Visualizations: {len(images)}")
        for img in sorted(images):
            mtime = datetime.fromtimestamp(os.path.getmtime(img))
            print(f"      {img}: {mtime.strftime('%H:%M:%S')}")
    else:
        print("   No visualizations yet")
    
    # Check for log files
    logs = glob.glob("*.log")
    if logs:
        print(f"   Log files: {len(logs)}")
    else:
        print("   No log files found")

def estimate_progress():
    """Estimate training progress based on time."""
    try:
        result = subprocess.run(
            ["ps", "-o", "etime=", "-p", "$(pgrep -f train_binary_autoencoder.py)"], 
            capture_output=True, 
            text=True, 
            shell=True
        )
        
        if result.returncode == 0:
            elapsed = result.stdout.strip()
            print(f"\n⏱️ Training Runtime: {elapsed}")
            
            # Estimate progress (assuming 25 epochs, ~2-3 minutes per epoch)
            parts = elapsed.split(':')
            if len(parts) >= 2:
                minutes = int(parts[-2])
                seconds = int(parts[-1])
                total_seconds = minutes * 60 + seconds
                
                estimated_epochs = total_seconds / 120  # ~2 min per epoch
                progress_pct = min(100, (estimated_epochs / 25) * 100)
                
                print(f"   Estimated progress: {progress_pct:.1f}% ({estimated_epochs:.1f}/25 epochs)")
        
    except Exception as e:
        print(f"⚠️ Error estimating progress: {e}")

def show_recent_activity():
    """Show recent file system activity."""
    print("\n📊 Recent Activity:")
    
    try:
        # Check data directory
        if os.path.exists('data'):
            print(f"   Data directory: {len(os.listdir('data'))} items")
        
        # Check for CIFAR-10 download
        cifar_path = 'data/cifar-10-batches-py'
        if os.path.exists(cifar_path):
            print("   ✅ CIFAR-10 dataset available")
        else:
            print("   ⏳ CIFAR-10 dataset downloading...")
        
        # Check virtual environment
        if os.path.exists('.venv_312'):
            print("   ✅ Python 3.12 environment active")
        
    except Exception as e:
        print(f"⚠️ Error checking activity: {e}")

def main():
    """Main monitoring function."""
    print("🔍 Binary Autoencoder Training Monitor")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check process status
    is_running = check_process_status()
    
    # Check output files
    check_output_files()
    
    # Estimate progress
    if is_running:
        estimate_progress()
    
    # Show recent activity
    show_recent_activity()
    
    print("\n" + "=" * 50)
    
    if is_running:
        print("🚀 Training is active! Check back in a few minutes for updates.")
        print("📁 Expected outputs:")
        print("   - Model checkpoints: best_binary_autoencoder.pth")
        print("   - Epoch checkpoints: binary_autoencoder_epoch_N.pth")
        print("   - Visualizations: reconstruction_epoch_N.png")
    else:
        print("⏹️ Training has completed or stopped.")
        print("📁 Check for final outputs and logs.")

if __name__ == "__main__":
    main() 