#!/usr/bin/env python3
"""
V100 Training Monitor - Track GPU usage during QuDiffuse training
"""

import subprocess
import time
import sys

def monitor_training():
    """Monitor GPU usage during training"""
    print("🔧 V100 Training Monitor Started")
    print("=" * 50)
    
    for i in range(60):  # Monitor for 60 seconds
        try:
            # Get nvidia-smi output
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                gpu_util, mem_used, mem_total, temp, power = output.split(', ')
                
                print(f"[{i+1:2d}/60] GPU: {gpu_util:>3}% | Mem: {mem_used:>5}/{mem_total} MB | Temp: {temp:>2}°C | Power: {power:>3}W")
            else:
                print(f"[{i+1:2d}/60] nvidia-smi error")
                
        except Exception as e:
            print(f"[{i+1:2d}/60] Monitor error: {e}")
        
        time.sleep(1)
    
    print("\n✅ Monitoring complete")

if __name__ == "__main__":
    monitor_training() 