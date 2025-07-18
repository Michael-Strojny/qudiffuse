#!/usr/bin/env python3
"""
GPU Verification Test - DEFINITIVE PROOF
Tests if operations actually run on GPU or are secretly running on CPU
"""

import torch
import time
import subprocess
import sys

def run_nvidia_smi():
    """Get nvidia-smi output"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception: {e}"

def test_gpu_vs_cpu():
    """Definitive test to prove GPU vs CPU usage"""
    
    print("🔬 DEFINITIVE GPU VERIFICATION TEST")
    print("=" * 50)
    
    # Check initial state
    print("📊 Initial State:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name()}")
    
    nvidia_smi_output = run_nvidia_smi()
    print(f"   nvidia-smi: {nvidia_smi_output}")
    print()
    
    # Test 1: Force CPU computation
    print("🧪 TEST 1: CPU-Only Computation")
    print("-" * 30)
    device_cpu = torch.device('cpu')
    
    start_time = time.time()
    a_cpu = torch.randn(2000, 2000, device=device_cpu)
    b_cpu = torch.randn(2000, 2000, device=device_cpu)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    print(f"   CPU Computation Time: {cpu_time:.3f}s")
    print(f"   Result shape: {c_cpu.shape}")
    print(f"   Result device: {c_cpu.device}")
    print(f"   nvidia-smi after CPU: {run_nvidia_smi()}")
    print()
    
    # Test 2: Try GPU computation
    print("🧪 TEST 2: GPU Computation (Attempted)")
    print("-" * 35)
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available - GPU test impossible")
        return
    
    device_gpu = torch.device('cuda')
    
    # Pre-GPU nvidia-smi
    pre_gpu = run_nvidia_smi()
    print(f"   Pre-GPU nvidia-smi: {pre_gpu}")
    
    start_time = time.time()
    a_gpu = torch.randn(2000, 2000, device=device_gpu)
    b_gpu = torch.randn(2000, 2000, device=device_gpu) 
    
    # Check memory allocation immediately
    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"   GPU Memory Allocated: {gpu_mem_allocated:.3f} GB")
    
    # During computation nvidia-smi
    during_gpu = run_nvidia_smi()
    print(f"   During-allocation nvidia-smi: {during_gpu}")
    
    # Heavy computation
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for completion
    gpu_time = time.time() - start_time
    
    # Post computation
    post_gpu = run_nvidia_smi()
    print(f"   GPU Computation Time: {gpu_time:.3f}s")
    print(f"   Result shape: {c_gpu.shape}")
    print(f"   Result device: {c_gpu.device}")
    print(f"   Post-GPU nvidia-smi: {post_gpu}")
    print()
    
    # Test 3: Speed comparison
    print("🧪 TEST 3: Performance Comparison")
    print("-" * 30)
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"   CPU Time: {cpu_time:.3f}s")
    print(f"   GPU Time: {gpu_time:.3f}s") 
    print(f"   Speedup: {speedup:.1f}x")
    
    if speedup > 2:
        print("   ✅ SIGNIFICANT SPEEDUP - GPU likely working")
    elif speedup > 1.1:
        print("   ⚠️ MODEST SPEEDUP - GPU may be working")
    else:
        print("   ❌ NO SPEEDUP - GPU not working or fallback to CPU")
    print()
    
    # Test 4: Memory verification
    print("🧪 TEST 4: Memory Verification")
    print("-" * 25)
    final_gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"   Final GPU Memory: {final_gpu_mem:.3f} GB")
    
    if final_gpu_mem > 0.01:  # More than 10MB
        print("   ✅ MEMORY ALLOCATED - GPU operations confirmed")
    else:
        print("   ❌ NO MEMORY ALLOCATED - GPU not being used")
    
    print()
    print("🔬 FINAL VERDICT:")
    print("=" * 20)
    
    # Parse nvidia-smi data for analysis
    try:
        pre_parts = pre_gpu.split(', ')
        post_parts = post_gpu.split(', ')
        
        if len(pre_parts) >= 3 and len(post_parts) >= 3:
            pre_mem = int(pre_parts[0])
            post_mem = int(post_parts[0])
            mem_increase = post_mem - pre_mem
            
            print(f"   Memory Change: {pre_mem}MB → {post_mem}MB (+{mem_increase}MB)")
            
            if mem_increase > 100:
                print("   ✅ REAL GPU MEMORY INCREASE")
            else:
                print("   ❌ NO SIGNIFICANT MEMORY INCREASE")
                
        if speedup > 1.5 and final_gpu_mem > 0.01:
            print("   🎉 CONFIRMED: GPU is working correctly")
        elif final_gpu_mem > 0.01:
            print("   ⚠️ PARTIAL: GPU memory used but performance unclear")
        else:
            print("   ❌ FAILED: GPU not being used effectively")
            
    except Exception as e:
        print(f"   Analysis error: {e}")

if __name__ == "__main__":
    test_gpu_vs_cpu() 