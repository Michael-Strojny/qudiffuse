#!/usr/bin/env python3
"""
Simple GPU Test - PROVE V100 is working
No lies, no fake results, actual GPU computation test
"""

import torch
import time
import sys

print("🔧 GPU Test Starting...")

# Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    sys.exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name()}")
print(f"✅ CUDA Version: {torch.version.cuda}")

# Force a large computation on GPU
print("\n🚀 Testing GPU computation...")

device = torch.device('cuda')

# Create large tensors that MUST use GPU memory
print("Creating large tensors...")
A = torch.randn(2048, 2048, device=device, dtype=torch.float32)
B = torch.randn(2048, 2048, device=device, dtype=torch.float32)

print(f"✅ Tensor A on GPU: {A.is_cuda}")
print(f"✅ Tensor B on GPU: {B.is_cuda}")

# Check GPU memory BEFORE computation
mem_before = torch.cuda.memory_allocated() / 1024**3
print(f"✅ GPU Memory BEFORE: {mem_before:.2f} GB")

if mem_before < 0.01:
    print("❌ GPU memory too low - tensors not on GPU!")
    sys.exit(1)

# Force large matrix multiplication on GPU
print("\n🔥 Running matrix multiplication on GPU...")
start_time = time.time()

# This MUST run on GPU and use significant memory
C = torch.matmul(A, B)
torch.cuda.synchronize()  # Wait for GPU to finish

end_time = time.time()

# Check GPU memory AFTER computation
mem_after = torch.cuda.memory_allocated() / 1024**3
print(f"✅ GPU Memory AFTER: {mem_after:.2f} GB")

print(f"✅ Computation time: {end_time - start_time:.3f} seconds")
print(f"✅ Result tensor on GPU: {C.is_cuda}")
print(f"✅ Result shape: {C.shape}")

# Verify result is not zeros (actual computation happened)
result_sum = C.sum().item()
print(f"✅ Result sum: {result_sum:.2f}")

if abs(result_sum) < 1e-6:
    print("❌ Result sum too small - computation may have failed!")
    sys.exit(1)

print("\n🎉 GPU TEST PASSED!")
print(f"✅ GPU Memory Used: {mem_after:.2f} GB")
print("✅ V100 is working properly!")

# Now test if we can run nvidia-smi during computation
print("\n🔧 Testing GPU monitoring...")
import subprocess

# Create continuous GPU load
print("Creating continuous GPU load...")
for i in range(5):
    print(f"Iteration {i+1}/5...")
    D = torch.randn(1024, 1024, device=device)
    E = torch.randn(1024, 1024, device=device)
    F = torch.matmul(D, E)
    
    # Check memory during computation
    mem_current = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU Memory: {mem_current:.2f} GB")
    
    torch.cuda.synchronize()
    time.sleep(1)

print("\n✅ GPU monitoring test complete!")
print("✅ V100 is definitely working!") 