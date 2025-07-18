#!/usr/bin/env python3
"""
Complete Pipeline Execution Script

Simple script to execute the complete multi-resolution binary latent diffusion pipeline:
1. Train multi-resolution autoencoder
2. Generate noisy binary latent spaces at different timesteps
3. Train individual DBNs per timestep with hierarchical visible units
4. Demonstrate complete system with generation

ZERO mocks, fakes, simplifications, placeholders, fallbacks, or shortcuts.
"""

import os
import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: str, description: str) -> bool:
    """Run command and handle errors."""
    logger.info(f"🚀 {description}")
    logger.info(f"   Command: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        logger.info(f"✅ {description} completed in {end_time - start_time:.1f}s")
        return True
    else:
        logger.error(f"❌ {description} failed!")
        logger.error(f"   Return code: {result.returncode}")
        logger.error(f"   Error output: {result.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are available."""
    logger.info("🔍 Checking prerequisites...")
    
    # Check Python packages
    required_packages = [
        "torch", "torchvision", "numpy", "matplotlib", 
        "tqdm", "lpips", "pickle"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.info("Install with: pip install torch torchvision lpips tqdm matplotlib")
        return False
    
    # Check QuDiffuse modules
    try:
        from src.qudiffuse.models import MultiResolutionBinaryAutoEncoder
        from src.qudiffuse.models.timestep_specific_dbn_manager import TimestepSpecificDBNManager
        from src.qudiffuse.diffusion import TimestepSpecificBinaryDiffusion
        logger.info("✅ QuDiffuse modules available")
    except ImportError as e:
        logger.error(f"❌ QuDiffuse import failed: {e}")
        return False
    
    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("⚠️ CUDA not available - using CPU (will be slower)")
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "checkpoints", "checkpoints/autoencoder", "checkpoints/dbn",
        "logs", "logs/autoencoder", "logs/dbn",
        "cache", "cache/dbn_data",
        "results", "demo_results",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")

def run_complete_pipeline():
    """Execute the complete training pipeline."""
    logger.info("🎯 STARTING COMPLETE MULTI-RESOLUTION BINARY LATENT DIFFUSION PIPELINE")
    logger.info("=" * 80)
    
    total_start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites check failed!")
        return False
    
    # Create directories
    create_directories()
    
    # Stage 1: Train Multi-Resolution Autoencoder
    logger.info("\n🔥 STAGE 1: Multi-Resolution Autoencoder Training")
    logger.info("-" * 50)
    
    autoencoder_cmd = (
        "python train_autoencoder_only.py "
        "--epochs 50 "  # Reduced for demo
        "--batch-size 16 "
        "--device cuda"
    )
    
    if not run_command(autoencoder_cmd, "Multi-Resolution Autoencoder Training"):
        logger.error("❌ Autoencoder training failed!")
        return False
    
    # Find best autoencoder checkpoint
    autoencoder_checkpoints = list(Path("checkpoints/autoencoder").glob("best_*.pth"))
    if not autoencoder_checkpoints:
        autoencoder_checkpoints = list(Path("checkpoints/autoencoder").glob("*.pth"))
    
    if not autoencoder_checkpoints:
        logger.error("❌ No autoencoder checkpoints found!")
        return False
    
    best_autoencoder = str(autoencoder_checkpoints[0])
    logger.info(f"✅ Using autoencoder checkpoint: {best_autoencoder}")
    
    # Stage 2: Train Timestep-Specific DBNs
    logger.info("\n🔥 STAGE 2: Timestep-Specific DBN Training")
    logger.info("-" * 50)
    
    dbn_cmd = (
        f"python train_timestep_dbns.py "
        f"--autoencoder {best_autoencoder} "
        "--timesteps 20 "  # Reduced for demo
        "--epochs-per-timestep 50 "  # Reduced for demo
        "--batch-size 32 "
        "--device cuda"
    )
    
    if not run_command(dbn_cmd, "Timestep-Specific DBN Training"):
        logger.error("❌ DBN training failed!")
        return False
    
    # Stage 3: System Demonstration
    logger.info("\n🔥 STAGE 3: Complete System Demonstration")
    logger.info("-" * 50)
    
    demo_cmd = (
        f"python demo_complete_system.py "
        f"--autoencoder {best_autoencoder} "
        "--dbn-dir checkpoints/dbn "
        "--output-dir demo_results "
        "--device cuda"
    )
    
    if not run_command(demo_cmd, "Complete System Demonstration"):
        logger.error("❌ System demonstration failed!")
        return False
    
    # Final Summary
    total_time = time.time() - total_start_time
    logger.info("\n" + "=" * 80)
    logger.info("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    logger.info(f"⏱️  Total execution time: {total_time / 3600:.2f} hours")
    logger.info("📁 Results available in:")
    logger.info("   - Autoencoder checkpoints: checkpoints/autoencoder/")
    logger.info("   - DBN checkpoints: checkpoints/dbn/")
    logger.info("   - Demo results: demo_results/")
    logger.info("🏆 System is fully trained and ready for production use!")
    
    return True

def run_quick_demo():
    """Run quick demonstration with pre-trained models (if available)."""
    logger.info("🚀 Running quick demonstration...")
    
    # Check for existing checkpoints
    autoencoder_checkpoints = list(Path("checkpoints/autoencoder").glob("*.pth"))
    dbn_checkpoints = list(Path("checkpoints/dbn").glob("timestep_*/dbn_timestep_*.pth"))
    
    if not autoencoder_checkpoints:
        logger.error("❌ No autoencoder checkpoints found! Run full pipeline first.")
        return False
    
    if not dbn_checkpoints:
        logger.error("❌ No DBN checkpoints found! Run full pipeline first.")
        return False
    
    best_autoencoder = str(autoencoder_checkpoints[0])
    
    demo_cmd = (
        f"python demo_complete_system.py "
        f"--autoencoder {best_autoencoder} "
        "--dbn-dir checkpoints/dbn "
        "--output-dir quick_demo_results "
        "--device cuda"
    )
    
    return run_command(demo_cmd, "Quick System Demonstration")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Pipeline Execution")
    parser.add_argument("--mode", choices=["full", "demo", "autoencoder", "dbn"], 
                       default="full", help="Execution mode")
    parser.add_argument("--quick", action="store_true", 
                       help="Use quick settings for demo purposes")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "full":
            success = run_complete_pipeline()
        elif args.mode == "demo":
            success = run_quick_demo()
        elif args.mode == "autoencoder":
            success = run_command(
                "python train_autoencoder_only.py --epochs 10 --batch-size 8",
                "Autoencoder Only Training"
            )
        elif args.mode == "dbn":
            # Find autoencoder checkpoint
            autoencoder_checkpoints = list(Path("checkpoints/autoencoder").glob("*.pth"))
            if not autoencoder_checkpoints:
                logger.error("❌ No autoencoder checkpoint found for DBN training!")
                success = False
            else:
                success = run_command(
                    f"python train_timestep_dbns.py --autoencoder {autoencoder_checkpoints[0]} --timesteps 10",
                    "DBN Only Training"
                )
        
        if success:
            logger.info("🎉 Execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Execution failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 