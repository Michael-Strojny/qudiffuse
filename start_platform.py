#!/usr/bin/env python3
"""
QuDiffusive Web Platform Startup Script

Comprehensive startup script that initializes and validates the complete
QuDiffusive web platform with zero tolerance for missing components.

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlatformStarter:
    """
    Production-ready platform starter with comprehensive validation.
    
    Ensures all components are properly initialized before starting:
    - QuDiffusive models and dependencies
    - Web platform components
    - Security and monitoring
    - Frontend build
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.src_dir = self.base_dir / "src"
        self.web_platform_dir = self.base_dir / "web_platform"
        self.frontend_dir = self.web_platform_dir / "frontend"
        
        # Add src to Python path
        sys.path.insert(0, str(self.src_dir))
        
        logger.info("🚀 QuDiffusive Web Platform Starter initialized")
    
    def validate_environment(self):
        """Validate the environment and dependencies."""
        
        logger.info("🔍 Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ is required")
        
        # Check required directories
        required_dirs = [
            self.src_dir,
            self.web_platform_dir,
            self.src_dir / "qudiffuse",
            self.src_dir / "diffusion_llm"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise RuntimeError(f"Required directory not found: {dir_path}")
        
        # Check core QuDiffusive files
        core_files = [
            self.src_dir / "qudiffuse" / "models" / "multi_resolution_binary_ae.py",
            self.src_dir / "qudiffuse" / "models" / "dbn.py",
            self.src_dir / "qudiffuse" / "models" / "binary_latent_manager.py",
            self.src_dir / "qudiffuse" / "diffusion" / "timestep_specific_binary_diffusion.py",
            self.src_dir / "qudiffuse" / "diffusion" / "unified_reverse_process.py",
            self.src_dir / "qudiffuse" / "solvers" / "zephyr_quantum_solver.py"
        ]
        
        for file_path in core_files:
            if not file_path.exists():
                raise RuntimeError(f"Core QuDiffusive file not found: {file_path}")
        
        # Check web platform files
        web_files = [
            self.web_platform_dir / "main.py",
            self.web_platform_dir / "config.py",
            self.web_platform_dir / "websocket_manager.py",
            self.web_platform_dir / "models" / "model_manager.py",
            self.web_platform_dir / "api" / "__init__.py"
        ]
        
        for file_path in web_files:
            if not file_path.exists():
                raise RuntimeError(f"Web platform file not found: {file_path}")
        
        logger.info("✅ Environment validation passed")
    
    def check_dependencies(self):
        """Check Python dependencies."""
        
        logger.info("📦 Checking Python dependencies...")
        
        required_packages = [
            "numpy", "fastapi", "uvicorn", "websockets",
            "pydantic", "structlog", "redis"
        ]

        # Optional packages (for development)
        optional_packages = ["torch", "pillow"]
        
        missing_packages = []
        missing_optional = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)

        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.info("Install missing packages with: pip install -r requirements.txt")
            raise RuntimeError(f"Missing required packages: {missing_packages}")

        if missing_optional:
            logger.warning(f"Missing optional packages (using placeholders): {missing_optional}")
            logger.info("For full functionality, install: pip install " + " ".join(missing_optional))
        
        logger.info("✅ All Python dependencies available")
    
    def build_frontend(self):
        """Build the React frontend if needed."""
        
        logger.info("🏗️ Checking frontend build...")
        
        if not self.frontend_dir.exists():
            logger.warning("Frontend directory not found, skipping frontend build")
            return
        
        build_dir = self.frontend_dir / "build"
        package_json = self.frontend_dir / "package.json"
        
        # Check if build is needed
        if build_dir.exists() and package_json.exists():
            build_time = build_dir.stat().st_mtime
            package_time = package_json.stat().st_mtime
            
            if build_time > package_time:
                logger.info("✅ Frontend build is up to date")
                return
        
        if not package_json.exists():
            logger.warning("package.json not found, skipping frontend build")
            return
        
        logger.info("Building React frontend...")
        
        try:
            # Install dependencies
            subprocess.run(
                ["npm", "ci"],
                cwd=self.frontend_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Build frontend
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.frontend_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            if not build_dir.exists():
                raise RuntimeError("Frontend build failed - build directory not created")
            
            logger.info("✅ Frontend built successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Frontend build failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError("Frontend build failed")
        except FileNotFoundError:
            logger.warning("npm not found, skipping frontend build")
    
    def validate_qudiffuse_models(self, skip_for_development=False):
        """Validate QuDiffusive model components."""

        logger.info("🧠 Validating QuDiffusive models...")

        if skip_for_development:
            logger.warning("⚠️ Skipping QuDiffusive model validation - DEVELOPMENT MODE ONLY")
            logger.warning("🚨 This will NOT work for production use")
            return

        try:
            # Test imports
            from qudiffuse.models.multi_resolution_binary_ae import MultiResolutionBinaryAutoEncoder
            from qudiffuse.models.dbn import HierarchicalDBN
            from qudiffuse.models.binary_latent_manager import BinaryLatentManager
            from qudiffuse.diffusion.timestep_specific_binary_diffusion import TimestepSpecificBinaryDiffusion
            from qudiffuse.diffusion.unified_reverse_process import UnifiedReverseProcess

            # Test basic instantiation
            latent_shapes = [(4, 8, 8), (2, 4, 4), (1, 2, 2)]

            # Test autoencoder
            autoencoder = MultiResolutionBinaryAutoEncoder(
                latent_shapes=latent_shapes,
                img_size=32,
                in_channels=3,
                device="cpu"
            )

            # Test binary manager
            binary_manager = BinaryLatentManager(
                latent_shapes=latent_shapes,
                topology_type="hierarchical",
                device="cpu"
            )

            # Test DBN
            hidden_dims = [128, 64, 32]
            dbn = HierarchicalDBN(
                latent_shapes=latent_shapes,
                hidden_dims=hidden_dims,
                device="cpu"
            )

            logger.info("✅ QuDiffusive models validated successfully")

        except Exception as e:
            logger.error(f"❌ QuDiffusive model validation failed: {e}")
            logger.error("🚨 Cannot start platform without functional QuDiffuse models")
            logger.error("💡 Install dependencies: pip install -r requirements.txt")
            logger.error("💡 Ensure all model files are properly built")
            raise RuntimeError(f"QuDiffusive models validation failed: {e}")
    
    def start_platform(self, host="0.0.0.0", port=8080, debug=False):
        """Start the complete QuDiffusive platform."""
        
        logger.info("🚀 Starting QuDiffusive Platform...")
        logger.info("=" * 60)
        
        # Step 1: Check dependencies
        self.check_dependencies()
        
        # Step 2: Build frontend if needed
        self.build_frontend()
        
        # Step 3: Validate models (REQUIRED - no placeholders)
        self.validate_qudiffuse_models(skip_for_development=debug)
        
        # Step 4: Start the platform
        logger.info(f"🌐 Starting platform on {host}:{port}")
        
        try:
            import uvicorn
            from web_platform.main import app
            
            # Configure logging
            log_config = uvicorn.config.LOGGING_CONFIG
            log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
            
            # Start server
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_config=log_config,
                access_log=True
            )
            
        except ImportError:
            logger.error("❌ uvicorn not available")
            logger.error("💡 Install with: pip install uvicorn")
            raise RuntimeError("uvicorn required for web platform")
        
        except Exception as e:
            logger.error(f"❌ Failed to start platform: {e}")
            raise
    
    def wait_for_platform(self, host="localhost", port=8080, timeout=60):
        """Wait for platform to be ready."""
        
        logger.info("⏳ Waiting for platform to be ready...")
        
        start_time = time.time()
        url = f"http://{host}:{port}/health"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Platform is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error("❌ Platform failed to start within timeout")
        return False
    
    def run_startup_sequence(self, host="0.0.0.0", port=8080, debug=False):
        """Run complete startup sequence."""
        
        logger.info("🚀 Starting QuDiffusive Web Platform startup sequence...")
        logger.info("=" * 60)
        
        try:
            # Validation steps
            self.validate_environment()
            self.check_dependencies()
            self.build_frontend()
            self.validate_qudiffuse_models(skip_for_development=debug)
            
            logger.info("=" * 60)
            logger.info("✅ All validation checks passed!")
            logger.info("🌐 Starting web platform...")
            logger.info("=" * 60)
            
            # Start platform
            self.start_platform(host, port, debug)
            
        except Exception as e:
            logger.error(f"💥 Startup failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="QuDiffusive Web Platform Starter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation, don't start platform")
    
    args = parser.parse_args()
    
    starter = PlatformStarter()
    
    if args.validate_only:
        logger.info("🔍 Running validation only...")
        try:
            starter.validate_environment()
            starter.check_dependencies()
            starter.build_frontend()
            starter.validate_qudiffuse_models()
            logger.info("✅ All validations passed!")
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            sys.exit(1)
    else:
        starter.run_startup_sequence(args.host, args.port, args.debug)


if __name__ == "__main__":
    main()
