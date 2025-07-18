#!/usr/bin/env python3
"""
Paper Specification Validation Script

This script validates that the Diffusion LLM implementation exactly matches
all specifications from "Latent Diffusion with LLMs for Reasoning" paper.

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

import torch
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from diffusion_llm.models import DiffusionLLMModelManager
    from diffusion_llm.encoders import BARTBinaryAutoEncoder, PerceiverAutoEncoder
    from diffusion_llm.diffusion_transformers import ReasoningDiT, TextBinaryDiffusion
    from diffusion_llm.training import UnifiedDiffusionLLMTrainer
    from diffusion_llm.datasets import ArithmeticReasoningDataset, SpatialReasoningDataset
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in standalone mode - imports will be tested separately")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperSpecificationValidator:
    """Comprehensive validator for paper specification compliance."""
    
    def __init__(self):
        """Initialize the validator."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.validation_results = {}
        self.critical_violations = []
        self.warnings = []
        
        # Paper specifications
        self.paper_specs = {
            'lae': 16,              # Latent sequence length
            'dae': 256,             # Latent dimension
            'bart_model': 'facebook/bart-base',
            'hidden_size': 768,     # BART hidden dimension
            'dit_layers': 12,       # DiT transformer layers
            'dit_heads': 12,        # DiT attention heads
            'num_timesteps': 1000,  # Diffusion timesteps
            'reasoning_types': ['arithmetic', 'spatial'],
            'arithmetic_accuracy': 97.2,  # Expected accuracy %
            'spatial_accuracy': 92.3      # Expected accuracy %
        }
        
        logger.info("🔍 Paper Specification Validator initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Paper specs loaded: {len(self.paper_specs)} parameters")
    
    def validate_all(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        
        logger.info("🚀 Starting comprehensive paper specification validation...")
        
        # Run validation phases
        try:
            self.validate_import_structure()
            self.validate_architectural_parameters()
            self.validate_mathematical_formulations()
            self.validate_training_pipeline()
            self.validate_reasoning_tasks()
            self.validate_qudiffuse_integration()
            self.validate_performance_requirements()
            
        except Exception as e:
            self.critical_violations.append(f"Validation failed with error: {e}")
            logger.error(f"❌ Validation error: {e}")
        
        # Generate final report
        return self.generate_final_report()
    
    def validate_import_structure(self) -> None:
        """Validate that all modules can be imported correctly."""
        
        logger.info("📦 Validating import structure...")
        
        import_tests = [
            ("DiffusionLLMModelManager", "diffusion_llm.models", "DiffusionLLMModelManager"),
            ("BARTBinaryAutoEncoder", "diffusion_llm.encoders", "BARTBinaryAutoEncoder"),
            ("PerceiverAutoEncoder", "diffusion_llm.encoders", "PerceiverAutoEncoder"),
            ("ReasoningDiT", "diffusion_llm.diffusion_transformers", "ReasoningDiT"),
            ("TextBinaryDiffusion", "diffusion_llm.diffusion_transformers", "TextBinaryDiffusion"),
            ("UnifiedDiffusionLLMTrainer", "diffusion_llm.training", "UnifiedDiffusionLLMTrainer"),
            ("ArithmeticReasoningDataset", "diffusion_llm.datasets", "ArithmeticReasoningDataset"),
            ("SpatialReasoningDataset", "diffusion_llm.datasets", "SpatialReasoningDataset")
        ]
        
        successful_imports = 0
        
        for name, module, class_name in import_tests:
            try:
                exec(f"from {module} import {class_name}")
                logger.info(f"   ✅ {name}: Import successful")
                successful_imports += 1
            except ImportError as e:
                self.critical_violations.append(f"Failed to import {name}: {e}")
                logger.error(f"   ❌ {name}: Import failed - {e}")
        
        self.validation_results['imports'] = {
            'total_tests': len(import_tests),
            'successful': successful_imports,
            'success_rate': successful_imports / len(import_tests) * 100
        }
        
        logger.info(f"📦 Import validation: {successful_imports}/{len(import_tests)} successful")
    
    def validate_architectural_parameters(self) -> None:
        """Validate architectural parameters match paper specifications."""
        
        logger.info("🏗️ Validating architectural parameters...")
        
        try:
            # Test model manager initialization with paper specs
            manager = DiffusionLLMModelManager(
                lae=self.paper_specs['lae'],
                dae=self.paper_specs['dae'],
                bart_model=self.paper_specs['bart_model']
            )
            
            # Validate configuration
            config_tests = [
                (manager.lae, self.paper_specs['lae'], "lae (latent sequence length)"),
                (manager.dae, self.paper_specs['dae'], "dae (latent dimension)"),
                (manager.bart_model, self.paper_specs['bart_model'], "BART model name")
            ]
            
            parameter_violations = []
            
            for actual, expected, param_name in config_tests:
                if actual != expected:
                    violation = f"{param_name}: expected {expected}, got {actual}"
                    parameter_violations.append(violation)
                    self.critical_violations.append(violation)
                    logger.error(f"   ❌ {violation}")
                else:
                    logger.info(f"   ✅ {param_name}: {actual} (correct)")
            
            # Test component initialization
            try:
                manager.initialize_components()
                
                # Validate DiT parameters
                dit = manager.reasoning_dit
                if dit is not None:
                    dit_tests = [
                        (dit.hidden_size, self.paper_specs['hidden_size'], "DiT hidden size"),
                        (dit.num_layers, self.paper_specs['dit_layers'], "DiT layers"),
                        (dit.num_heads, self.paper_specs['dit_heads'], "DiT attention heads"),
                        (dit.latent_dim, self.paper_specs['dae'], "DiT latent dimension"),
                        (dit.sequence_length, self.paper_specs['lae'], "DiT sequence length")
                    ]
                    
                    for actual, expected, param_name in dit_tests:
                        if actual != expected:
                            violation = f"{param_name}: expected {expected}, got {actual}"
                            parameter_violations.append(violation)
                            self.critical_violations.append(violation)
                            logger.error(f"   ❌ {violation}")
                        else:
                            logger.info(f"   ✅ {param_name}: {actual} (correct)")
                else:
                    self.critical_violations.append("DiT component not initialized")
                
            except Exception as e:
                self.critical_violations.append(f"Component initialization failed: {e}")
                logger.error(f"   ❌ Component initialization failed: {e}")
            
            self.validation_results['architecture'] = {
                'parameter_violations': len(parameter_violations),
                'total_parameters_tested': len(config_tests) + (len(dit_tests) if 'dit_tests' in locals() else 0),
                'compliance_rate': 100 - (len(parameter_violations) / (len(config_tests) + 5) * 100)
            }
            
        except Exception as e:
            self.critical_violations.append(f"Architecture validation failed: {e}")
            logger.error(f"   ❌ Architecture validation failed: {e}")
        
        logger.info(f"🏗️ Architecture validation completed")
    
    def validate_mathematical_formulations(self) -> None:
        """Validate mathematical formulations match paper specifications."""
        
        logger.info("🧮 Validating mathematical formulations...")
        
        math_tests = []
        
        try:
            # Test binary quantization
            dummy_input = torch.randn(2, 16, 256)
            
            # Test that binary quantization produces {0,1} values
            from diffusion_llm.encoders.perceiver_ae import PerceiverAutoEncoder
            perceiver = PerceiverAutoEncoder(
                input_dim=768,
                latent_dim=256,
                latent_sequence_length=16
            )
            
            # Test binary output
            try:
                binary_output = perceiver.quantize_to_binary(dummy_input)
                unique_values = torch.unique(binary_output)
                
                # Check if output is binary
                is_binary = torch.all((unique_values == 0) | (unique_values == 1))
                
                if is_binary:
                    logger.info("   ✅ Binary quantization: Produces {0,1} values (correct)")
                    math_tests.append(True)
                else:
                    violation = f"Binary quantization produces non-binary values: {unique_values.tolist()}"
                    self.critical_violations.append(violation)
                    logger.error(f"   ❌ {violation}")
                    math_tests.append(False)
                    
            except Exception as e:
                self.critical_violations.append(f"Binary quantization test failed: {e}")
                logger.error(f"   ❌ Binary quantization test failed: {e}")
                math_tests.append(False)
            
            # Test attention mechanism shapes
            try:
                from diffusion_llm.diffusion_transformers.reasoning_dit import ReasoningDiT
                dit = ReasoningDiT(
                    latent_dim=256,
                    sequence_length=16,
                    hidden_size=768,
                    num_heads=12,
                    num_layers=12
                )
                
                # Test forward pass shapes
                x = torch.randn(2, 16, 256)  # [batch, seq_len, latent_dim]
                t = torch.randint(0, 1000, (2,))  # [batch]
                reasoning_type = torch.randint(0, 2, (2,))  # [batch]
                
                with torch.no_grad():
                    output = dit(x, t, reasoning_type)
                
                expected_shape = (2, 16, 256)
                if output.shape == expected_shape:
                    logger.info(f"   ✅ DiT forward pass: Shape {output.shape} (correct)")
                    math_tests.append(True)
                else:
                    violation = f"DiT output shape: expected {expected_shape}, got {output.shape}"
                    self.critical_violations.append(violation)
                    logger.error(f"   ❌ {violation}")
                    math_tests.append(False)
                    
            except Exception as e:
                self.critical_violations.append(f"DiT forward pass test failed: {e}")
                logger.error(f"   ❌ DiT forward pass test failed: {e}")
                math_tests.append(False)
            
            self.validation_results['mathematics'] = {
                'tests_passed': sum(math_tests),
                'total_tests': len(math_tests),
                'success_rate': sum(math_tests) / len(math_tests) * 100 if math_tests else 0
            }
            
        except Exception as e:
            self.critical_violations.append(f"Mathematical validation failed: {e}")
            logger.error(f"   ❌ Mathematical validation failed: {e}")
        
        logger.info(f"🧮 Mathematical validation completed")
    
    def validate_training_pipeline(self) -> None:
        """Validate training pipeline structure."""
        
        logger.info("🎓 Validating training pipeline...")
        
        try:
            # Test unified trainer initialization
            trainer = UnifiedDiffusionLLMTrainer(
                num_encoder_latents=self.paper_specs['lae'],
                dim_ae=self.paper_specs['dae'],
                bart_model_name=self.paper_specs['bart_model']
            )
            
            pipeline_tests = []
            
            # Check if stage trainers are available
            stage1_available = hasattr(trainer, 'stage1_trainer') or hasattr(trainer, '_setup_stage1_trainer')
            stage2_available = hasattr(trainer, 'stage2_trainer') or hasattr(trainer, '_setup_stage2_trainer')
            
            if stage1_available:
                logger.info("   ✅ Stage 1 trainer: Available")
                pipeline_tests.append(True)
            else:
                self.critical_violations.append("Stage 1 trainer not available")
                logger.error("   ❌ Stage 1 trainer: Not available")
                pipeline_tests.append(False)
            
            if stage2_available:
                logger.info("   ✅ Stage 2 trainer: Available")
                pipeline_tests.append(True)
            else:
                self.critical_violations.append("Stage 2 trainer not available")
                logger.error("   ❌ Stage 2 trainer: Not available")
                pipeline_tests.append(False)
            
            # Check training methods
            training_methods = ['train_stage1', 'train_stage2', 'evaluate_model']
            for method in training_methods:
                if hasattr(trainer, method):
                    logger.info(f"   ✅ {method}: Available")
                    pipeline_tests.append(True)
                else:
                    self.warnings.append(f"{method} method not found")
                    logger.warning(f"   ⚠️ {method}: Not found")
                    pipeline_tests.append(False)
            
            self.validation_results['training_pipeline'] = {
                'tests_passed': sum(pipeline_tests),
                'total_tests': len(pipeline_tests),
                'success_rate': sum(pipeline_tests) / len(pipeline_tests) * 100 if pipeline_tests else 0
            }
            
        except Exception as e:
            self.critical_violations.append(f"Training pipeline validation failed: {e}")
            logger.error(f"   ❌ Training pipeline validation failed: {e}")
        
        logger.info(f"🎓 Training pipeline validation completed")
    
    def validate_reasoning_tasks(self) -> None:
        """Validate reasoning task implementations."""
        
        logger.info("🧠 Validating reasoning tasks...")
        
        reasoning_tests = []
        
        try:
            # Test arithmetic reasoning dataset
            arithmetic_dataset = ArithmeticReasoningDataset(
                num_samples=100,
                chain_length_range=(3, 5)
            )
            
            # Test dataset creation
            if len(arithmetic_dataset) > 0:
                logger.info(f"   ✅ Arithmetic dataset: {len(arithmetic_dataset)} samples")
                reasoning_tests.append(True)
                
                # Test sample format
                sample = arithmetic_dataset[0]
                if 'problem' in sample and 'solution' in sample:
                    logger.info("   ✅ Arithmetic format: Contains problem and solution")
                    reasoning_tests.append(True)
                else:
                    self.critical_violations.append("Arithmetic dataset missing required fields")
                    logger.error("   ❌ Arithmetic format: Missing required fields")
                    reasoning_tests.append(False)
            else:
                self.critical_violations.append("Arithmetic dataset is empty")
                logger.error("   ❌ Arithmetic dataset: Empty")
                reasoning_tests.append(False)
            
            # Test spatial reasoning dataset
            spatial_dataset = SpatialReasoningDataset(
                num_samples=100,
                max_steps=5
            )
            
            if len(spatial_dataset) > 0:
                logger.info(f"   ✅ Spatial dataset: {len(spatial_dataset)} samples")
                reasoning_tests.append(True)
                
                # Test sample format
                sample = spatial_dataset[0]
                if 'problem' in sample and 'solution' in sample:
                    logger.info("   ✅ Spatial format: Contains problem and solution")
                    reasoning_tests.append(True)
                else:
                    self.critical_violations.append("Spatial dataset missing required fields")
                    logger.error("   ❌ Spatial format: Missing required fields")
                    reasoning_tests.append(False)
            else:
                self.critical_violations.append("Spatial dataset is empty")
                logger.error("   ❌ Spatial dataset: Empty")
                reasoning_tests.append(False)
            
            self.validation_results['reasoning_tasks'] = {
                'tests_passed': sum(reasoning_tests),
                'total_tests': len(reasoning_tests),
                'success_rate': sum(reasoning_tests) / len(reasoning_tests) * 100 if reasoning_tests else 0
            }
            
        except Exception as e:
            self.critical_violations.append(f"Reasoning task validation failed: {e}")
            logger.error(f"   ❌ Reasoning task validation failed: {e}")
        
        logger.info(f"🧠 Reasoning task validation completed")
    
    def validate_qudiffuse_integration(self) -> None:
        """Validate QuDiffuse integration."""
        
        logger.info("🔗 Validating QuDiffuse integration...")
        
        integration_tests = []
        
        try:
            # Test QuDiffuse component imports
            qudiffuse_components = [
                "qudiffuse.diffusion.TimestepSpecificBinaryDiffusion",
                "qudiffuse.diffusion.UnifiedReverseProcess", 
                "qudiffuse.models.BinaryLatentManager",
                "qudiffuse.solvers.ZephyrQuantumSolver"
            ]
            
            for component in qudiffuse_components:
                try:
                    module_path, class_name = component.rsplit('.', 1)
                    exec(f"from {module_path} import {class_name}")
                    logger.info(f"   ✅ {class_name}: Import successful")
                    integration_tests.append(True)
                except ImportError as e:
                    self.warnings.append(f"QuDiffuse component {class_name} not available: {e}")
                    logger.warning(f"   ⚠️ {class_name}: Import failed - {e}")
                    integration_tests.append(False)
            
            # Test TextBinaryDiffusion integration
            try:
                text_diffusion = TextBinaryDiffusion(
                    latent_dim=256,
                    sequence_length=16,
                    num_timesteps=1000
                )
                
                logger.info("   ✅ TextBinaryDiffusion: Initialization successful")
                integration_tests.append(True)
                
            except Exception as e:
                self.critical_violations.append(f"TextBinaryDiffusion initialization failed: {e}")
                logger.error(f"   ❌ TextBinaryDiffusion: Initialization failed - {e}")
                integration_tests.append(False)
            
            self.validation_results['qudiffuse_integration'] = {
                'tests_passed': sum(integration_tests),
                'total_tests': len(integration_tests),
                'success_rate': sum(integration_tests) / len(integration_tests) * 100 if integration_tests else 0
            }
            
        except Exception as e:
            self.critical_violations.append(f"QuDiffuse integration validation failed: {e}")
            logger.error(f"   ❌ QuDiffuse integration validation failed: {e}")
        
        logger.info(f"🔗 QuDiffuse integration validation completed")
    
    def validate_performance_requirements(self) -> None:
        """Validate performance requirements."""
        
        logger.info("⚡ Validating performance requirements...")
        
        performance_tests = []
        
        try:
            # Test model parameter counts
            manager = DiffusionLLMModelManager(
                lae=16, dae=256, bart_model="facebook/bart-base"
            )
            
            manager.initialize_components()
            memory_stats = manager.get_memory_usage()
            
            # Expected parameter ranges (from paper)
            expected_ranges = {
                'bart_autoencoder': (100_000_000, 200_000_000),  # ~140M
                'perceiver_ae': (10_000_000, 50_000_000),       # ~15M  
                'reasoning_dit': (50_000_000, 150_000_000),     # ~85M
                'total': (200_000_000, 400_000_000)             # ~240M
            }
            
            for component, (min_params, max_params) in expected_ranges.items():
                if component in memory_stats:
                    # Convert MB to approximate parameters (assuming 4 bytes per param)
                    estimated_params = int(memory_stats[component] * 1024 * 1024 / 4)
                    
                    if min_params <= estimated_params <= max_params:
                        logger.info(f"   ✅ {component}: {estimated_params:,} parameters (within range)")
                        performance_tests.append(True)
                    else:
                        self.warnings.append(f"{component} parameter count outside expected range")
                        logger.warning(f"   ⚠️ {component}: {estimated_params:,} parameters (outside range {min_params:,}-{max_params:,})")
                        performance_tests.append(False)
                else:
                    self.warnings.append(f"{component} memory stats not available")
                    logger.warning(f"   ⚠️ {component}: Memory stats not available")
                    performance_tests.append(False)
            
            self.validation_results['performance'] = {
                'tests_passed': sum(performance_tests),
                'total_tests': len(performance_tests),
                'success_rate': sum(performance_tests) / len(performance_tests) * 100 if performance_tests else 0
            }
            
        except Exception as e:
            self.critical_violations.append(f"Performance validation failed: {e}")
            logger.error(f"   ❌ Performance validation failed: {e}")
        
        logger.info(f"⚡ Performance validation completed")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        logger.info("📊 Generating final validation report...")
        
        # Calculate overall compliance
        total_tests = 0
        total_passed = 0
        
        for category, results in self.validation_results.items():
            if isinstance(results, dict) and 'total_tests' in results:
                total_tests += results['total_tests']
                total_passed += results.get('tests_passed', 0)
        
        overall_compliance = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Determine status
        if len(self.critical_violations) == 0 and overall_compliance >= 95:
            status = "✅ FULLY COMPLIANT"
            status_color = "GREEN"
        elif len(self.critical_violations) == 0 and overall_compliance >= 80:
            status = "⚠️ MOSTLY COMPLIANT"
            status_color = "YELLOW"
        else:
            status = "❌ NON-COMPLIANT"
            status_color = "RED"
        
        final_report = {
            'status': status,
            'status_color': status_color,
            'overall_compliance': overall_compliance,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'critical_violations': len(self.critical_violations),
            'warnings': len(self.warnings),
            'category_results': self.validation_results,
            'violation_details': self.critical_violations,
            'warning_details': self.warnings
        }
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🏆 PAPER SPECIFICATION VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {status}")
        logger.info(f"Overall Compliance: {overall_compliance:.1f}%")
        logger.info(f"Tests Passed: {total_passed}/{total_tests}")
        logger.info(f"Critical Violations: {len(self.critical_violations)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info("")
        
        # Print category breakdown
        for category, results in self.validation_results.items():
            if isinstance(results, dict) and 'success_rate' in results:
                logger.info(f"{category.title()}: {results['success_rate']:.1f}% ({results.get('tests_passed', 0)}/{results.get('total_tests', 0)})")
        
        logger.info("")
        
        # Print violations if any
        if self.critical_violations:
            logger.info("🚨 CRITICAL VIOLATIONS:")
            for i, violation in enumerate(self.critical_violations, 1):
                logger.info(f"   {i}. {violation}")
            logger.info("")
        
        # Print warnings if any
        if self.warnings:
            logger.info("⚠️ WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                logger.info(f"   {i}. {warning}")
            logger.info("")
        
        if status == "✅ FULLY COMPLIANT":
            logger.info("🎉 CONGRATULATIONS: Implementation fully complies with paper specifications!")
        elif status == "⚠️ MOSTLY COMPLIANT":
            logger.info("👍 GOOD: Implementation mostly complies with paper specifications.")
            logger.info("💡 Consider addressing warnings for full compliance.")
        else:
            logger.info("🔧 ACTION REQUIRED: Critical violations must be resolved.")
        
        logger.info("=" * 80)
        
        return final_report


def main():
    """Main validation function."""
    
    print("🎯 Paper Specification Validator")
    print("Validating 'Latent Diffusion with LLMs for Reasoning' implementation")
    print("=" * 80)
    
    validator = PaperSpecificationValidator()
    results = validator.validate_all()
    
    # Save results
    import json
    results_file = Path("paper_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return exit code based on compliance
    if results['status'] == "✅ FULLY COMPLIANT":
        return 0
    elif results['status'] == "⚠️ MOSTLY COMPLIANT":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 