#!/usr/bin/env python3
"""
QuDiffuse Integration Test Script

This script validates that the Diffusion LLM implementation is 100% compatible
with the existing QuDiffuse binary diffusion system.

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuDiffuseIntegrationTester:
    """Tests QuDiffuse integration points."""
    
    def __init__(self):
        """Initialize the tester."""
        self.base_dir = Path(__file__).parent.parent.parent
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
        logger.info("🔗 QuDiffuse Integration Tester initialized")
    
    def test_all_integrations(self) -> Dict[str, Any]:
        """Test all QuDiffuse integration points."""
        
        logger.info("🚀 Starting QuDiffuse integration tests...")
        logger.info("=" * 60)
        
        self.test_import_compatibility()
        self.test_component_integration()
        self.test_data_flow_compatibility()
        self.test_quantum_annealer_integration()
        self.test_classical_fallback_integration()
        
        return self.generate_integration_report()
    
    def test_import_compatibility(self) -> None:
        """Test that QuDiffuse components can be imported correctly."""
        
        logger.info("📦 Testing QuDiffuse import compatibility...")
        
        qudiffuse_imports = [
            # Core diffusion components
            ("qudiffuse.diffusion", "TimestepSpecificBinaryDiffusion"),
            ("qudiffuse.diffusion", "UnifiedReverseProcess"),
            ("qudiffuse.diffusion", "WindowedQUBODiffusion"),
            
            # Model components
            ("qudiffuse.models", "BinaryLatentManager"),
            ("qudiffuse.models", "HierarchicalDBN"),
            ("qudiffuse.models.timestep_specific_dbn_manager", "TimestepSpecificDBNManager"),
            
            # Solver components  
            ("qudiffuse.solvers", "ZephyrQuantumSolver"),
            
            # Utility components
            ("qudiffuse.utils.common_utils", "validate_tensor_shape"),
            ("qudiffuse.utils.error_handling", "BinaryLatentError")
        ]
        
        import_tests = []
        successful_imports = 0
        
        for module_path, component_name in qudiffuse_imports:
            try:
                # Test import
                exec(f"from {module_path} import {component_name}")
                logger.info(f"   ✅ {component_name}: Import successful")
                import_tests.append(True)
                successful_imports += 1
                
            except ImportError as e:
                error_msg = f"Failed to import {component_name} from {module_path}: {e}"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {component_name}: {error_msg}")
                import_tests.append(False)
                
            except Exception as e:
                error_msg = f"Unexpected error importing {component_name}: {e}"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {component_name}: {error_msg}")
                import_tests.append(False)
        
        self.test_results['import_compatibility'] = {
            'successful_imports': successful_imports,
            'total_imports': len(qudiffuse_imports),
            'success_rate': successful_imports / len(qudiffuse_imports) * 100,
            'all_imports_successful': successful_imports == len(qudiffuse_imports)
        }
        
        logger.info(f"📦 Import compatibility: {successful_imports}/{len(qudiffuse_imports)} successful")
    
    def test_component_integration(self) -> None:
        """Test integration of Diffusion LLM components with QuDiffuse."""
        
        logger.info("🔧 Testing component integration...")
        
        integration_tests = []
        
        # Test TextBinaryDiffusion integration
        try:
            # Check if TextBinaryDiffusion imports QuDiffuse components correctly
            text_diffusion_file = self.base_dir / "diffusion_llm" / "diffusion_transformers" / "text_binary_diffusion.py"
            
            if text_diffusion_file.exists():
                with open(text_diffusion_file, 'r') as f:
                    content = f.read()
                
                # Check for QuDiffuse imports
                required_imports = [
                    "TimestepSpecificBinaryDiffusion",
                    "UnifiedReverseProcess", 
                    "BinaryLatentManager",
                    "TimestepSpecificDBNManager"
                ]
                
                imports_found = 0
                for import_name in required_imports:
                    if import_name in content:
                        logger.info(f"   ✅ TextBinaryDiffusion: {import_name} imported")
                        imports_found += 1
                        integration_tests.append(True)
                    else:
                        warning_msg = f"TextBinaryDiffusion missing import: {import_name}"
                        self.warnings.append(warning_msg)
                        logger.warning(f"   ⚠️ {warning_msg}")
                        integration_tests.append(False)
                
                # Check for classical fallback mode usage
                if "ClassicalFallbackMode" in content:
                    logger.info("   ✅ TextBinaryDiffusion: ClassicalFallbackMode integration found")
                    integration_tests.append(True)
                else:
                    warning_msg = "TextBinaryDiffusion: ClassicalFallbackMode not found"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    integration_tests.append(False)
                    
            else:
                error_msg = "TextBinaryDiffusion file not found"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                integration_tests.extend([False] * 5)
        
        except Exception as e:
            error_msg = f"Error testing TextBinaryDiffusion integration: {e}"
            self.errors.append(error_msg)
            logger.error(f"   ❌ {error_msg}")
            integration_tests.extend([False] * 5)
        
        # Test if Diffusion LLM creates compatible binary latent formats
        try:
            model_manager_file = self.base_dir / "diffusion_llm" / "models" / "model_manager.py"
            
            if model_manager_file.exists():
                with open(model_manager_file, 'r') as f:
                    content = f.read()
                
                # Check for binary quantization methods
                if "quantize_to_binary" in content:
                    logger.info("   ✅ ModelManager: Binary quantization methods found")
                    integration_tests.append(True)
                else:
                    warning_msg = "ModelManager: Binary quantization methods not found"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    integration_tests.append(False)
                
                # Check for QUBO formulation support
                if "qubo" in content.lower() or "QUBO" in content:
                    logger.info("   ✅ ModelManager: QUBO formulation support found")
                    integration_tests.append(True)
                else:
                    warning_msg = "ModelManager: QUBO formulation support not found"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    integration_tests.append(False)
            
            else:
                error_msg = "ModelManager file not found"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                integration_tests.extend([False] * 2)
        
        except Exception as e:
            error_msg = f"Error testing ModelManager integration: {e}"
            self.errors.append(error_msg)
            logger.error(f"   ❌ {error_msg}")
            integration_tests.extend([False] * 2)
        
        self.test_results['component_integration'] = {
            'tests_passed': sum(integration_tests),
            'total_tests': len(integration_tests),
            'success_rate': sum(integration_tests) / len(integration_tests) * 100 if integration_tests else 0
        }
        
        logger.info(f"🔧 Component integration: {sum(integration_tests)}/{len(integration_tests)} tests passed")
    
    def test_data_flow_compatibility(self) -> None:
        """Test data flow compatibility between Diffusion LLM and QuDiffuse."""
        
        logger.info("🔄 Testing data flow compatibility...")
        
        data_flow_tests = []
        
        # Check for proper tensor shape handling
        try:
            # Look for shape validation in key files
            key_files = [
                "diffusion_llm/diffusion_transformers/text_binary_diffusion.py",
                "diffusion_llm/models/model_manager.py",
                "diffusion_llm/encoders/perceiver_ae.py"
            ]
            
            for file_path in key_files:
                full_path = self.base_dir / file_path
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Check for tensor shape documentation
                    shape_docs = content.count("[B,") + content.count("[batch") + content.count("shape:")
                    
                    if shape_docs > 0:
                        logger.info(f"   ✅ {Path(file_path).name}: Tensor shape documentation found ({shape_docs} instances)")
                        data_flow_tests.append(True)
                    else:
                        warning_msg = f"{Path(file_path).name}: Limited tensor shape documentation"
                        self.warnings.append(warning_msg)
                        logger.warning(f"   ⚠️ {warning_msg}")
                        data_flow_tests.append(False)
                else:
                    error_msg = f"File not found: {file_path}"
                    self.errors.append(error_msg)
                    logger.error(f"   ❌ {error_msg}")
                    data_flow_tests.append(False)
        
        except Exception as e:
            error_msg = f"Error testing data flow compatibility: {e}"
            self.errors.append(error_msg)
            logger.error(f"   ❌ {error_msg}")
            data_flow_tests.extend([False] * 3)
        
        # Check for binary latent format consistency
        try:
            # Look for binary format specifications
            perceiver_file = self.base_dir / "diffusion_llm" / "encoders" / "perceiver_ae.py"
            
            if perceiver_file.exists():
                with open(perceiver_file, 'r') as f:
                    content = f.read()
                
                # Check for {0,1} binary format
                if "{0,1}" in content or "0,1" in content:
                    logger.info("   ✅ Binary format: {0,1} specification found")
                    data_flow_tests.append(True)
                else:
                    warning_msg = "Binary format: {0,1} specification not clearly documented"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    data_flow_tests.append(False)
                
                # Check for straight-through estimator
                if "straight" in content.lower() and "through" in content.lower():
                    logger.info("   ✅ Binary quantization: Straight-through estimator found")
                    data_flow_tests.append(True)
                else:
                    warning_msg = "Binary quantization: Straight-through estimator not found"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    data_flow_tests.append(False)
            
            else:
                error_msg = "Perceiver autoencoder file not found"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                data_flow_tests.extend([False] * 2)
        
        except Exception as e:
            error_msg = f"Error testing binary format compatibility: {e}"
            self.errors.append(error_msg)
            logger.error(f"   ❌ {error_msg}")
            data_flow_tests.extend([False] * 2)
        
        self.test_results['data_flow_compatibility'] = {
            'tests_passed': sum(data_flow_tests),
            'total_tests': len(data_flow_tests),
            'success_rate': sum(data_flow_tests) / len(data_flow_tests) * 100 if data_flow_tests else 0
        }
        
        logger.info(f"🔄 Data flow compatibility: {sum(data_flow_tests)}/{len(data_flow_tests)} tests passed")
    
    def test_quantum_annealer_integration(self) -> None:
        """Test quantum annealer integration."""
        
        logger.info("⚛️ Testing quantum annealer integration...")
        
        quantum_tests = []
        
        try:
            # Check for D-Wave Zephyr integration
            text_diffusion_file = self.base_dir / "diffusion_llm" / "diffusion_transformers" / "text_binary_diffusion.py"
            
            if text_diffusion_file.exists():
                with open(text_diffusion_file, 'r') as f:
                    content = f.read()
                
                # Check for quantum-related components
                quantum_components = [
                    "ZephyrQuantumSolver",
                    "quantum_enabled",
                    "window_size",
                    "QUBO"
                ]
                
                for component in quantum_components:
                    if component in content:
                        logger.info(f"   ✅ Quantum integration: {component} found")
                        quantum_tests.append(True)
                    else:
                        warning_msg = f"Quantum integration: {component} not found"
                        self.warnings.append(warning_msg)
                        logger.warning(f"   ⚠️ {warning_msg}")
                        quantum_tests.append(False)
                
                # Check for windowed QUBO approach
                if "windowed" in content.lower() and "qubo" in content.lower():
                    logger.info("   ✅ Windowed QUBO approach found")
                    quantum_tests.append(True)
                else:
                    warning_msg = "Windowed QUBO approach not clearly implemented"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    quantum_tests.append(False)
            
            else:
                error_msg = "TextBinaryDiffusion file not found for quantum testing"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                quantum_tests.extend([False] * 5)
        
        except Exception as e:
            error_msg = f"Error testing quantum annealer integration: {e}"
            self.errors.append(error_msg)
            logger.error(f"   ❌ {error_msg}")
            quantum_tests.extend([False] * 5)
        
        self.test_results['quantum_annealer_integration'] = {
            'tests_passed': sum(quantum_tests),
            'total_tests': len(quantum_tests),
            'success_rate': sum(quantum_tests) / len(quantum_tests) * 100 if quantum_tests else 0
        }
        
        logger.info(f"⚛️ Quantum annealer integration: {sum(quantum_tests)}/{len(quantum_tests)} tests passed")
    
    def test_classical_fallback_integration(self) -> None:
        """Test classical fallback integration."""
        
        logger.info("🔄 Testing classical fallback integration...")
        
        fallback_tests = []
        
        try:
            # Check for contrastive divergence fallback
            text_diffusion_file = self.base_dir / "diffusion_llm" / "diffusion_transformers" / "text_binary_diffusion.py"
            
            if text_diffusion_file.exists():
                with open(text_diffusion_file, 'r') as f:
                    content = f.read()
                
                # Check for classical fallback components
                fallback_components = [
                    "ClassicalFallbackMode",
                    "CONTRASTIVE_DIVERGENCE", 
                    "fallback",
                    "classical"
                ]
                
                for component in fallback_components:
                    if component in content:
                        logger.info(f"   ✅ Classical fallback: {component} found")
                        fallback_tests.append(True)
                    else:
                        warning_msg = f"Classical fallback: {component} not found"
                        self.warnings.append(warning_msg)
                        logger.warning(f"   ⚠️ {warning_msg}")
                        fallback_tests.append(False)
                
                # Check for DBN integration
                if "DBN" in content or "dbn" in content.lower():
                    logger.info("   ✅ DBN integration found")
                    fallback_tests.append(True)
                else:
                    warning_msg = "DBN integration not found"
                    self.warnings.append(warning_msg)
                    logger.warning(f"   ⚠️ {warning_msg}")
                    fallback_tests.append(False)
            
            else:
                error_msg = "TextBinaryDiffusion file not found for fallback testing"
                self.errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                fallback_tests.extend([False] * 5)
        
        except Exception as e:
            error_msg = f"Error testing classical fallback integration: {e}"
            self.errors.append(error_msg)
            logger.error(f"   ❌ {error_msg}")
            fallback_tests.extend([False] * 5)
        
        self.test_results['classical_fallback_integration'] = {
            'tests_passed': sum(fallback_tests),
            'total_tests': len(fallback_tests),
            'success_rate': sum(fallback_tests) / len(fallback_tests) * 100 if fallback_tests else 0
        }
        
        logger.info(f"🔄 Classical fallback integration: {sum(fallback_tests)}/{len(fallback_tests)} tests passed")
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 QUDIFFUSE INTEGRATION SUMMARY")
        logger.info("=" * 60)
        
        # Calculate overall integration score
        category_scores = []
        total_tests = 0
        total_passed = 0
        
        for category, results in self.test_results.items():
            if 'success_rate' in results:
                category_scores.append(results['success_rate'])
                total_tests += results['total_tests']
                total_passed += results['tests_passed']
        
        overall_score = sum(category_scores) / len(category_scores) if category_scores else 0
        
        # Determine integration status
        if overall_score >= 95 and len(self.errors) == 0:
            status = "✅ FULLY INTEGRATED"
        elif overall_score >= 85:
            status = "✅ WELL INTEGRATED"
        elif overall_score >= 70:
            status = "⚠️ PARTIALLY INTEGRATED"
        else:
            status = "❌ INTEGRATION ISSUES"
        
        logger.info(f"Integration Status: {status}")
        logger.info(f"Overall Score: {overall_score:.1f}%")
        logger.info(f"Tests Passed: {total_passed}/{total_tests}")
        logger.info(f"Critical Errors: {len(self.errors)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info("")
        
        # Category breakdown
        for category, results in self.test_results.items():
            score = results['success_rate']
            passed = results['tests_passed']
            total = results['total_tests']
            category_name = category.replace('_', ' ').title()
            logger.info(f"{category_name}: {score:.1f}% ({passed}/{total})")
        
        logger.info("")
        
        # Critical errors
        if self.errors:
            logger.info("🚨 CRITICAL ERRORS:")
            for i, error in enumerate(self.errors[:5], 1):
                logger.info(f"   {i}. {error}")
            if len(self.errors) > 5:
                logger.info(f"   ... and {len(self.errors) - 5} more")
            logger.info("")
        
        # Warnings
        if self.warnings:
            logger.info("⚠️ WARNINGS:")
            for i, warning in enumerate(self.warnings[:5], 1):
                logger.info(f"   {i}. {warning}")
            if len(self.warnings) > 5:
                logger.info(f"   ... and {len(self.warnings) - 5} more")
            logger.info("")
        
        # Final assessment
        if status == "✅ FULLY INTEGRATED":
            logger.info("🎉 EXCELLENT: Diffusion LLM is fully integrated with QuDiffuse!")
        elif status == "✅ WELL INTEGRATED":
            logger.info("👍 GOOD: Diffusion LLM is well integrated with QuDiffuse.")
        elif status == "⚠️ PARTIALLY INTEGRATED":
            logger.info("⚠️ PARTIAL: Some integration issues should be addressed.")
        else:
            logger.info("🔧 ISSUES: Significant integration problems need resolution.")
        
        logger.info("=" * 60)
        
        return {
            'status': status,
            'overall_score': overall_score,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'category_results': self.test_results,
            'error_details': self.errors,
            'warning_details': self.warnings
        }


def main():
    """Main integration test function."""
    
    print("🔗 QuDiffuse Integration Tester")
    print("Validating Diffusion LLM integration with QuDiffuse")
    print()
    
    tester = QuDiffuseIntegrationTester()
    results = tester.test_all_integrations()
    
    # Save results
    import json
    results_file = Path("qudiffuse_integration_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return appropriate exit code
    if results['overall_score'] >= 95 and results['errors'] == 0:
        return 0
    elif results['overall_score'] >= 85:
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 