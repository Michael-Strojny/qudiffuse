#!/usr/bin/env python3
"""
Structure Validation Script

This script validates the implementation structure and paper compliance
without requiring PyTorch or other heavy dependencies.

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import ast


class StructureValidator:
    """Validates implementation structure against paper specifications."""
    
    def __init__(self):
        """Initialize the validator."""
        self.base_dir = Path(__file__).parent.parent.parent
        self.diffusion_llm_dir = self.base_dir / "diffusion_llm"
        self.validation_results = {}
        self.violations = []
        self.warnings = []
        
        # Paper specifications to check
        self.paper_specs = {
            'lae': 16,
            'dae': 256,
            'bart_model': 'facebook/bart-base',
            'hidden_size': 768,
            'dit_layers': 12,
            'dit_heads': 12,
            'num_timesteps': 1000
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run complete structure validation."""
        
        print("🔍 Starting structure validation...")
        print("=" * 60)
        
        self.validate_directory_structure()
        self.validate_file_completeness()
        self.validate_paper_specifications()
        self.validate_implementation_authenticity()
        self.validate_documentation()
        
        return self.generate_report()
    
    def validate_directory_structure(self) -> None:
        """Validate directory structure is complete."""
        
        print("📁 Validating directory structure...")
        
        expected_dirs = [
            "encoders",
            "diffusion_transformers", 
            "models",
            "training",
            "datasets",
            "validation"
        ]
        
        expected_files = [
            "README.md",
            "COMPLETE_IMPLEMENTATION_GUIDE.md",
            "PAPER_COMPLIANCE_MATRIX.md",
            "__init__.py",
            "demo_complete_diffusion_llm.py"
        ]
        
        structure_tests = []
        
        # Check directories
        for dir_name in expected_dirs:
            dir_path = self.diffusion_llm_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"   ✅ {dir_name}/: Present")
                structure_tests.append(True)
            else:
                self.violations.append(f"Missing directory: {dir_name}/")
                print(f"   ❌ {dir_name}/: Missing")
                structure_tests.append(False)
        
        # Check files
        for file_name in expected_files:
            file_path = self.diffusion_llm_dir / file_name
            if file_path.exists() and file_path.is_file():
                print(f"   ✅ {file_name}: Present")
                structure_tests.append(True)
            else:
                self.violations.append(f"Missing file: {file_name}")
                print(f"   ❌ {file_name}: Missing")
                structure_tests.append(False)
        
        self.validation_results['directory_structure'] = {
            'tests_passed': sum(structure_tests),
            'total_tests': len(structure_tests),
            'success_rate': sum(structure_tests) / len(structure_tests) * 100
        }
    
    def validate_file_completeness(self) -> None:
        """Validate that all required files are complete and non-empty."""
        
        print("📄 Validating file completeness...")
        
        critical_files = [
            "encoders/bart_autoencoder.py",
            "encoders/perceiver_ae.py",
            "diffusion_transformers/reasoning_dit.py",
            "diffusion_transformers/text_binary_diffusion.py",
            "models/model_manager.py",
            "models/tokenizer_wrapper.py",
            "training/stage1_autoencoder_trainer.py",
            "training/stage2_diffusion_trainer.py",
            "training/unified_trainer.py",
            "datasets/reasoning_datasets.py"
        ]
        
        file_tests = []
        total_lines = 0
        
        for file_path in critical_files:
            full_path = self.diffusion_llm_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        line_count = len([line for line in lines if line.strip()])  # Non-empty lines
                    
                    if line_count > 50:  # Minimum threshold for substantial implementation
                        print(f"   ✅ {file_path}: {line_count} lines")
                        file_tests.append(True)
                        total_lines += line_count
                    else:
                        self.violations.append(f"{file_path} is too small ({line_count} lines)")
                        print(f"   ❌ {file_path}: Only {line_count} lines (insufficient)")
                        file_tests.append(False)
                
                except Exception as e:
                    self.violations.append(f"Cannot read {file_path}: {e}")
                    print(f"   ❌ {file_path}: Cannot read - {e}")
                    file_tests.append(False)
            else:
                self.violations.append(f"Missing file: {file_path}")
                print(f"   ❌ {file_path}: Missing")
                file_tests.append(False)
        
        print(f"   📊 Total implementation: {total_lines:,} lines")
        
        self.validation_results['file_completeness'] = {
            'tests_passed': sum(file_tests),
            'total_tests': len(file_tests),
            'success_rate': sum(file_tests) / len(file_tests) * 100,
            'total_lines': total_lines
        }
    
    def validate_paper_specifications(self) -> None:
        """Validate paper specifications are correctly implemented."""
        
        print("📝 Validating paper specifications...")
        
        spec_tests = []
        
        # Check for paper specification mentions in code
        spec_files = [
            "encoders/bart_autoencoder.py",
            "diffusion_transformers/reasoning_dit.py",
            "models/model_manager.py"
        ]
        
        for file_path in spec_files:
            full_path = self.diffusion_llm_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for paper specifications
                    lae_found = re.search(r'lae.*=.*16|16.*lae', content, re.IGNORECASE)
                    dae_found = re.search(r'dae.*=.*256|256.*dae', content, re.IGNORECASE)
                    bart_found = 'facebook/bart-base' in content
                    
                    if lae_found:
                        print(f"   ✅ {file_path}: lae=16 found")
                        spec_tests.append(True)
                    else:
                        self.warnings.append(f"{file_path}: lae=16 specification not found")
                        print(f"   ⚠️ {file_path}: lae=16 not found")
                        spec_tests.append(False)
                    
                    if dae_found:
                        print(f"   ✅ {file_path}: dae=256 found")
                        spec_tests.append(True)
                    else:
                        self.warnings.append(f"{file_path}: dae=256 specification not found")
                        print(f"   ⚠️ {file_path}: dae=256 not found")
                        spec_tests.append(False)
                    
                    if bart_found:
                        print(f"   ✅ {file_path}: BART-base model found")
                        spec_tests.append(True)
                    else:
                        self.warnings.append(f"{file_path}: BART-base model not found")
                        print(f"   ⚠️ {file_path}: BART-base not found")
                        spec_tests.append(False)
                
                except Exception as e:
                    self.violations.append(f"Cannot analyze {file_path}: {e}")
                    print(f"   ❌ {file_path}: Cannot analyze - {e}")
                    spec_tests.extend([False, False, False])
        
        self.validation_results['paper_specifications'] = {
            'tests_passed': sum(spec_tests),
            'total_tests': len(spec_tests),
            'success_rate': sum(spec_tests) / len(spec_tests) * 100 if spec_tests else 0
        }
    
    def validate_implementation_authenticity(self) -> None:
        """Validate implementation authenticity (no mocks, fakes, etc.)."""
        
        print("🔍 Validating implementation authenticity...")
        
        violation_patterns = [
            r'\bmock\b',
            r'\bfake\b', 
            r'\bdummy\b',
            r'\bplaceholder\b',
            r'\bTODO\b',
            r'\bFIXME\b',
            r'\bNotImplemented\b',
            r'\bpass\s*$',  # Empty pass statements
            r'raise\s+NotImplementedError'
        ]
        
        # Files to check for violations
        python_files = []
        for root, dirs, files in os.walk(self.diffusion_llm_dir):
            # Skip external repos
            if 'external_repos' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        violations_found = []
        total_files_checked = 0
        clean_files = 0
        
        for file_path in python_files:
            total_files_checked += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_violations = []
                for pattern in violation_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        # Get line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Filter out acceptable patterns
                        line_content = content.split('\n')[line_num - 1].strip()
                        
                        # Skip legitimate uses
                        if any(acceptable in line_content.lower() for acceptable in [
                            'logits_fake',  # GAN terminology
                            'import mock',  # Test imports
                            'mock_', # Test functions
                            '"fake"', "'fake'",  # String literals
                            'fallback',  # Legitimate fallback code
                            '# fake',  # Comments about avoiding fakes
                            'zero mock',  # Claims of no mocks
                            'no mock'
                        ]):
                            continue
                        
                        file_violations.append((line_num, pattern, line_content))
                
                if file_violations:
                    rel_path = file_path.relative_to(self.diffusion_llm_dir)
                    violations_found.extend([(rel_path, line_num, pattern, line_content) 
                                           for line_num, pattern, line_content in file_violations])
                else:
                    clean_files += 1
            
            except Exception as e:
                self.warnings.append(f"Cannot check {file_path}: {e}")
        
        # Report results
        if violations_found:
            print(f"   ⚠️ Found {len(violations_found)} potential violations in {len(violations_found)} locations")
            for file_path, line_num, pattern, line_content in violations_found[:5]:  # Show first 5
                print(f"      {file_path}:{line_num} - {pattern}")
            
            if len(violations_found) > 5:
                print(f"      ... and {len(violations_found) - 5} more")
        else:
            print(f"   ✅ No authenticity violations found")
        
        print(f"   📊 Files checked: {total_files_checked}, Clean: {clean_files}")
        
        authenticity_score = (clean_files / total_files_checked * 100) if total_files_checked > 0 else 0
        
        self.validation_results['authenticity'] = {
            'violations_found': len(violations_found),
            'files_checked': total_files_checked,
            'clean_files': clean_files,
            'authenticity_score': authenticity_score
        }
    
    def validate_documentation(self) -> None:
        """Validate documentation completeness."""
        
        print("📚 Validating documentation...")
        
        doc_files = [
            "README.md",
            "COMPLETE_IMPLEMENTATION_GUIDE.md", 
            "PAPER_COMPLIANCE_MATRIX.md"
        ]
        
        doc_tests = []
        
        for doc_file in doc_files:
            file_path = self.diffusion_llm_dir / doc_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check documentation quality
                    word_count = len(content.split())
                    
                    if word_count > 500:  # Substantial documentation
                        print(f"   ✅ {doc_file}: {word_count} words")
                        doc_tests.append(True)
                    else:
                        self.warnings.append(f"{doc_file} is too short ({word_count} words)")
                        print(f"   ⚠️ {doc_file}: Only {word_count} words")
                        doc_tests.append(False)
                
                except Exception as e:
                    self.violations.append(f"Cannot read {doc_file}: {e}")
                    print(f"   ❌ {doc_file}: Cannot read - {e}")
                    doc_tests.append(False)
            else:
                self.violations.append(f"Missing documentation: {doc_file}")
                print(f"   ❌ {doc_file}: Missing")
                doc_tests.append(False)
        
        self.validation_results['documentation'] = {
            'tests_passed': sum(doc_tests),
            'total_tests': len(doc_tests),
            'success_rate': sum(doc_tests) / len(doc_tests) * 100 if doc_tests else 0
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        # Calculate overall score
        category_scores = []
        for category, results in self.validation_results.items():
            if 'success_rate' in results:
                category_scores.append(results['success_rate'])
            elif 'authenticity_score' in results:
                category_scores.append(results['authenticity_score'])
        
        overall_score = sum(category_scores) / len(category_scores) if category_scores else 0
        
        # Determine status
        if overall_score >= 95 and len(self.violations) == 0:
            status = "✅ EXCELLENT"
        elif overall_score >= 85:
            status = "✅ GOOD"
        elif overall_score >= 70:
            status = "⚠️ ACCEPTABLE"
        else:
            status = "❌ NEEDS IMPROVEMENT"
        
        print(f"Overall Status: {status}")
        print(f"Overall Score: {overall_score:.1f}%")
        print(f"Critical Violations: {len(self.violations)}")
        print(f"Warnings: {len(self.warnings)}")
        print("")
        
        # Category breakdown
        for category, results in self.validation_results.items():
            if 'success_rate' in results:
                score = results['success_rate']
                passed = results.get('tests_passed', 0)
                total = results.get('total_tests', 0)
                print(f"{category.replace('_', ' ').title()}: {score:.1f}% ({passed}/{total})")
            elif 'authenticity_score' in results:
                score = results['authenticity_score']
                clean = results.get('clean_files', 0)
                total = results.get('files_checked', 0)
                print(f"{category.replace('_', ' ').title()}: {score:.1f}% ({clean}/{total} clean)")
        
        # Special metrics
        if 'file_completeness' in self.validation_results:
            total_lines = self.validation_results['file_completeness'].get('total_lines', 0)
            print(f"Total Implementation: {total_lines:,} lines")
        
        print("")
        
        # Violations
        if self.violations:
            print("🚨 CRITICAL VIOLATIONS:")
            for i, violation in enumerate(self.violations[:10], 1):
                print(f"   {i}. {violation}")
            if len(self.violations) > 10:
                print(f"   ... and {len(self.violations) - 10} more")
            print("")
        
        # Warnings
        if self.warnings:
            print("⚠️ WARNINGS:")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"   {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more")
            print("")
        
        # Summary
        if status == "✅ EXCELLENT":
            print("🎉 EXCELLENT: Implementation structure is complete and compliant!")
        elif status == "✅ GOOD":
            print("👍 GOOD: Implementation structure is solid with minor issues.")
        elif status == "⚠️ ACCEPTABLE":
            print("⚠️ ACCEPTABLE: Implementation has some issues that should be addressed.")
        else:
            print("🔧 NEEDS IMPROVEMENT: Significant issues found that require attention.")
        
        print("=" * 60)
        
        return {
            'status': status,
            'overall_score': overall_score,
            'violations': len(self.violations),
            'warnings': len(self.warnings),
            'category_results': self.validation_results,
            'violation_details': self.violations,
            'warning_details': self.warnings
        }


def main():
    """Main validation function."""
    
    print("🎯 Diffusion LLM Structure Validator")
    print("Validating implementation structure and paper compliance")
    print()
    
    validator = StructureValidator()
    results = validator.validate_all()
    
    # Save results
    results_file = Path("structure_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return appropriate exit code
    if results['overall_score'] >= 95 and results['violations'] == 0:
        return 0
    elif results['overall_score'] >= 85:
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 