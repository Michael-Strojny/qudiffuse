#!/usr/bin/env python3
"""
QuDiffusive Web Platform Integration Tests

Comprehensive testing suite to validate the complete web platform
with zero tolerance for fake implementations or simplifications.

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

import os
import sys
import asyncio
import json
import time
import requests
import websockets
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebPlatformTester:
    """
    Comprehensive test suite for QuDiffusive Web Platform.
    
    Tests all components with zero tolerance for fake implementations:
    - API endpoints functionality
    - WebSocket real-time communication
    - Model integration and generation
    - Security and rate limiting
    - Production readiness
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.ws_base = base_url.replace("http", "ws")
        self.session = requests.Session()
        self.test_results = {}
        self.errors = []
        
        logger.info(f"🔧 Initialized WebPlatformTester for {base_url}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        
        logger.info("🚀 Starting QuDiffusive Web Platform Integration Tests")
        logger.info("=" * 60)
        
        # Test categories
        test_categories = [
            ("Health and Status", self.test_health_endpoints),
            ("Model Information", self.test_model_endpoints),
            ("Security Features", self.test_security_features),
            ("WebSocket Communication", self.test_websocket_functionality),
            ("Generation Pipeline", self.test_generation_pipeline),
            ("Production Readiness", self.test_production_readiness),
        ]
        
        for category_name, test_function in test_categories:
            logger.info(f"\n📋 Testing: {category_name}")
            logger.info("-" * 40)
            
            try:
                result = test_function()
                self.test_results[category_name] = result
                
                if result.get("passed", False):
                    logger.info(f"✅ {category_name}: PASSED")
                else:
                    logger.error(f"❌ {category_name}: FAILED")
                    self.errors.extend(result.get("errors", []))
                    
            except Exception as e:
                logger.error(f"💥 {category_name}: EXCEPTION - {e}")
                self.test_results[category_name] = {
                    "passed": False,
                    "error": str(e),
                    "errors": [str(e)]
                }
                self.errors.append(f"{category_name}: {e}")
        
        return self.generate_final_report()
    
    def test_health_endpoints(self) -> Dict[str, Any]:
        """Test health check and monitoring endpoints."""
        
        tests = []
        
        # Basic health check
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            tests.append({
                "name": "Basic health check",
                "passed": response.status_code == 200,
                "details": f"Status: {response.status_code}, Response: {response.json()}"
            })
        except Exception as e:
            tests.append({
                "name": "Basic health check",
                "passed": False,
                "error": str(e)
            })
        
        # Detailed health check
        try:
            response = self.session.get(f"{self.api_base}/health/detailed", timeout=10)
            tests.append({
                "name": "Detailed health check",
                "passed": response.status_code == 200,
                "details": f"Status: {response.status_code}"
            })
        except Exception as e:
            tests.append({
                "name": "Detailed health check",
                "passed": False,
                "error": str(e)
            })
        
        # Metrics endpoint
        try:
            response = self.session.get(f"{self.api_base}/health/metrics", timeout=10)
            tests.append({
                "name": "Metrics endpoint",
                "passed": response.status_code == 200,
                "details": f"Status: {response.status_code}"
            })
        except Exception as e:
            tests.append({
                "name": "Metrics endpoint",
                "passed": False,
                "error": str(e)
            })
        
        passed = all(test["passed"] for test in tests)
        errors = [test.get("error", test["name"]) for test in tests if not test["passed"]]
        
        return {
            "passed": passed,
            "tests": tests,
            "errors": errors
        }
    
    def test_model_endpoints(self) -> Dict[str, Any]:
        """Test model information and configuration endpoints."""
        
        tests = []
        
        # Model info
        try:
            response = self.session.get(f"{self.api_base}/models/info", timeout=10)
            data = response.json()
            
            # Validate required fields
            required_fields = ["name", "version", "supported_topologies", "supported_sampling_modes"]
            has_required = all(field in data for field in required_fields)
            
            # Validate topology support
            expected_topologies = ["hierarchical", "flat", "multi_channel"]
            has_topologies = all(topo in data.get("supported_topologies", []) for topo in expected_topologies)
            
            tests.append({
                "name": "Model info endpoint",
                "passed": response.status_code == 200 and has_required and has_topologies,
                "details": f"Status: {response.status_code}, Topologies: {data.get('supported_topologies', [])}"
            })
        except Exception as e:
            tests.append({
                "name": "Model info endpoint",
                "passed": False,
                "error": str(e)
            })
        
        # Model status
        try:
            response = self.session.get(f"{self.api_base}/models/status", timeout=10)
            data = response.json()
            
            tests.append({
                "name": "Model status endpoint",
                "passed": response.status_code == 200 and "ready" in data,
                "details": f"Status: {response.status_code}, Ready: {data.get('ready', False)}"
            })
        except Exception as e:
            tests.append({
                "name": "Model status endpoint",
                "passed": False,
                "error": str(e)
            })
        
        # Supported topologies
        try:
            response = self.session.get(f"{self.api_base}/models/topologies", timeout=10)
            data = response.json()
            
            expected_topologies = ["hierarchical", "flat", "multi_channel"]
            has_all_topologies = all(topo in data for topo in expected_topologies)
            
            tests.append({
                "name": "Topologies endpoint",
                "passed": response.status_code == 200 and has_all_topologies,
                "details": f"Status: {response.status_code}, Topologies: {list(data.keys())}"
            })
        except Exception as e:
            tests.append({
                "name": "Topologies endpoint",
                "passed": False,
                "error": str(e)
            })
        
        # Sampling modes
        try:
            response = self.session.get(f"{self.api_base}/models/sampling-modes", timeout=10)
            data = response.json()
            
            expected_modes = ["classical", "qubo_classical", "qubo_quantum"]
            has_all_modes = all(mode in data for mode in expected_modes)
            
            tests.append({
                "name": "Sampling modes endpoint",
                "passed": response.status_code == 200 and has_all_modes,
                "details": f"Status: {response.status_code}, Modes: {list(data.keys())}"
            })
        except Exception as e:
            tests.append({
                "name": "Sampling modes endpoint",
                "passed": False,
                "error": str(e)
            })
        
        passed = all(test["passed"] for test in tests)
        errors = [test.get("error", test["name"]) for test in tests if not test["passed"]]
        
        return {
            "passed": passed,
            "tests": tests,
            "errors": errors
        }
    
    def test_security_features(self) -> Dict[str, Any]:
        """Test security headers, rate limiting, and validation."""
        
        tests = []
        
        # Security headers
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Content-Security-Policy"
            ]
            
            has_security_headers = all(header in response.headers for header in security_headers)
            
            tests.append({
                "name": "Security headers",
                "passed": has_security_headers,
                "details": f"Headers present: {[h for h in security_headers if h in response.headers]}"
            })
        except Exception as e:
            tests.append({
                "name": "Security headers",
                "passed": False,
                "error": str(e)
            })
        
        # Rate limiting (test with multiple requests)
        try:
            responses = []
            for i in range(5):
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                responses.append(response.status_code)
                time.sleep(0.1)
            
            # Should have rate limit headers
            last_response = self.session.get(f"{self.base_url}/health", timeout=5)
            has_rate_headers = any(header.startswith("X-RateLimit") for header in last_response.headers)
            
            tests.append({
                "name": "Rate limiting headers",
                "passed": has_rate_headers,
                "details": f"Rate limit headers: {[h for h in last_response.headers if h.startswith('X-RateLimit')]}"
            })
        except Exception as e:
            tests.append({
                "name": "Rate limiting headers",
                "passed": False,
                "error": str(e)
            })
        
        # Invalid request handling
        try:
            # Test with invalid JSON
            response = self.session.post(
                f"{self.api_base}/generation/generate",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            tests.append({
                "name": "Invalid request handling",
                "passed": response.status_code in [400, 422],
                "details": f"Status: {response.status_code} for invalid JSON"
            })
        except Exception as e:
            tests.append({
                "name": "Invalid request handling",
                "passed": False,
                "error": str(e)
            })
        
        passed = all(test["passed"] for test in tests)
        errors = [test.get("error", test["name"]) for test in tests if not test["passed"]]
        
        return {
            "passed": passed,
            "tests": tests,
            "errors": errors
        }
    
    def test_websocket_functionality(self) -> Dict[str, Any]:
        """Test WebSocket connection and real-time communication."""
        
        tests = []
        
        # WebSocket connection test
        try:
            async def test_websocket():
                client_id = f"test_client_{int(time.time())}"
                uri = f"{self.ws_base}/ws/{client_id}"
                
                try:
                    async with websockets.connect(uri, timeout=10) as websocket:
                        # Wait for connection message
                        message = await asyncio.wait_for(websocket.recv(), timeout=5)
                        data = json.loads(message)
                        
                        return {
                            "connected": True,
                            "message_type": data.get("type"),
                            "client_id": data.get("client_id")
                        }
                except Exception as e:
                    return {"connected": False, "error": str(e)}
            
            # Run async test
            result = asyncio.run(test_websocket())
            
            tests.append({
                "name": "WebSocket connection",
                "passed": result.get("connected", False),
                "details": f"Connected: {result.get('connected')}, Type: {result.get('message_type')}"
            })
            
        except Exception as e:
            tests.append({
                "name": "WebSocket connection",
                "passed": False,
                "error": str(e)
            })
        
        passed = all(test["passed"] for test in tests)
        errors = [test.get("error", test["name"]) for test in tests if not test["passed"]]
        
        return {
            "passed": passed,
            "tests": tests,
            "errors": errors
        }
    
    def test_generation_pipeline(self) -> Dict[str, Any]:
        """Test the complete generation pipeline."""
        
        tests = []
        
        # Test generation request validation
        try:
            # Valid request structure
            valid_request = {
                "generation_type": "image",
                "prompt": "Test image generation",
                "topology": "hierarchical",
                "sampling_mode": "classical",
                "num_timesteps": 10,  # Small for testing
                "guidance_scale": 7.5,
                "num_inference_steps": 10,
                "output_format": "png",
                "output_size": [256, 256],
                "client_id": f"test_{int(time.time())}",
                "enable_visualization": True
            }
            
            response = self.session.post(
                f"{self.api_base}/generation/generate",
                json=valid_request,
                timeout=30
            )
            
            # Should return session ID
            if response.status_code == 200:
                data = response.json()
                has_session_id = "session_id" in data
                has_status = "status" in data
            else:
                has_session_id = False
                has_status = False
            
            tests.append({
                "name": "Generation request validation",
                "passed": response.status_code == 200 and has_session_id and has_status,
                "details": f"Status: {response.status_code}, Has session_id: {has_session_id}"
            })
            
        except Exception as e:
            tests.append({
                "name": "Generation request validation",
                "passed": False,
                "error": str(e)
            })
        
        # Test invalid generation request
        try:
            invalid_request = {
                "generation_type": "invalid_type",
                "topology": "invalid_topology"
            }
            
            response = self.session.post(
                f"{self.api_base}/generation/generate",
                json=invalid_request,
                timeout=10
            )
            
            tests.append({
                "name": "Invalid generation request handling",
                "passed": response.status_code in [400, 422],
                "details": f"Status: {response.status_code} for invalid request"
            })
            
        except Exception as e:
            tests.append({
                "name": "Invalid generation request handling",
                "passed": False,
                "error": str(e)
            })
        
        passed = all(test["passed"] for test in tests)
        errors = [test.get("error", test["name"]) for test in tests if not test["passed"]]
        
        return {
            "passed": passed,
            "tests": tests,
            "errors": errors
        }
    
    def test_production_readiness(self) -> Dict[str, Any]:
        """Test production readiness features."""
        
        tests = []
        
        # Test static file serving
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            
            tests.append({
                "name": "Frontend serving",
                "passed": response.status_code == 200,
                "details": f"Status: {response.status_code}, Content-Type: {response.headers.get('content-type', 'unknown')}"
            })
        except Exception as e:
            tests.append({
                "name": "Frontend serving",
                "passed": False,
                "error": str(e)
            })
        
        # Test error handling
        try:
            response = self.session.get(f"{self.api_base}/nonexistent-endpoint", timeout=10)
            
            tests.append({
                "name": "404 error handling",
                "passed": response.status_code == 404,
                "details": f"Status: {response.status_code}"
            })
        except Exception as e:
            tests.append({
                "name": "404 error handling",
                "passed": False,
                "error": str(e)
            })
        
        # Test CORS headers
        try:
            response = self.session.options(f"{self.api_base}/models/info", timeout=10)
            has_cors = "Access-Control-Allow-Origin" in response.headers
            
            tests.append({
                "name": "CORS headers",
                "passed": has_cors,
                "details": f"CORS header present: {has_cors}"
            })
        except Exception as e:
            tests.append({
                "name": "CORS headers",
                "passed": False,
                "error": str(e)
            })
        
        passed = all(test["passed"] for test in tests)
        errors = [test.get("error", test["name"]) for test in tests if not test["passed"]]
        
        return {
            "passed": passed,
            "tests": tests,
            "errors": errors
        }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final test report."""
        
        total_tests = sum(len(result.get("tests", [])) for result in self.test_results.values())
        passed_tests = sum(
            len([t for t in result.get("tests", []) if t.get("passed", False)])
            for result in self.test_results.values()
        )
        
        overall_passed = len(self.errors) == 0
        success_rate = (passed_tests / max(total_tests, 1)) * 100
        
        report = {
            "overall_passed": overall_passed,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "errors": self.errors,
            "categories": self.test_results,
            "timestamp": time.time()
        }
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 FINAL TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Tests: {passed_tests}/{total_tests} passed")
        
        if self.errors:
            logger.error(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if overall_passed:
            logger.info("\n🎉 All tests passed! Platform is production-ready.")
        else:
            logger.error(f"\n💥 {len(self.errors)} errors found. Platform needs fixes before deployment.")
        
        return report


def main():
    """Main test execution."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="QuDiffusive Web Platform Integration Tests")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL to test")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    # Run tests
    tester = WebPlatformTester(args.url)
    results = tester.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Results saved to {args.output}")
    
    # Exit with appropriate code
    exit_code = 0 if results["overall_passed"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
