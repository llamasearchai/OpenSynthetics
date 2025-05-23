#!/usr/bin/env python3
"""
Comprehensive API Test Suite for OpenSynthetics
Tests all major API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
from typing import Dict, Any, List

class OpenSyntheticsAPITester:
    """Comprehensive API tester for OpenSynthetics platform."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "default-key"):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': api_key
        }
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "[PASS]" if success else "[FAIL]"
        self.test_results.append((test_name, success, details))
        print(f"{status} {test_name}: {details}")
    
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    self.log_test("Health Endpoint", True, f"Version: {data.get('version', 'unknown')}")
                    return True
                else:
                    self.log_test("Health Endpoint", False, f"Invalid status: {data.get('status')}")
                    return False
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_workspaces_endpoint(self) -> bool:
        """Test the workspaces listing endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/workspaces", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                workspaces = data.get("workspaces", [])
                self.log_test("Workspaces List", True, f"Found {len(workspaces)} workspaces")
                return True
            else:
                self.log_test("Workspaces List", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Workspaces List", False, f"Exception: {str(e)}")
            return False
    
    def test_strategies_endpoint(self) -> bool:
        """Test the strategies endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/strategies", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                strategies = data.get("strategies", {})
                self.log_test("Strategies List", True, f"Found {len(strategies)} strategies")
                return True
            else:
                self.log_test("Strategies List", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Strategies List", False, f"Exception: {str(e)}")
            return False
    
    def test_config_endpoint(self) -> bool:
        """Test the config endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/config", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                base_dir = data.get("base_dir")
                self.log_test("Config Endpoint", True, f"Base dir: {base_dir}")
                return True
            else:
                self.log_test("Config Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Config Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_workspace_creation(self) -> bool:
        """Test workspace creation."""
        try:
            workspace_data = {
                "name": f"api_test_workspace_{int(time.time())}",
                "description": "Test workspace created via API",
                "tags": ["test", "api"]
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/workspaces", 
                headers=self.headers,
                json=workspace_data
            )
            
            if response.status_code == 200:
                data = response.json()
                workspace_name = data.get("name")
                self.log_test("Workspace Creation", True, f"Created: {workspace_name}")
                return True
            else:
                self.log_test("Workspace Creation", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Workspace Creation", False, f"Exception: {str(e)}")
            return False
    
    def test_data_generation(self) -> bool:
        """Test data generation endpoint."""
        try:
            generation_data = {
                "workspace": f"api_test_workspace_{int(time.time())}",
                "strategy": "tabular_random",
                "parameters": {
                    "num_rows": 100,
                    "num_columns": 5
                },
                "dataset": "test_dataset"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                headers=self.headers,
                json=generation_data
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("count", 0)
                self.log_test("Data Generation", True, f"Generated {count} rows")
                return True
            else:
                self.log_test("Data Generation", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Data Generation", False, f"Exception: {str(e)}")
            return False
    
    def test_web_ui_access(self) -> bool:
        """Test web UI accessibility."""
        try:
            response = requests.get(f"{self.base_url}/ui/")
            
            if response.status_code == 200:
                content = response.text
                if "OpenSynthetics" in content and "<!DOCTYPE html>" in content:
                    self.log_test("Web UI Access", True, "UI loads correctly")
                    return True
                else:
                    self.log_test("Web UI Access", False, "Invalid HTML content")
                    return False
            else:
                self.log_test("Web UI Access", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Web UI Access", False, f"Exception: {str(e)}")
            return False
    
    def test_integrations_endpoint(self) -> bool:
        """Test integrations endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/integrations/api-keys", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                api_keys = data.get("api_keys", [])
                self.log_test("Integrations API", True, f"Found {len(api_keys)} API keys")
                return True
            else:
                self.log_test("Integrations API", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Integrations API", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        print("🧪 OpenSynthetics API Comprehensive Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_health_endpoint,
            self.test_workspaces_endpoint,
            self.test_strategies_endpoint,
            self.test_config_endpoint,
            self.test_workspace_creation,
            self.test_data_generation,
            self.test_web_ui_access,
            self.test_integrations_endpoint,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
            print()  # Add spacing between tests
        
        success_rate = (passed / total) * 100
        
        print("=" * 60)
        print(f"[INFO] TEST RESULTS SUMMARY")
        print(f"="*60)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"{"="*60}")
        
        if success_rate == 100:
            print("[SUCCESS] ALL TESTS PASSED! API is fully functional.")
        elif success_rate >= 80:
            print(f"[WARNING] {total - passed} tests failed. Please review the issues.")
        else:
            print(f"[ERROR] {total - passed} tests failed. API needs attention.")
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": success_rate,
            "results": self.test_results
        }


def main():
    """Main test execution."""
    print("Starting OpenSynthetics API Test Suite...")
    print("Make sure the API server is running on http://localhost:8000")
    print()
    
    tester = OpenSyntheticsAPITester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results["success_rate"] == 100:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main() 