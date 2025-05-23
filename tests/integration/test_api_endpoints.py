"""Integration tests for OpenSynthetics API endpoints."""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from opensynthetics.api.main import app
from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            self.test_config = Config({"storage": {"base_dir": str(self.temp_dir)}})
            
            with patch('opensynthetics.core.config.Config.load', return_value=self.test_config):
                with patch('opensynthetics.api.auth.get_api_key', return_value="test-key"):
                    self.client = TestClient(app)
                    yield
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "features" in data
        assert "api_stats" in data
    
    def test_root_endpoint(self):
        """Test root endpoint returns UI information."""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "OpenSynthetics API is running"
        assert data["docs_url"] == "/docs"
        assert data["ui_url"] == "/ui"
    
    def test_list_workspaces_empty(self):
        """Test listing workspaces when none exist."""
        response = self.client.get("/api/v1/workspaces")
        assert response.status_code == 200
        
        data = response.json()
        assert data["workspaces"] == []
    
    def test_create_workspace(self):
        """Test workspace creation."""
        workspace_data = {
            "name": "test_workspace",
            "description": "Test workspace for integration tests"
        }
        
        response = self.client.post(
            "/api/v1/workspaces",
            json=workspace_data,
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "test_workspace"
        assert data["description"] == "Test workspace for integration tests"
        assert "created_at" in data
        assert "updated_at" in data
        assert "path" in data
    
    def test_create_workspace_missing_name(self):
        """Test workspace creation with missing name."""
        workspace_data = {
            "description": "Test workspace without name"
        }
        
        response = self.client.post(
            "/api/v1/workspaces",
            json=workspace_data
        )
        assert response.status_code == 400
        assert "Workspace name is required" in response.json()["detail"]
    
    def test_get_workspace(self):
        """Test getting workspace details."""
        # First create a workspace
        workspace = Workspace.create(
            name="test_workspace",
            path=self.temp_dir / "test_workspace",
            description="Test workspace"
        )
        
        response = self.client.get(f"/api/v1/workspaces/{workspace.path}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "test_workspace"
        assert data["description"] == "Test workspace"
        assert data["datasets"] == []
    
    def test_get_nonexistent_workspace(self):
        """Test getting details of non-existent workspace."""
        response = self.client.get("/api/v1/workspaces/nonexistent")
        assert response.status_code == 404
    
    def test_list_strategies(self):
        """Test listing available generation strategies."""
        response = self.client.get("/api/v1/strategies")
        assert response.status_code == 200
        
        data = response.json()
        assert "strategies" in data
        strategies = data["strategies"]
        
        # Check for expected strategies
        assert "tabular_random" in strategies
        assert "customer_data" in strategies
        assert "sales_data" in strategies
        
        # Validate strategy structure
        tabular_strategy = strategies["tabular_random"]
        assert "name" in tabular_strategy
        assert "description" in tabular_strategy
        assert "schema" in tabular_strategy
        
        # Validate schema structure
        schema = tabular_strategy["schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "num_rows" in schema["properties"]
        assert "num_columns" in schema["properties"]
    
    def test_get_config(self):
        """Test getting system configuration."""
        response = self.client.get("/api/v1/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "base_dir" in data
        assert "environment" in data
        assert "has_openai_key" in data
        assert "version" in data
        
        # Ensure sensitive information is not exposed
        assert data["has_openai_key"] is False
        assert data["environment"] == "development"
    
    def test_generate_data_tabular_random(self):
        """Test data generation with tabular_random strategy."""
        generation_request = {
            "workspace": "test_workspace",
            "strategy": "tabular_random", 
            "parameters": {
                "num_rows": 10,
                "num_columns": 5
            },
            "dataset": "test_dataset"
        }
        
        response = self.client.post(
            "/api/v1/generate",
            json=generation_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 10
        assert data["strategy"] == "tabular_random"
        assert data["output_dataset"] == "test_dataset"
        assert data["workspace"] == "test_workspace"
        assert "timestamp" in data
        assert "sample_items" in data
        
        # Validate sample data structure
        sample_items = data["sample_items"]
        assert len(sample_items) <= 3  # Should be limited to 3 samples
        for item in sample_items:
            assert "id" in item
            assert isinstance(item["id"], int)
    
    def test_generate_data_missing_parameters(self):
        """Test data generation with missing required parameters."""
        generation_request = {
            "strategy": "tabular_random",
            "parameters": {"num_rows": 10}
            # Missing workspace and dataset
        }
        
        response = self.client.post(
            "/api/v1/generate",
            json=generation_request
        )
        assert response.status_code == 400
        assert "workspace, strategy, and dataset are required" in response.json()["detail"]
    
    def test_list_datasets_in_workspace(self):
        """Test listing datasets in a workspace."""
        # Create workspace and dataset
        workspace = Workspace.create(
            name="test_workspace",
            path=self.temp_dir / "test_workspace",
            description="Test workspace"
        )
        
        dataset = workspace.create_dataset(
            name="test_dataset",
            description="Test dataset"
        )
        
        response = self.client.get(f"/api/v1/workspaces/{workspace.path}/datasets")
        assert response.status_code == 200
        
        data = response.json()
        assert "datasets" in data
        datasets = data["datasets"]
        assert len(datasets) == 1
        assert datasets[0]["name"] == "test_dataset"
    
    def test_get_dataset_details(self):
        """Test getting dataset details."""
        # Create workspace and dataset
        workspace = Workspace.create(
            name="test_workspace", 
            path=self.temp_dir / "test_workspace",
            description="Test workspace"
        )
        
        dataset = workspace.create_dataset(
            name="test_dataset",
            description="Test dataset"
        )
        
        # Add some test data
        test_data = [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200}
        ]
        dataset.add_data(test_data)
        
        response = self.client.get(
            f"/api/v1/workspaces/{workspace.path}/datasets/test_dataset"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "test_dataset"
        assert data["description"] == "Test dataset"
        assert "tables" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. List workspaces (should be empty)
        response = self.client.get("/api/v1/workspaces")
        assert response.status_code == 200
        assert response.json()["workspaces"] == []
        
        # 2. Create workspace
        workspace_data = {
            "name": "e2e_workspace",
            "description": "End-to-end test workspace"
        }
        response = self.client.post("/api/v1/workspaces", json=workspace_data)
        assert response.status_code == 200
        workspace_info = response.json()
        
        # 3. List workspaces (should have one)
        response = self.client.get("/api/v1/workspaces")
        assert response.status_code == 200
        workspaces = response.json()["workspaces"]
        assert len(workspaces) == 1
        assert workspaces[0]["name"] == "e2e_workspace"
        
        # 4. Generate data
        generation_request = {
            "workspace": "e2e_workspace",
            "strategy": "customer_data",
            "parameters": {"count": 25},
            "dataset": "customers"
        }
        response = self.client.post("/api/v1/generate", json=generation_request)
        assert response.status_code == 200
        generation_result = response.json()
        assert generation_result["output_dataset"] == "customers"
        
        # 5. List datasets in workspace
        workspace_path = workspace_info["path"]
        response = self.client.get(f"/api/v1/workspaces/{workspace_path}/datasets")
        assert response.status_code == 200
        datasets = response.json()["datasets"]
        assert len(datasets) == 1
        assert datasets[0]["name"] == "customers"
        
        # 6. Get dataset details
        response = self.client.get(
            f"/api/v1/workspaces/{workspace_path}/datasets/customers"
        )
        assert response.status_code == 200
        dataset_info = response.json()
        assert dataset_info["name"] == "customers"
    
    def test_api_error_handling(self):
        """Test API error handling for various scenarios."""
        # Test invalid JSON
        response = self.client.post(
            "/api/v1/workspaces",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing content type
        response = self.client.post(
            "/api/v1/workspaces",
            data='{"name": "test"}'
        )
        assert response.status_code in [400, 422]
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        results = []
        
        def create_workspace(index: int) -> Dict[str, Any]:
            """Create a workspace in a separate thread."""
            workspace_data = {
                "name": f"concurrent_workspace_{index}",
                "description": f"Concurrent test workspace {index}"
            }
            response = self.client.post("/api/v1/workspaces", json=workspace_data)
            return {"index": index, "status_code": response.status_code}
        
        # Create multiple workspaces concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_workspace, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert len(results) == 5
        for result in results:
            assert result["status_code"] == 200
    
    def test_performance_benchmarks(self):
        """Test basic performance benchmarks."""
        # Workspace creation performance
        start_time = time.time()
        workspace_data = {
            "name": "perf_workspace",
            "description": "Performance test workspace"
        }
        response = self.client.post("/api/v1/workspaces", json=workspace_data)
        workspace_creation_time = time.time() - start_time
        
        assert response.status_code == 200
        assert workspace_creation_time < 0.5  # Should be under 500ms
        
        # Data generation performance
        start_time = time.time()
        generation_request = {
            "workspace": "perf_workspace",
            "strategy": "tabular_random",
            "parameters": {"num_rows": 1000, "num_columns": 10},
            "dataset": "perf_dataset"
        }
        response = self.client.post("/api/v1/generate", json=generation_request)
        generation_time = time.time() - start_time
        
        assert response.status_code == 200
        assert generation_time < 2.0  # Should be under 2 seconds for 1K rows
        
        # API response time
        start_time = time.time()
        response = self.client.get("/api/v1/strategies")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1  # Should be under 100ms


class TestAPISecurityAndValidation:
    """Tests for API security and input validation."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            self.test_config = Config({"storage": {"base_dir": str(self.temp_dir)}})
            
            with patch('opensynthetics.core.config.Config.load', return_value=self.test_config):
                # Test with no API key authentication
                self.client = TestClient(app)
                yield
    
    def test_api_key_required(self):
        """Test that API key is required for protected endpoints."""
        response = self.client.get("/api/v1/workspaces")
        assert response.status_code == 401
    
    def test_input_validation(self):
        """Test input validation for various endpoints."""
        with patch('opensynthetics.api.auth.get_api_key', return_value="test-key"):
            client = TestClient(app)
            
            # Test workspace creation with invalid data
            invalid_data = {
                "name": "",  # Empty name
                "description": "x" * 10000  # Very long description
            }
            response = client.post("/api/v1/workspaces", json=invalid_data)
            assert response.status_code == 400
            
            # Test generation with invalid parameters
            invalid_generation = {
                "workspace": "test",
                "strategy": "tabular_random",
                "parameters": {
                    "num_rows": -1,  # Negative number
                    "num_columns": 1000  # Too many columns
                },
                "dataset": "test"
            }
            response = client.post("/api/v1/generate", json=invalid_generation)
            # Should either reject or handle gracefully
            assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 