"""Training utilities for OpenSynthetics."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import OpenSyntheticsError, TrainingError
from opensynthetics.core.workspace import Dataset, Workspace


class TrainingManager:
    """Manager for training and fine-tuning models."""

    def __init__(self, workspace: Workspace) -> None:
        """Initialize training manager.

        Args:
            workspace: Workspace to use
        """
        self.workspace = workspace
        self.config = Config.load()
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize the database for training job tracking."""
        db_path = self.workspace.path / "workspace.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ft_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                training_file TEXT NOT NULL,
                parameters TEXT NOT NULL,
                result TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _execute_db_query(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[Dict[str, Any]]:
        """Execute a database query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
            
        Raises:
            TrainingError: If query fails
        """
        db_path = self.workspace.path / "workspace.db"
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            results = [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return results
        except sqlite3.Error as e:
            raise TrainingError(f"Database query failed: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
                
    def _execute_db_update(self, query: str, params: Optional[Union[tuple, dict]] = None) -> int:
        """Execute a database update.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            int: Number of affected rows
            
        Raises:
            TrainingError: If update fails
        """
        db_path = self.workspace.path / "workspace.db"
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            affected_rows = cursor.rowcount
            conn.commit()
            return affected_rows
        except sqlite3.Error as e:
            raise TrainingError(f"Database update failed: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def prepare_openai_ft_data(
        self,
        dataset_name: str,
        output_path: Optional[Path] = None,
        prompt_field: str = "problem_description",
        completion_field: str = "expected_solution",
        table: str = "data",
        filter_query: Optional[str] = None,
    ) -> Path:
        """Prepare data for OpenAI fine-tuning.

        Args:
            dataset_name: Dataset name
            output_path: Output path for JSONL file
            prompt_field: Field to use as prompt
            completion_field: Field to use as completion
            table: Table to query
            filter_query: SQL WHERE clause to filter data

        Returns:
            Path: Path to generated JSONL file

        Raises:
            TrainingError: If preparation fails
        """
        try:
            # Validate dataset name
            if not dataset_name or not isinstance(dataset_name, str):
                raise TrainingError(f"Invalid dataset name: {dataset_name}")
                
            # Get dataset
            try:
                dataset = self.workspace.get_dataset(dataset_name)
            except Exception as e:
                raise TrainingError(f"Failed to get dataset '{dataset_name}': {e}")
            
            # Validate field names
            if not prompt_field or not completion_field:
                raise TrainingError("Prompt and completion field names cannot be empty")
                
            # Validate table name
            tables = dataset.get_tables()
            if table not in tables:
                available_tables = ", ".join(tables)
                raise TrainingError(f"Table '{table}' not found in dataset. Available tables: {available_tables}")
            
            # Build query
            query = f'SELECT * FROM "{table}"'
            if filter_query:
                query += f" WHERE {filter_query}"
                
            # Fetch data
            try:
                data = dataset.query(query)
            except Exception as e:
                raise TrainingError(f"Failed to query dataset: {e}")
            
            if not data:
                raise TrainingError(f"No data found in table '{table}' with the given filter")
                
            # Check if required fields exist in the data
            if data and (prompt_field not in data[0] or completion_field not in data[0]):
                available_fields = ", ".join(data[0].keys())
                raise TrainingError(f"Required fields '{prompt_field}' or '{completion_field}' not found in data. Available fields: {available_fields}")
            
            # Prepare fine-tuning data
            ft_data = []
            for item in data:
                if prompt_field not in item or completion_field not in item:
                    logger.warning(f"Skipping item missing required fields: {item}")
                    continue
                    
                ft_data.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": item[prompt_field]},
                        {"role": "assistant", "content": item[completion_field]},
                    ]
                })
            
            if not ft_data:
                raise TrainingError("No valid data found for fine-tuning")
                
            # Determine output path
            if output_path is None:
                output_dir = self.workspace.path / "exports"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{dataset_name}_openai_ft.jsonl"
            else:
                # Ensure the output path is absolute
                output_path = Path(output_path).resolve()
                # Create parent directories if they don't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
            # Write JSONL file
            with open(output_path, "w") as f:
                for item in ft_data:
                    f.write(json.dumps(item) + "\n")
                    
            logger.info(f"Prepared {len(ft_data)} examples for OpenAI fine-tuning at {output_path}")
            return output_path
            
        except TrainingError:
            # Re-raise TrainingError as is
            raise
        except Exception as e:
            raise TrainingError(f"Failed to prepare OpenAI fine-tuning data: {e}")

    def start_openai_ft_job(
        self,
        training_file: Path,
        model: str = "gpt-3.5-turbo",
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start an OpenAI fine-tuning job.

        Args:
            training_file: Path to training file
            model: Base model to fine-tune
            suffix: Suffix for fine-tuned model name
            hyperparameters: Fine-tuning hyperparameters

        Returns:
            Dict[str, Any]: Job information

        Raises:
            TrainingError: If job creation fails
        """
        try:
            # Validate training file
            training_file = Path(training_file).resolve()
            if not training_file.exists():
                raise TrainingError(f"Training file not found: {training_file}")
                
            # Validate model
            if not model:
                raise TrainingError("Model name cannot be empty")
                
            # Import provider
            try:
                from opensynthetics.llm_core.providers import OpenAIProvider
            except ImportError as e:
                raise TrainingError(f"Failed to import OpenAI provider: {e}")
                
            # Create provider
            try:
                provider = OpenAIProvider(self.config)
            except Exception as e:
                raise TrainingError(f"Failed to create OpenAI provider: {e}")
            
            # Upload file
            try:
                with open(training_file, "rb") as f:
                    file_response = provider.client.files.create(
                        file=f,
                        purpose="fine-tune",
                    )
                    
                file_id = file_response.id
                logger.info(f"Uploaded file with ID: {file_id}")
            except Exception as e:
                raise TrainingError(f"Failed to upload training file: {e}")
            
            # Create fine-tuning job
            job_params = {
                "training_file": file_id,
                "model": model,
            }
            
            if suffix:
                job_params["suffix"] = suffix
                
            if hyperparameters:
                job_params.update(hyperparameters)
            
            try:
                job_response = provider.client.fine_tuning.jobs.create(**job_params)
            except Exception as e:
                raise TrainingError(f"Failed to create fine-tuning job: {e}")
            
            # Store job info in workspace database
            job_info = {
                "id": job_response.id,
                "status": job_response.status,
                "model": job_response.model,
                "created_at": job_response.created_at,
                "training_file": file_id,
                "parameters": json.dumps(job_params),
                "result": "",
            }
            
            insert_query = """
                INSERT INTO ft_jobs (id, status, model, created_at, training_file, parameters, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            self._execute_db_update(
                insert_query,
                (
                    job_info["id"],
                    job_info["status"],
                    job_info["model"],
                    job_info["created_at"],
                    job_info["training_file"],
                    job_info["parameters"],
                    job_info["result"],
                ),
            )
            
            logger.info(f"Started fine-tuning job with ID: {job_response.id}")
            
            return {
                "job_id": job_response.id,
                "status": job_response.status,
                "model": job_response.model,
                "created_at": job_response.created_at,
            }
            
        except TrainingError:
            # Re-raise TrainingError as is
            raise
        except Exception as e:
            raise TrainingError(f"Failed to start OpenAI fine-tuning job: {e}")

    def check_ft_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of a fine-tuning job.

        Args:
            job_id: Job ID

        Returns:
            Dict[str, Any]: Job status

        Raises:
            TrainingError: If status check fails
        """
        try:
            # Validate job ID
            if not job_id:
                raise TrainingError("Job ID cannot be empty")
                
            # Import provider
            from opensynthetics.llm_core.providers import OpenAIProvider
            
            # Create provider
            try:
                provider = OpenAIProvider(self.config)
            except Exception as e:
                raise TrainingError(f"Failed to create OpenAI provider: {e}")
            
            # Check job status
            try:
                job_response = provider.client.fine_tuning.jobs.retrieve(job_id)
            except Exception as e:
                raise TrainingError(f"Failed to retrieve fine-tuning job: {e}")
            
            # Update job info in workspace database
            job_result = ""
            if job_response.status == "succeeded":
                job_result = json.dumps({
                    "fine_tuned_model": job_response.fine_tuned_model,
                    "finished_at": job_response.finished_at,
                    "training_metrics": job_response.training_metrics,
                })
                
            update_query = """
                UPDATE ft_jobs
                SET status = ?, result = ?
                WHERE id = ?
            """
            
            self._execute_db_update(
                update_query,
                (job_response.status, job_result, job_id),
            )
            
            return {
                "job_id": job_response.id,
                "status": job_response.status,
                "model": job_response.model,
                "fine_tuned_model": job_response.fine_tuned_model,
                "created_at": job_response.created_at,
                "finished_at": job_response.finished_at,
                "training_metrics": job_response.training_metrics,
            }
            
        except TrainingError:
            # Re-raise TrainingError as is
            raise
        except Exception as e:
            raise TrainingError(f"Failed to check fine-tuning job status: {e}")
            
    def list_ft_jobs(self) -> List[Dict[str, Any]]:
        """List all fine-tuning jobs.
        
        Returns:
            List[Dict[str, Any]]: List of fine-tuning jobs
            
        Raises:
            TrainingError: If listing fails
        """
        try:
            query = "SELECT * FROM ft_jobs ORDER BY created_at DESC"
            jobs = self._execute_db_query(query)
            
            # Parse JSON fields
            for job in jobs:
                if job.get("parameters"):
                    try:
                        job["parameters"] = json.loads(job["parameters"])
                    except json.JSONDecodeError:
                        job["parameters"] = {}
                        
                if job.get("result"):
                    try:
                        job["result"] = json.loads(job["result"])
                    except json.JSONDecodeError:
                        job["result"] = {}
                        
            return jobs
        except Exception as e:
            raise TrainingError(f"Failed to list fine-tuning jobs: {e}")