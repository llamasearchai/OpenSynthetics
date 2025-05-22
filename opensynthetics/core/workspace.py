"""Workspace management for OpenSynthetics."""

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile # Added for __main__ example

from loguru import logger

from opensynthetics.core.config import Config
# Define custom exceptions if not already broadly available
class WorkspaceError(Exception):
    pass

class DatasetError(Exception):
    pass

METADATA_FILE = "metadata.json"
WORKSPACE_CONFIG_FILE = "workspace_config.json" # Could be used for workspace-specific settings
DATASETS_DIR = "datasets"
MODELS_DIR = "models"
EMBEDDINGS_DIR = "embeddings"
EXPORTS_DIR = "exports"
DB_FILE = "workspace.db" # Main DB for workspace-level metadata like registered datasets

class WorkspaceMetadata:
    """Model for workspace metadata."""
    def __init__(self, name: str, description: str = "", created_at: Optional[str] = None, 
                 updated_at: Optional[str] = None, version: str = "0.1.0", tags: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = updated_at or datetime.now(timezone.utc).isoformat()
        self.version = version
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceMetadata":
        return cls(**data)

class Dataset:
    """Manages a dataset within a workspace."""
    def __init__(self, path: Path, name: str, workspace: "Workspace"):
        self.path = path
        self.name = name
        self.workspace = workspace
        self.db_path = self.path / f"{name}.db"
        self.metadata_path = self.path / METADATA_FILE
        self._metadata: Optional[Dict[str, Any]] = None
        self._init_dataset()

    def _init_dataset(self) -> None:
        """Initialize dataset directory and database."""
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            # Create default metadata if it doesn't exist
            default_meta = {
                "name": self.name,
                "description": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tags": [],
                "schema": {} # Could store inferred or defined schema for tables
            }
            self._save_metadata(default_meta)
        self._load_metadata() # Load it into self._metadata

    def _connect_db(self) -> sqlite3.Connection:
        """Connect to the dataset's SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row # Access columns by name
        # Apply PRAGMA settings from global config if available
        # config = Config.load()
        # pragmas = config.storage.sqlite_pragmas
        # for key, value in pragmas.items():
        #     conn.execute(f"PRAGMA {key} = {value}")
        return conn

    def _load_metadata(self) -> None:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self._metadata = json.load(f)
        else:
            # This case should be handled by _init_dataset creating a default one
            self._metadata = {} 

    def _save_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save dataset metadata to metadata.json."""
        meta_to_save = metadata or self._metadata
        if meta_to_save is not None:
            meta_to_save["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.metadata_path, "w") as f:
                json.dump(meta_to_save, f, indent=2)
            self._metadata = meta_to_save # Update cache

    @property
    def description(self) -> str:
        return self._metadata.get("description", "") if self._metadata else ""

    @property
    def tags(self) -> List[str]:
        return self._metadata.get("tags", []) if self._metadata else []

    def add_data(self, data: List[Dict[str, Any]], table: str = "data") -> None:
        """Add data to a table in the dataset.

        Args:
            data: List of dictionaries representing rows.
            table: Name of the table to add data to.
        """
        if not data:
            return

        conn = self._connect_db()
        cursor = conn.cursor()

        # Infer schema from the first item and create table if not exists
        columns = list(data[0].keys())
        column_defs = ", ".join([f'"{col}" TEXT' for col in columns]) # Simple TEXT type for now
        # Ensure table name is safe
        safe_table_name = "".join(c if c.isalnum() or c == '_' else '_' for c in table)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS \"{safe_table_name}\" ({column_defs})")

        # Insert data
        placeholders = ", ".join(["?" for _ in columns])
        rows_to_insert = [[item.get(col) for col in columns] for item in data]
        
        try:
            column_names = ", ".join([f'"{col}"' for col in columns])
            cursor.executemany(f'INSERT INTO "{safe_table_name}" ({column_names}) VALUES ({placeholders})', rows_to_insert)
            conn.commit()
            logger.info(f"Added {len(data)} rows to table '{safe_table_name}' in dataset '{self.name}'.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding data to dataset '{self.name}': {e}")
            raise DatasetError(f"Failed to add data: {e}")
        finally:
            conn.close()
        self._update_schema_in_metadata(safe_table_name, columns)

    def _update_schema_in_metadata(self, table_name: str, columns: List[str]):
        """Update the schema information in the dataset metadata."""
        if self._metadata is None: self._load_metadata()
        if self._metadata is None: self._metadata = {} # Should not happen if init is correct
        
        schemas = self._metadata.setdefault("schema", {})
        # For simplicity, just storing column names. Could store types too.
        schemas[table_name] = {col: "TEXT" for col in columns} 
        self._save_metadata()

    def query(self, query_string: str, params: Optional[Union[Dict, List]] = None) -> List[Dict[str, Any]]:
        """Query data from the dataset using SQL.

        Args:
            query_string: SQL query string.
            params: Parameters for the SQL query.

        Returns:
            List of dictionaries representing rows.
        """
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query_string, params)
            else:
                cursor.execute(query_string)
            
            # Convert rows to dictionaries
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except sqlite3.Error as e:
            logger.error(f"Error querying dataset '{self.name}': {e}")
            raise DatasetError(f"SQL query failed: {e}")
        finally:
            conn.close()

    def get_tables(self) -> List[str]:
        """Get a list of tables in the dataset."""
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row["name"] for row in cursor.fetchall()]
            # Filter out sqlite internal tables if any, though typically not needed with sqlite_master
            return [t for t in tables if not t.startswith("sqlite_")]
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the dataset."""
        if self._metadata is None: self._load_metadata()
        stats = self._metadata.copy() if self._metadata else {}
        stats["name"] = self.name
        stats["path"] = str(self.path)
        
        # Add table information
        tables = self.get_tables()
        stats["tables"] = {}
        
        conn = self._connect_db()
        cursor = conn.cursor()
        
        for table in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) as count FROM "{table}"')
                row_count = cursor.fetchone()["count"]
                
                # Get sample columns (up to 5)
                cursor.execute(f'SELECT * FROM "{table}" LIMIT 1')
                columns = [column[0] for column in cursor.description]
                
                stats["tables"][table] = {
                    "row_count": row_count,
                    "columns": columns[:5] + (["..."] if len(columns) > 5 else [])
                }
            except Exception as e:
                logger.error(f"Error getting stats for table '{table}': {e}")
                stats["tables"][table] = {
                    "error": str(e)
                }
        
        conn.close()
        return stats

    def update_metadata(self, description: Optional[str] = None, tags: Optional[List[str]] = None) -> None:
        """Update dataset metadata.
        
        Args:
            description: New description
            tags: New tags
        """
        if self._metadata is None: self._load_metadata()
        if self._metadata is None: self._metadata = {}
        
        if description is not None:
            self._metadata["description"] = description
            
        if tags is not None:
            self._metadata["tags"] = tags
            
        self._save_metadata()

class Workspace:
    """Manages an OpenSynthetics workspace."""
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).resolve() # Ensure absolute path
        self.config = Config.load() # Global config
        self._metadata: Optional[WorkspaceMetadata] = None
        self._db_path = self.path / DB_FILE
        
        if not self.path.exists() or not (self.path / METADATA_FILE).exists():
            # This path is for loading existing workspaces. Creation should use Workspace.create()
            raise WorkspaceError(f"Workspace not found or metadata missing at {self.path}")

        self._load_metadata()
        self._init_db() # Initialize workspace-level DB

    def _init_db(self) -> None:
        """Initialize the workspace SQLite database for managing datasets etc."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()
        # Table to register datasets within the workspace
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registered_datasets (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                description TEXT,
                tags TEXT, -- JSON string for list of tags
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def name(self) -> str:
        return self._metadata.name if self._metadata else self.path.name

    @property
    def metadata(self) -> WorkspaceMetadata:
        if not self._metadata:
            self._load_metadata()
        if not self._metadata: # Should not happen if constructor works
             raise WorkspaceError("Workspace metadata could not be loaded.")
        return self._metadata 

    def _load_metadata(self) -> None:
        metadata_path = self.path / METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
                self._metadata = WorkspaceMetadata.from_dict(data)
        else:
            # This should be caught by constructor if file is missing for an existing workspace
            logger.warning(f"Metadata file not found for workspace {self.path}. This might indicate an issue.")
            # Create a default one based on path name if truly necessary (though create() should handle new ones)
            self._metadata = WorkspaceMetadata(name=self.path.name) 
            self._save_metadata() # Save this default one

    def _save_metadata(self) -> None:
        if self._metadata:
            self._metadata.updated_at = datetime.now(timezone.utc).isoformat()
            metadata_path = self.path / METADATA_FILE
            with open(metadata_path, "w") as f:
                json.dump(self._metadata.to_dict(), f, indent=2)

    @classmethod
    def create(
        cls,
        name: str,
        path: Optional[Union[str, Path]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> "Workspace":
        """Create a new workspace."""
        config = Config.load()
        base_dir = config.base_dir

        if path:
            workspace_path = Path(path).resolve()
            # If a full path is given, it might include the intended workspace name
            # or it might be a parent directory. We'll assume it IS the workspace path.
            # If name differs from path.name, it could be confusing, but let's allow it.
        else:
            workspace_path = base_dir / name

        if workspace_path.exists() and list(workspace_path.iterdir()):
            raise WorkspaceError(f"Workspace directory {workspace_path} already exists and is not empty.")
        
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (workspace_path / DATASETS_DIR).mkdir(exist_ok=True)
        (workspace_path / MODELS_DIR).mkdir(exist_ok=True)
        (workspace_path / EMBEDDINGS_DIR).mkdir(exist_ok=True)
        (workspace_path / EXPORTS_DIR).mkdir(exist_ok=True)

        # Create metadata
        metadata = WorkspaceMetadata(name=name, description=description, tags=tags or [])
        with open(workspace_path / METADATA_FILE, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Workspace '{name}' created at {workspace_path}")
        return cls(workspace_path) # Initialize and return instance

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Workspace":
        """Load an existing workspace."""
        workspace_path = Path(path).resolve()
        if not (workspace_path / METADATA_FILE).exists():
            raise WorkspaceError(f"Workspace metadata.json not found at {workspace_path}")
        return cls(workspace_path)

    def create_dataset(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dataset:
        """Create a new dataset within the workspace."""
        dataset_path = self.path / DATASETS_DIR / name
        if dataset_path.exists():
            raise DatasetError(f"Dataset '{name}' already exists at {dataset_path}")

        dataset = Dataset(path=dataset_path, name=name, workspace=self)
        
        # Update dataset metadata with description and tags
        dataset_metadata_content = {
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tags": tags or [],
            "schema": {}
        }
        dataset._save_metadata(dataset_metadata_content)

        # Register dataset in workspace DB
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO registered_datasets (name, path, description, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (name, str(dataset_path), description, json.dumps(tags or []), dataset_metadata_content["created_at"], dataset_metadata_content["updated_at"])
            )
            conn.commit()
            logger.info(f"Dataset '{name}' created and registered in workspace '{self.name}'.")
        except sqlite3.IntegrityError:
            conn.rollback()
            # This might happen if there's a race condition or inconsistent state
            logger.warning(f"Dataset '{name}' might already be registered or path conflict.")
            # Sill return the dataset object as the directory was created
        finally:
            conn.close()
        return dataset

    def get_dataset(self, name: str) -> Dataset:
        """Get an existing dataset from the workspace."""
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM registered_datasets WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            # Fallback: check if directory exists even if not in DB (e.g. imported manually)
            # This part makes it more resilient but could also indicate an issue if DB is out of sync
            potential_path = self.path / DATASETS_DIR / name
            if potential_path.exists() and (potential_path / METADATA_FILE).exists(): # check for metadata too
                 logger.warning(f"Dataset '{name}' found on disk but not in workspace DB. Consider re-registering.")
                 return Dataset(path=potential_path, name=name, workspace=self)
            raise WorkspaceError(f"Dataset '{name}' not found in workspace '{self.name}'.")
        
        dataset_path = Path(row["path"])
        return Dataset(path=dataset_path, name=name, workspace=self)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in the workspace."""
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name, description, tags, created_at, updated_at FROM registered_datasets ORDER BY name")
        datasets_info = [dict(row) for row in cursor.fetchall()]
        conn.close()
        # Parse tags from JSON string
        for ds_info in datasets_info:
            if ds_info.get("tags"): 
                try:
                    ds_info["tags"] = json.loads(ds_info["tags"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse tags for dataset {ds_info['name']}. Tags: {ds_info['tags']}")
                    ds_info["tags"] = [] # Default to empty list on error
            else:
                ds_info["tags"] = []

        return datasets_info

    def remove_dataset(self, name: str, confirm: bool = False) -> None:
        """Remove a dataset from the workspace (files and registration)."""
        dataset_path = self.path / DATASETS_DIR / name
        if not dataset_path.exists():
            logger.warning(f"Dataset '{name}' directory not found at {dataset_path}. Attempting to unregister.")
        elif confirm:
            shutil.rmtree(dataset_path)
            logger.info(f"Removed dataset files for '{name}' from {dataset_path}")
        else:
            raise WorkspaceError(f"Confirmation not given to delete dataset files for '{name}'. Set confirm=True.")

        # Unregister from workspace DB
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM registered_datasets WHERE name = ?", (name,))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Dataset '{name}' unregistered from workspace '{self.name}'.")
            else:
                logger.warning(f"Dataset '{name}' was not found in the workspace registry.")
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error unregistering dataset '{name}': {e}")
        finally:
            conn.close()

    def close(self) -> None:
        """Close any open resources (e.g., database connections). Currently a placeholder."""
        # In a more complex scenario, this might close a persistent DB connection pool
        logger.debug(f"Workspace '{self.name}' closed.")

    def get_datasette_metadata(self) -> Dict[str, Any]:
        """Generate metadata for Datasette."""
        # This is a basic example. You might want to customize titles, descriptions, etc.
        # based on workspace and dataset metadata.
        metadata = {
            "title": f"OpenSynthetics Workspace: {self.name}",
            "description": self.metadata.description if self._metadata else "Data explorer for OpenSynthetics workspace",
            "databases": {}
        }
        for ds_info in self.list_datasets():
            dataset_name = ds_info["name"]
            dataset_instance = self.get_dataset(dataset_name)
            db_file_path = dataset_instance.db_path
            metadata["databases"][dataset_name] = {
                "title": f"Dataset: {dataset_name}",
                "description": dataset_instance.description or f"Data for {dataset_name}",
                "file": str(db_file_path.relative_to(self.path)) # Path relative to where datasette will serve from
            }
        return metadata

    def serve_datasette(self, host: str = "localhost", port: int = 8001, open_browser: bool = True):
        """Serve the workspace datasets using Datasette."""
        import subprocess
        import webbrowser

        config = Config.load()
        # datasette_port_from_config = config.storage.datasette_port (if using pydantic config)
        # For now, use function arg or default
        actual_port = port

        # Prepare file list for datasette
        db_files_to_serve = []
        for ds_info in self.list_datasets():
            dataset = self.get_dataset(ds_info["name"])
            db_files_to_serve.append(str(dataset.db_path))
        
        if not db_files_to_serve:
            logger.warning("No datasets found to serve with Datasette.")
            return

        # Generate a datasette metadata file for this specific launch
        datasette_metadata_content = self.get_datasette_metadata()
        # Adjust file paths in datasette metadata to be just the filename for serving
        # as Datasette will be run with CWD as the workspace path or serve files from their absolute paths
        # The `datasette serve /path/to/db1.db /path/to/db2.db --metadata metadata.json` command
        # typically expects db names in metadata.json to match the file stems.
        
        # Correcting paths for datasette metadata for `datasette serve file1.db file2.db ... -m meta.json`
        # where meta.json refers to `file1` and `file2` as keys in `databases`.
        corrected_databases_meta = {}
        for ds_name_key, ds_meta_val in datasette_metadata_content.get("databases", {}).items():
            db_path = Path(ds_meta_val["file"]) # This was relative to workspace path
            # Datasette `serve file1.db file2.db` uses `file1`, `file2` as DB names
            # So, the keys in metadata.json's `databases` object should match these.            
            corrected_databases_meta[db_path.stem] = ds_meta_val
            # The 'file' entry in ds_meta_val is already relative to workspace_path, which is good.
        datasette_metadata_content["databases"] = corrected_databases_meta

        datasette_metadata_file = self.path / "datasette-metadata.json"
        with open(datasette_metadata_file, "w") as f:
            json.dump(datasette_metadata_content, f, indent=2)

        cmd = [
            "datasette", "serve",
            *db_files_to_serve, # List of DB file paths
            "--host", host,
            "--port", str(actual_port),
            "--cors", # Enable CORS for API access
            "--metadata", str(datasette_metadata_file),
            # "--root", str(self.path) # Serve from workspace root, not strictly needed if files are absolute paths
        ]
        
        logger.info(f"Starting Datasette server for workspace '{self.name}' at http://{host}:{actual_port}")
        logger.info(f"Serving databases: {', '.join(db_files_to_serve)}")
        logger.info(f"Using metadata: {datasette_metadata_file}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run Datasette as a subprocess
            # For development, you might want to see its output directly.
            # For production, consider redirecting stdout/stderr.
            process = subprocess.Popen(cmd, cwd=self.path) # Run from workspace dir for relative paths if any
            if open_browser:
                import time
                time.sleep(1) # Give server a moment to start
                webbrowser.open(f"http://{host}:{actual_port}")
            process.wait() # Keep this script running until Datasette is closed
        except FileNotFoundError:
            logger.error("Datasette command not found. Please ensure Datasette is installed and in your PATH.")
        except Exception as e:
            logger.error(f"Failed to start Datasette: {e}")
        finally:
            if datasette_metadata_file.exists():
                # Clean up the temporary metadata file, or leave it if it's generally useful
                # os.remove(datasette_metadata_file)
                pass 

# Example Usage (for testing this file directly)
if __name__ == "__main__":
    # Ensure a config file exists for the global Config.load() to work as expected
    # You might need to manually create ~/.opensynthetics/opensynthetics_config.json or set OPENSYNTHETICS_CONFIG_PATH
    
    temp_workspace_dir = Path(tempfile.mkdtemp()) / "test_workspace_main"
    print(f"Temporary workspace will be at: {temp_workspace_dir}")

    try:
        # Create workspace
        ws = Workspace.create(name="my_test_ws", path=temp_workspace_dir, description="A test workspace for main execution")
        print(f"Workspace '{ws.name}' created at {ws.path}")
        print(f"Workspace metadata: {ws.metadata.to_dict()}")

        # Create a dataset
        ds1 = ws.create_dataset(name="sample_data", description="Sample dataset with items", tags=["testdata", "items"])
        print(f"Dataset '{ds1.name}' created. Path: {ds1.path}")

        # Add data
        items_data = [
            {"id": "1", "item_name": "Laptop", "price": 1200.00, "category": "Electronics"},
            {"id": "2", "item_name": "Mouse", "price": 25.00, "category": "Electronics"},
            {"id": "3", "item_name": "Book", "price": 15.75, "category": "Books"},
        ]
        ds1.add_data(items_data, table="products")

        items_data_2 = [
            {"user_id": "u1", "username": "alice"},
            {"user_id": "u2", "username": "bob"},
        ]
        ds1.add_data(items_data_2, table="users")

        # Query data
        print("\nQuerying products table:")
        results = ds1.query("SELECT * FROM products WHERE category = ?", ("Electronics",))
        for row in results:
            print(dict(row))
        
        print("\nAll tables in dataset:", ds1.get_tables())
        print("\nDataset stats:", json.dumps(ds1.get_stats(), indent=2))

        # List datasets in workspace
        print("\nDatasets in workspace:", ws.list_datasets())

        # Get another dataset instance
        retrieved_ds1 = ws.get_dataset("sample_data")
        print(f"\nRetrieved dataset '{retrieved_ds1.name}' description: {retrieved_ds1.description}")

        # Serve with Datasette (requires Datasette to be installed)
        # print("\nAttempting to serve with Datasette...")
        # ws.serve_datasette(open_browser=False) # Set to True to open browser

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary workspace directory
        if temp_workspace_dir.exists():
            print(f"Cleaning up temporary workspace: {temp_workspace_dir}")
            # shutil.rmtree(temp_workspace_dir)
            print(f"Please manually remove {temp_workspace_dir} if you ran this example.")
            #pass # Comment out rmtree for inspection