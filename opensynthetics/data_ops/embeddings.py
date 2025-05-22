"""Embedding generation and storage for OpenSynthetics."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from loguru import logger

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import OpenSyntheticsError
from opensynthetics.core.workspace import Workspace
from opensynthetics.core.mlx_utils import MLXAccelerator


class EmbeddingStore:
    """Storage for embeddings with vector search capabilities."""

    def __init__(self, workspace: Workspace) -> None:
        """Initialize embedding store.

        Args:
            workspace: Workspace to use
        """
        self.workspace = workspace
        self.config = Config.load()
        self.embedding_dir = workspace.path / "embeddings"
        self.embedding_dir.mkdir(exist_ok=True)
        self.mlx = MLXAccelerator(self.config)
        
        # Initialize SQLite database for embedding metadata
        self.db_path = self.embedding_dir / "embeddings.db"
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON embeddings(source_type, source_id)")
        
        conn.commit()
        conn.close()
        
    def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """Generate an embedding for text.

        Args:
            text: Text to embed
            model: Model to use for embedding

        Returns:
            np.ndarray: Embedding vector
        """
        from opensynthetics.llm_core.providers import ProviderFactory
        
        model = model or "text-embedding-3-small"
        provider = ProviderFactory.get_provider("openai", self.config)
        
        try:
            response = provider.client.embeddings.create(
                input=text,
                model=model,
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise OpenSyntheticsError(f"Failed to generate embedding: {e}")
    
    def store_embedding(
        self,
        embedding: np.ndarray,
        source_type: str,
        source_id: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an embedding.

        Args:
            embedding: Embedding vector
            source_type: Type of source (e.g., "document", "query")
            source_id: ID of source
            model: Model used to generate embedding
            metadata: Additional metadata

        Returns:
            str: Embedding ID
        """
        import uuid
        from datetime import datetime
        
        embedding_id = str(uuid.uuid4())
        
        # Save embedding vector
        np.save(
            self.embedding_dir / f"{embedding_id}.npy",
            embedding,
        )
        
        # Store metadata
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO embeddings
            (id, source_type, source_id, model, dimensions, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                source_type,
                source_id,
                model,
                len(embedding),
                datetime.now().isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        
        conn.commit()
        conn.close()
        
        return embedding_id
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            source_type: Filter by source type

        Returns:
            List[Dict[str, Any]]: Search results
        """
        # Get all embeddings matching the filter
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if source_type:
            cursor.execute(
                "SELECT * FROM embeddings WHERE source_type = ?",
                (source_type,),
            )
        else:
            cursor.execute("SELECT * FROM embeddings")
            
        results = []
        
        for row in cursor.fetchall():
            embedding_id = row["id"]
            try:
                # Load embedding
                embedding = np.load(self.embedding_dir / f"{embedding_id}.npy")
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                results.append({
                    "id": embedding_id,
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "similarity": float(similarity),
                    "metadata": json.loads(row["metadata"]),
                })
            except Exception as e:
                logger.error(f"Error processing embedding {embedding_id}: {e}")
        
        conn.close()
        
        # Sort by similarity (highest first) and limit results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            float: Cosine similarity
        """
        if self.mlx.is_available():
            # Use MLX if available
            a_mx = self.mlx.array(a)
            b_mx = self.mlx.array(b)
            
            dot_product = self.mlx.mx.sum(a_mx * b_mx)
            norm_a = self.mlx.mx.sqrt(self.mlx.mx.sum(a_mx * a_mx))
            norm_b = self.mlx.mx.sqrt(self.mlx.mx.sum(b_mx * b_mx))
            
            similarity = dot_product / (norm_a * norm_b)
            return float(self.mlx.to_numpy(similarity))
        else:
            # Fall back to numpy
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            return dot_product / (norm_a * norm_b)