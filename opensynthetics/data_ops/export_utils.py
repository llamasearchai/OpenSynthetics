"""Export utilities for OpenSynthetics with comprehensive format support."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
import gzip
import zipfile
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("PyArrow not available. Parquet export will be disabled.")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py not available. HDF5 export will be disabled.")

from opensynthetics.core.exceptions import ProcessingError


class ExportConfig(BaseModel):
    """Configuration for data export."""
    
    format: Literal["json", "jsonl", "csv", "parquet", "hdf5", "excel", "feather"] = Field(
        "json", description="Export format"
    )
    compression: Optional[Literal["gzip", "bz2", "zip", "snappy", "lz4"]] = Field(
        None, description="Compression type"
    )
    chunk_size: int = Field(10000, description="Chunk size for large datasets", ge=1)
    include_metadata: bool = Field(True, description="Include metadata in export")
    preserve_dtypes: bool = Field(True, description="Preserve data types in export")
    nested_json_as_string: bool = Field(False, description="Convert nested JSON to string")
    decimal_precision: int = Field(6, description="Decimal precision for floats", ge=0, le=15)
    
    # Parquet specific options
    parquet_engine: Literal["pyarrow", "fastparquet"] = Field("pyarrow", description="Parquet engine")
    parquet_compression: Optional[Literal["snappy", "gzip", "brotli", "lz4", "zstd"]] = Field(
        "snappy", description="Parquet compression algorithm"
    )
    row_group_size: int = Field(50000, description="Parquet row group size", ge=1000)
    
    # CSV specific options
    csv_separator: str = Field(",", description="CSV separator")
    csv_quoting: Literal["minimal", "all", "nonnumeric", "none"] = Field(
        "minimal", description="CSV quoting strategy"
    )
    csv_encoding: str = Field("utf-8", description="CSV encoding")
    
    # Quality options
    validate_schema: bool = Field(True, description="Validate data schema before export")
    create_checksums: bool = Field(True, description="Create file checksums")
    split_large_files: bool = Field(True, description="Split large files automatically")
    max_file_size_mb: int = Field(500, description="Maximum file size in MB", ge=10)


class ExportMetadata(BaseModel):
    """Metadata for exported datasets."""
    
    export_timestamp: datetime = Field(default_factory=datetime.now)
    total_records: int = Field(0, description="Total number of records")
    file_count: int = Field(1, description="Number of output files")
    total_size_bytes: int = Field(0, description="Total size in bytes")
    schema_info: Dict[str, Any] = Field(default_factory=dict)
    compression_ratio: Optional[float] = Field(None, description="Compression ratio if applicable")
    export_config: Dict[str, Any] = Field(default_factory=dict)
    checksums: Dict[str, str] = Field(default_factory=dict)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)


class DataExporter:
    """Comprehensive data exporter with multiple format support."""
    
    def __init__(self, config: ExportConfig = None) -> None:
        """Initialize the data exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        
    def export_dataset(
        self,
        data: Union[List[Dict[str, Any]], pd.DataFrame],
        output_path: Union[str, Path],
        config: Optional[ExportConfig] = None
    ) -> ExportMetadata:
        """Export dataset to specified format.
        
        Args:
            data: Data to export
            output_path: Output file path
            config: Optional export configuration override
            
        Returns:
            ExportMetadata with export information
            
        Raises:
            ProcessingError: If export fails
        """
        export_config = config or self.config
        output_path = Path(output_path)
        
        logger.info(f"Exporting dataset to {output_path} in {export_config.format} format")
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = self._list_to_dataframe(data, export_config)
            else:
                df = data.copy()
            
            # Validate schema if requested
            if export_config.validate_schema:
                self._validate_schema(df)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            export_metadata = self._export_by_format(df, output_path, export_config)
            
            # Add configuration to metadata
            export_metadata.export_config = export_config.model_dump()
            
            # Calculate quality metrics
            if export_config.include_metadata:
                export_metadata.quality_metrics = self._calculate_quality_metrics(df)
            
            # Create checksums if requested
            if export_config.create_checksums:
                export_metadata.checksums = self._create_checksums(output_path)
            
            logger.info(f"Export completed: {export_metadata.total_records} records, {export_metadata.file_count} files")
            return export_metadata
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ProcessingError(f"Failed to export dataset: {e}")
    
    def _list_to_dataframe(self, data: List[Dict[str, Any]], config: ExportConfig) -> pd.DataFrame:
        """Convert list of dictionaries to DataFrame."""
        if not data:
            return pd.DataFrame()
        
        # Handle nested JSON if requested
        if config.nested_json_as_string:
            processed_data = []
            for item in data:
                processed_item = {}
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        processed_item[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        processed_item[key] = value
                processed_data.append(processed_item)
            data = processed_data
        
        df = pd.DataFrame(data)
        
        # Preserve data types if requested
        if config.preserve_dtypes:
            df = self._optimize_dtypes(df)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage efficiency."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    # Try to convert to datetime
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        # Keep as string but optimize
                        if df[col].nunique() / len(df) < 0.5:
                            df[col] = df[col].astype('category')
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            elif col_type in ['float64', 'float32']:
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame schema."""
        if df.empty:
            raise ProcessingError("Cannot export empty dataset")
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            raise ProcessingError("Dataset contains duplicate column names")
        
        # Check for very large objects that might cause issues
        for col in df.select_dtypes(include=['object']).columns:
            max_size = df[col].astype(str).str.len().max()
            if max_size > 1000000:  # 1MB per cell
                logger.warning(f"Column '{col}' contains very large values (max: {max_size} chars)")
    
    def _export_by_format(
        self, 
        df: pd.DataFrame, 
        output_path: Path, 
        config: ExportConfig
    ) -> ExportMetadata:
        """Export DataFrame based on specified format."""
        
        start_time = time.time()
        total_records = len(df)
        
        if config.format == "json":
            metadata = self._export_json(df, output_path, config)
        elif config.format == "jsonl":
            metadata = self._export_jsonl(df, output_path, config)
        elif config.format == "csv":
            metadata = self._export_csv(df, output_path, config)
        elif config.format == "parquet":
            metadata = self._export_parquet(df, output_path, config)
        elif config.format == "hdf5":
            metadata = self._export_hdf5(df, output_path, config)
        elif config.format == "excel":
            metadata = self._export_excel(df, output_path, config)
        elif config.format == "feather":
            metadata = self._export_feather(df, output_path, config)
        else:
            raise ProcessingError(f"Unsupported format: {config.format}")
        
        metadata.total_records = total_records
        metadata.export_timestamp = datetime.now()
        
        return metadata
    
    def _export_json(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to JSON format."""
        output_files = []
        
        if config.split_large_files and len(df) > config.chunk_size:
            # Split into multiple files
            for i, chunk in enumerate(np.array_split(df, len(df) // config.chunk_size + 1)):
                chunk_path = output_path.with_stem(f"{output_path.stem}_part_{i+1}")
                actual_path = self._write_json_chunk(chunk, chunk_path, config)
                output_files.append(actual_path)
        else:
            actual_path = self._write_json_chunk(df, output_path, config)
            output_files.append(actual_path)
        
        total_size = sum(f.stat().st_size for f in output_files)
        
        return ExportMetadata(
            file_count=len(output_files),
            total_size_bytes=total_size,
            schema_info=self._get_schema_info(df)
        )
    
    def _write_json_chunk(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> Path:
        """Write a DataFrame chunk to JSON."""
        data = df.to_dict('records')
        
        if config.compression == "gzip":
            actual_path = Path(f"{output_path}.gz")
            with gzip.open(actual_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return actual_path
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return output_path
    
    def _export_jsonl(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to JSONL format."""
        output_files = []
        
        if config.split_large_files and len(df) > config.chunk_size:
            # Split into multiple files
            for i, chunk in enumerate(np.array_split(df, len(df) // config.chunk_size + 1)):
                chunk_path = output_path.with_stem(f"{output_path.stem}_part_{i+1}")
                actual_path = self._write_jsonl_chunk(chunk, chunk_path, config)
                output_files.append(actual_path)
        else:
            actual_path = self._write_jsonl_chunk(df, output_path, config)
            output_files.append(actual_path)
        
        total_size = sum(f.stat().st_size for f in output_files)
        
        return ExportMetadata(
            file_count=len(output_files),
            total_size_bytes=total_size,
            schema_info=self._get_schema_info(df)
        )
    
    def _write_jsonl_chunk(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> Path:
        """Write a DataFrame chunk to JSONL."""
        if config.compression == "gzip":
            actual_path = Path(f"{output_path}.gz")
            with gzip.open(actual_path, 'wt', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False, default=str)
                    f.write('\n')
            return actual_path
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False, default=str)
                    f.write('\n')
            return output_path
    
    def _export_csv(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to CSV format."""
        csv_params = {
            'sep': config.csv_separator,
            'encoding': config.csv_encoding,
            'index': False,
            'float_format': f'%.{config.decimal_precision}f'
        }
        
        if config.csv_quoting == "all":
            csv_params['quoting'] = 1  # QUOTE_ALL
        elif config.csv_quoting == "nonnumeric":
            csv_params['quoting'] = 2  # QUOTE_NONNUMERIC
        elif config.csv_quoting == "none":
            csv_params['quoting'] = 3  # QUOTE_NONE
        
        if config.compression:
            csv_params['compression'] = config.compression
        
        df.to_csv(output_path, **csv_params)
        
        return ExportMetadata(
            file_count=1,
            total_size_bytes=output_path.stat().st_size,
            schema_info=self._get_schema_info(df)
        )
    
    def _export_parquet(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to Parquet format."""
        if not PARQUET_AVAILABLE:
            raise ProcessingError("PyArrow not available for Parquet export")
        
        parquet_params = {
            'compression': config.parquet_compression,
            'row_group_size': config.row_group_size,
        }
        
        # Convert DataFrame to Arrow table for better control
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        # Add metadata
        if config.include_metadata:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'total_records': str(len(df)),
                'opensynthetics_version': '0.1.0'
            }
            
            # Update table metadata
            existing_metadata = table.schema.metadata or {}
            existing_metadata.update({k.encode(): v.encode() for k, v in metadata.items()})
            table = table.replace_schema_metadata(existing_metadata)
        
        # Write parquet file
        pq.write_table(table, output_path, **parquet_params)
        
        # Calculate compression ratio
        uncompressed_size = df.memory_usage(deep=True).sum()
        compressed_size = output_path.stat().st_size
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
        
        return ExportMetadata(
            file_count=1,
            total_size_bytes=compressed_size,
            schema_info=self._get_schema_info(df),
            compression_ratio=compression_ratio
        )
    
    def _export_hdf5(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to HDF5 format."""
        if not HDF5_AVAILABLE:
            raise ProcessingError("h5py not available for HDF5 export")
        
        # Use pandas HDF5 support
        store = pd.HDFStore(str(output_path), mode='w', complevel=9, complib='zlib')
        
        try:
            store.put('data', df, format='table', data_columns=True)
            
            # Add metadata
            if config.include_metadata:
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_records': len(df),
                    'opensynthetics_version': '0.1.0'
                }
                store.get_storer('data').attrs.metadata = json.dumps(metadata)
            
        finally:
            store.close()
        
        return ExportMetadata(
            file_count=1,
            total_size_bytes=output_path.stat().st_size,
            schema_info=self._get_schema_info(df)
        )
    
    def _export_excel(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to Excel format."""
        excel_params = {
            'index': False,
            'engine': 'openpyxl' if output_path.suffix == '.xlsx' else 'xlwt'
        }
        
        with pd.ExcelWriter(output_path, **excel_params) as writer:
            df.to_excel(writer, sheet_name='data', **excel_params)
            
            # Add metadata sheet if requested
            if config.include_metadata:
                metadata_df = pd.DataFrame([
                    ['Export Timestamp', datetime.now().isoformat()],
                    ['Total Records', len(df)],
                    ['OpenSynthetics Version', '0.1.0']
                ], columns=['Key', 'Value'])
                
                metadata_df.to_excel(writer, sheet_name='metadata', index=False)
        
        return ExportMetadata(
            file_count=1,
            total_size_bytes=output_path.stat().st_size,
            schema_info=self._get_schema_info(df)
        )
    
    def _export_feather(self, df: pd.DataFrame, output_path: Path, config: ExportConfig) -> ExportMetadata:
        """Export to Feather format."""
        if not PARQUET_AVAILABLE:
            raise ProcessingError("PyArrow not available for Feather export")
        
        feather_params = {
            'compression': config.compression or 'uncompressed'
        }
        
        df.to_feather(output_path, **feather_params)
        
        return ExportMetadata(
            file_count=1,
            total_size_bytes=output_path.stat().st_size,
            schema_info=self._get_schema_info(df)
        )
    
    def _get_schema_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get schema information for the DataFrame."""
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        
        return schema_info
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset."""
        metrics = {
            'completeness': {
                'overall': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
                'by_column': {col: 1 - (df[col].isnull().sum() / len(df)) for col in df.columns}
            },
            'uniqueness': {
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_ratio': df.duplicated().sum() / len(df),
                'unique_ratios': {col: df[col].nunique() / len(df) for col in df.columns}
            },
            'consistency': {
                'dtype_consistency': all(df[col].dtype == df[col].infer_objects().dtype for col in df.columns),
                'encoding_issues': sum(1 for col in df.select_dtypes(include=['object']).columns 
                                     if df[col].astype(str).str.contains(r'[^\x00-\x7F]').any())
            }
        }
        
        # Add statistical summaries for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metrics['statistics'] = {
                'numeric_summary': df[numeric_cols].describe().to_dict(),
                'correlations': df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}
            }
        
        return metrics
    
    def _create_checksums(self, output_path: Path) -> Dict[str, str]:
        """Create checksums for exported files."""
        import hashlib
        
        checksums = {}
        
        if output_path.is_file():
            files_to_check = [output_path]
        else:
            files_to_check = list(output_path.glob('*'))
        
        for file_path in files_to_check:
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    content = f.read()
                    checksums[str(file_path)] = {
                        'md5': hashlib.md5(content).hexdigest(),
                        'sha256': hashlib.sha256(content).hexdigest()
                    }
        
        return checksums


class BatchExporter:
    """Batch exporter for handling multiple datasets."""
    
    def __init__(self) -> None:
        """Initialize batch exporter."""
        self.exporter = DataExporter()
    
    def export_multiple(
        self,
        datasets: Dict[str, Union[List[Dict[str, Any]], pd.DataFrame]],
        output_dir: Union[str, Path],
        configs: Optional[Dict[str, ExportConfig]] = None,
        default_config: Optional[ExportConfig] = None
    ) -> Dict[str, ExportMetadata]:
        """Export multiple datasets.
        
        Args:
            datasets: Dictionary of dataset name -> data
            output_dir: Output directory
            configs: Optional per-dataset configurations
            default_config: Default configuration for all datasets
            
        Returns:
            Dictionary of dataset name -> export metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = configs or {}
        default_config = default_config or ExportConfig()
        
        results = {}
        
        for name, data in datasets.items():
            try:
                config = configs.get(name, default_config)
                output_path = output_dir / f"{name}.{config.format}"
                
                logger.info(f"Exporting dataset: {name}")
                metadata = self.exporter.export_dataset(data, output_path, config)
                results[name] = metadata
                
            except Exception as e:
                logger.error(f"Failed to export dataset {name}: {e}")
                results[name] = None
        
        # Create summary report
        self._create_summary_report(results, output_dir)
        
        return results
    
    def _create_summary_report(
        self, 
        results: Dict[str, Optional[ExportMetadata]], 
        output_dir: Path
    ) -> None:
        """Create a summary report of all exports."""
        summary = {
            'export_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_datasets': len(results),
                'successful_exports': sum(1 for r in results.values() if r is not None),
                'failed_exports': sum(1 for r in results.values() if r is None),
                'total_records': sum(r.total_records for r in results.values() if r is not None),
                'total_size_bytes': sum(r.total_size_bytes for r in results.values() if r is not None)
            },
            'dataset_details': {}
        }
        
        for name, metadata in results.items():
            if metadata:
                summary['dataset_details'][name] = {
                    'records': metadata.total_records,
                    'size_bytes': metadata.total_size_bytes,
                    'files': metadata.file_count,
                    'compression_ratio': metadata.compression_ratio
                }
            else:
                summary['dataset_details'][name] = {'status': 'failed'}
        
        # Save summary
        summary_path = output_dir / 'export_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Export summary saved to: {summary_path}") 