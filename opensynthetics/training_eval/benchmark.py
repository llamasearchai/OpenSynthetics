"""Comprehensive benchmarking module for synthetic dataset evaluation."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Visualization features will be disabled.")

try:
    from scipy.stats import ks_2samp, wasserstein_distance
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False
    logger.warning("SciPy stats not available. Some statistical tests will be disabled.")

from opensynthetics.core.exceptions import EvaluationError


class DatasetQualityMetrics(BaseModel):
    """Metrics for dataset quality assessment."""
    
    # Data integrity metrics
    completeness_score: float = Field(..., description="Data completeness ratio (0-1)", ge=0.0, le=1.0)
    consistency_score: float = Field(..., description="Data consistency score (0-1)", ge=0.0, le=1.0)
    uniqueness_score: float = Field(..., description="Data uniqueness score (0-1)", ge=0.0, le=1.0)
    validity_score: float = Field(..., description="Data validity score (0-1)", ge=0.0, le=1.0)
    
    # Statistical quality metrics
    distribution_similarity: Dict[str, float] = Field(default_factory=dict, description="Distribution similarity scores")
    correlation_preservation: float = Field(0.0, description="Correlation structure preservation", ge=0.0, le=1.0)
    statistical_fidelity: float = Field(0.0, description="Overall statistical fidelity", ge=0.0, le=1.0)
    
    # Machine learning utility metrics
    ml_utility_score: float = Field(0.0, description="ML utility score", ge=0.0, le=1.0)
    predictive_performance: Dict[str, float] = Field(default_factory=dict, description="Predictive performance metrics")
    
    # Privacy and security metrics
    privacy_score: float = Field(0.0, description="Privacy preservation score", ge=0.0, le=1.0)
    disclosure_risk: float = Field(0.0, description="Re-identification risk", ge=0.0, le=1.0)
    
    # Overall quality score
    overall_quality_score: float = Field(0.0, description="Overall quality score", ge=0.0, le=1.0)


class BenchmarkConfig(BaseModel):
    """Configuration for dataset benchmarking."""
    
    # Quality assessment options
    assess_completeness: bool = Field(True, description="Assess data completeness")
    assess_consistency: bool = Field(True, description="Assess data consistency")
    assess_uniqueness: bool = Field(True, description="Assess data uniqueness")
    assess_validity: bool = Field(True, description="Assess data validity")
    
    # Statistical analysis options
    test_distributions: bool = Field(True, description="Test distribution similarity")
    test_correlations: bool = Field(True, description="Test correlation preservation")
    statistical_tests: List[str] = Field(
        default_factory=lambda: ["ks_test", "chi_square", "wasserstein"],
        description="Statistical tests to perform"
    )
    
    # Machine learning evaluation
    ml_evaluation: bool = Field(True, description="Perform ML utility evaluation")
    ml_tasks: List[str] = Field(
        default_factory=lambda: ["classification", "regression"],
        description="ML tasks to evaluate"
    )
    test_size: float = Field(0.2, description="Test set size for ML evaluation", gt=0.0, lt=1.0)
    
    # Privacy assessment
    privacy_evaluation: bool = Field(False, description="Perform privacy evaluation")
    k_anonymity_check: bool = Field(False, description="Check k-anonymity")
    l_diversity_check: bool = Field(False, description="Check l-diversity")
    
    # Output options
    generate_report: bool = Field(True, description="Generate comprehensive report")
    create_visualizations: bool = Field(True, description="Create visualization plots")
    save_detailed_results: bool = Field(True, description="Save detailed results")
    
    # Performance options
    sample_size: Optional[int] = Field(None, description="Sample size for large datasets")
    parallel_processing: bool = Field(True, description="Use parallel processing")
    verbose: bool = Field(True, description="Verbose output")


class SyntheticDatasetBenchmark:
    """Comprehensive benchmark for synthetic dataset evaluation."""
    
    def __init__(self, config: BenchmarkConfig = None) -> None:
        """Initialize the benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results = {}
        
    def benchmark_dataset(
        self,
        synthetic_data: Union[pd.DataFrame, List[Dict[str, Any]]],
        reference_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        target_column: Optional[str] = None
    ) -> DatasetQualityMetrics:
        """Benchmark a synthetic dataset.
        
        Args:
            synthetic_data: Synthetic dataset to evaluate
            reference_data: Optional reference dataset for comparison
            target_column: Target column for ML evaluation
            
        Returns:
            DatasetQualityMetrics with comprehensive evaluation results
            
        Raises:
            EvaluationError: If benchmark evaluation fails
        """
        logger.info("Starting comprehensive dataset benchmarking")
        
        try:
            # Convert to DataFrame if needed
            if isinstance(synthetic_data, list):
                synthetic_df = pd.DataFrame(synthetic_data)
            else:
                synthetic_df = synthetic_data.copy()
            
            if reference_data is not None:
                if isinstance(reference_data, list):
                    reference_df = pd.DataFrame(reference_data)
                else:
                    reference_df = reference_data.copy()
            else:
                reference_df = None
            
            # Sample if dataset is too large
            if self.config.sample_size and len(synthetic_df) > self.config.sample_size:
                synthetic_df = synthetic_df.sample(n=self.config.sample_size, random_state=42)
                if reference_df is not None and len(reference_df) > self.config.sample_size:
                    reference_df = reference_df.sample(n=self.config.sample_size, random_state=42)
            
            # Initialize metrics
            metrics = DatasetQualityMetrics(
                completeness_score=0.0,
                consistency_score=0.0,
                uniqueness_score=0.0,
                validity_score=0.0
            )
            
            # Data quality assessment
            if self.config.assess_completeness:
                metrics.completeness_score = self._assess_completeness(synthetic_df)
            
            if self.config.assess_consistency:
                metrics.consistency_score = self._assess_consistency(synthetic_df)
            
            if self.config.assess_uniqueness:
                metrics.uniqueness_score = self._assess_uniqueness(synthetic_df)
            
            if self.config.assess_validity:
                metrics.validity_score = self._assess_validity(synthetic_df)
            
            # Statistical analysis
            if reference_df is not None:
                if self.config.test_distributions:
                    metrics.distribution_similarity = self._test_distributions(synthetic_df, reference_df)
                
                if self.config.test_correlations:
                    metrics.correlation_preservation = self._test_correlations(synthetic_df, reference_df)
                
                metrics.statistical_fidelity = self._calculate_statistical_fidelity(
                    synthetic_df, reference_df
                )
            
            # Machine learning evaluation
            if self.config.ml_evaluation and target_column and target_column in synthetic_df.columns:
                ml_results = self._evaluate_ml_utility(synthetic_df, reference_df, target_column)
                metrics.ml_utility_score = ml_results["utility_score"]
                metrics.predictive_performance = ml_results["performance"]
            
            # Privacy evaluation
            if self.config.privacy_evaluation:
                privacy_results = self._evaluate_privacy(synthetic_df, reference_df)
                metrics.privacy_score = privacy_results["privacy_score"]
                metrics.disclosure_risk = privacy_results["disclosure_risk"]
            
            # Calculate overall quality score
            metrics.overall_quality_score = self._calculate_overall_score(metrics)
            
            # Store detailed results
            self.results = {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "config": self.config.model_dump(),
                "dataset_info": {
                    "synthetic_shape": synthetic_df.shape,
                    "reference_shape": reference_df.shape if reference_df is not None else None,
                    "target_column": target_column
                }
            }
            
            # Generate report if requested
            if self.config.generate_report:
                self._generate_report(metrics, synthetic_df, reference_df)
            
            # Create visualizations if requested
            if self.config.create_visualizations and PLOTTING_AVAILABLE:
                self._create_visualizations(synthetic_df, reference_df)
            
            logger.info(f"Benchmarking completed. Overall quality score: {metrics.overall_quality_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmark evaluation failed: {e}")
            raise EvaluationError(f"Failed to benchmark dataset: {e}")
    
    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        logger.debug(f"Completeness assessment: {completeness:.3f}")
        return completeness
    
    def _assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency."""
        consistency_scores = []
        
        # Check dtype consistency
        for col in df.columns:
            try:
                # Check if column can be consistently parsed
                if df[col].dtype == 'object':
                    # For object columns, check string consistency
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # Check for mixed types (strings vs numbers)
                        str_count = sum(isinstance(x, str) for x in non_null_values)
                        consistency_scores.append(str_count / len(non_null_values))
                else:
                    # For numeric columns, check for consistency
                    consistency_scores.append(1.0)
            except:
                consistency_scores.append(0.5)  # Partial consistency for problematic columns
        
        consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        logger.debug(f"Consistency assessment: {consistency:.3f}")
        return consistency
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> float:
        """Assess data uniqueness."""
        # Calculate uniqueness ratio for each column
        uniqueness_scores = []
        
        for col in df.columns:
            total_count = len(df[col].dropna())
            unique_count = df[col].nunique()
            uniqueness = unique_count / total_count if total_count > 0 else 0.0
            uniqueness_scores.append(uniqueness)
        
        # Also check for duplicate rows
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        row_uniqueness = unique_rows / total_rows if total_rows > 0 else 0.0
        
        # Combine column and row uniqueness
        overall_uniqueness = (np.mean(uniqueness_scores) + row_uniqueness) / 2
        
        logger.debug(f"Uniqueness assessment: {overall_uniqueness:.3f}")
        return overall_uniqueness
    
    def _assess_validity(self, df: pd.DataFrame) -> float:
        """Assess data validity."""
        validity_scores = []
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                validity_scores.append(0.0)
                continue
            
            # Check for valid ranges and formats
            if col_data.dtype in ['int64', 'float64']:
                # Check for infinite or extreme values
                finite_ratio = np.isfinite(col_data).sum() / len(col_data)
                validity_scores.append(finite_ratio)
            elif col_data.dtype == 'object':
                # Check for valid string formats (no control characters)
                valid_strings = col_data.astype(str).str.match(r'^[^\x00-\x1f]*$').sum()
                validity_scores.append(valid_strings / len(col_data))
            else:
                validity_scores.append(1.0)  # Assume valid for other types
        
        validity = np.mean(validity_scores) if validity_scores else 1.0
        
        logger.debug(f"Validity assessment: {validity:.3f}")
        return validity
    
    def _test_distributions(self, synthetic_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, float]:
        """Test distribution similarity between synthetic and reference data."""
        if not SCIPY_STATS_AVAILABLE:
            logger.warning("SciPy not available. Skipping distribution tests.")
            return {}
        
        similarity_scores = {}
        
        for col in synthetic_df.columns:
            if col not in reference_df.columns:
                continue
            
            synthetic_col = synthetic_df[col].dropna()
            reference_col = reference_df[col].dropna()
            
            if len(synthetic_col) == 0 or len(reference_col) == 0:
                continue
            
            try:
                if synthetic_col.dtype in ['int64', 'float64'] and reference_col.dtype in ['int64', 'float64']:
                    # Numerical columns - use KS test and Wasserstein distance
                    ks_stat, ks_p = ks_2samp(synthetic_col, reference_col)
                    wasserstein_dist = wasserstein_distance(synthetic_col, reference_col)
                    
                    # Convert to similarity score (0-1, higher is better)
                    ks_similarity = 1 - ks_stat
                    
                    # Normalize Wasserstein distance
                    range_val = max(reference_col.max() - reference_col.min(), 1e-10)
                    wasserstein_similarity = 1 - min(wasserstein_dist / range_val, 1.0)
                    
                    similarity_scores[col] = (ks_similarity + wasserstein_similarity) / 2
                
                else:
                    # Categorical columns - use chi-square test
                    synthetic_counts = synthetic_col.value_counts()
                    reference_counts = reference_col.value_counts()
                    
                    # Align categories
                    all_categories = set(synthetic_counts.index) | set(reference_counts.index)
                    synthetic_aligned = [synthetic_counts.get(cat, 0) for cat in all_categories]
                    reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                    
                    # Chi-square test
                    if sum(reference_aligned) > 0 and len(all_categories) > 1:
                        chi2_stat, chi2_p = stats.chisquare(synthetic_aligned, reference_aligned)
                        similarity_scores[col] = max(0.0, 1 - (chi2_stat / len(all_categories)))
                    else:
                        similarity_scores[col] = 0.0
            
            except Exception as e:
                logger.warning(f"Distribution test failed for column {col}: {e}")
                similarity_scores[col] = 0.0
        
        logger.debug(f"Distribution similarity scores: {similarity_scores}")
        return similarity_scores
    
    def _test_correlations(self, synthetic_df: pd.DataFrame, reference_df: pd.DataFrame) -> float:
        """Test correlation preservation between datasets."""
        try:
            # Get numeric columns
            synthetic_numeric = synthetic_df.select_dtypes(include=[np.number])
            reference_numeric = reference_df.select_dtypes(include=[np.number])
            
            # Find common columns
            common_cols = list(set(synthetic_numeric.columns) & set(reference_numeric.columns))
            
            if len(common_cols) < 2:
                return 1.0  # No correlations to compare
            
            synthetic_corr = synthetic_numeric[common_cols].corr()
            reference_corr = reference_numeric[common_cols].corr()
            
            # Calculate correlation of correlations
            synthetic_values = synthetic_corr.values[np.triu_indices_from(synthetic_corr.values, k=1)]
            reference_values = reference_corr.values[np.triu_indices_from(reference_corr.values, k=1)]
            
            if len(synthetic_values) > 0:
                correlation_preservation = np.corrcoef(synthetic_values, reference_values)[0, 1]
                correlation_preservation = max(0.0, correlation_preservation)  # Ensure non-negative
            else:
                correlation_preservation = 1.0
            
            logger.debug(f"Correlation preservation: {correlation_preservation:.3f}")
            return correlation_preservation
        
        except Exception as e:
            logger.warning(f"Correlation test failed: {e}")
            return 0.0
    
    def _calculate_statistical_fidelity(self, synthetic_df: pd.DataFrame, reference_df: pd.DataFrame) -> float:
        """Calculate overall statistical fidelity."""
        fidelity_scores = []
        
        # Mean absolute percentage error for means
        for col in synthetic_df.columns:
            if col not in reference_df.columns:
                continue
            
            if synthetic_df[col].dtype in ['int64', 'float64'] and reference_df[col].dtype in ['int64', 'float64']:
                synthetic_mean = synthetic_df[col].mean()
                reference_mean = reference_df[col].mean()
                
                if reference_mean != 0:
                    mape = abs(synthetic_mean - reference_mean) / abs(reference_mean)
                    fidelity_scores.append(max(0.0, 1 - mape))
        
        return np.mean(fidelity_scores) if fidelity_scores else 0.0
    
    def _evaluate_ml_utility(
        self, 
        synthetic_df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame], 
        target_column: str
    ) -> Dict[str, Any]:
        """Evaluate machine learning utility of synthetic data."""
        try:
            # Prepare features and target
            feature_cols = [col for col in synthetic_df.columns if col != target_column]
            X_synthetic = synthetic_df[feature_cols]
            y_synthetic = synthetic_df[target_column]
            
            # Handle categorical variables
            label_encoders = {}
            for col in X_synthetic.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_synthetic.loc[:, col] = le.fit_transform(X_synthetic[col].astype(str))
                label_encoders[col] = le
            
            # Handle target variable
            if y_synthetic.dtype == 'object':
                target_le = LabelEncoder()
                y_synthetic = target_le.fit_transform(y_synthetic.astype(str))
                task_type = 'classification'
            else:
                task_type = 'regression'
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_synthetic, y_synthetic, test_size=self.config.test_size, random_state=42
            )
            
            # Train model
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                performance = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                utility_score = performance['f1']
            
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                
                # Calculate R²
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                performance = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': max(0.0, r2)
                }
                utility_score = performance['r2']
            
            # Compare with reference data if available
            if reference_df is not None and target_column in reference_df.columns:
                # Train on reference, test on synthetic
                X_ref = reference_df[feature_cols]
                y_ref = reference_df[target_column]
                
                # Apply same preprocessing
                for col in X_ref.select_dtypes(include=['object']).columns:
                    if col in label_encoders:
                        try:
                            X_ref[col] = label_encoders[col].transform(X_ref[col].astype(str))
                        except:
                            # Handle unseen categories
                            X_ref[col] = 0
                
                if task_type == 'classification' and 'target_le' in locals():
                    try:
                        y_ref = target_le.transform(y_ref.astype(str))
                    except:
                        # Skip comparison if target categories don't match
                        pass
                
                # This would involve more complex cross-validation
                # For now, we'll use the synthetic data performance as utility score
            
            return {
                "utility_score": utility_score,
                "performance": performance,
                "task_type": task_type
            }
        
        except Exception as e:
            logger.warning(f"ML utility evaluation failed: {e}")
            return {
                "utility_score": 0.0,
                "performance": {},
                "task_type": "unknown"
            }
    
    def _evaluate_privacy(
        self, 
        synthetic_df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Evaluate privacy preservation of synthetic data."""
        # Simplified privacy evaluation
        # In practice, this would involve more sophisticated privacy metrics
        
        privacy_score = 1.0  # Default high privacy score for synthetic data
        disclosure_risk = 0.0  # Default low disclosure risk
        
        if reference_df is not None:
            # Check for exact matches (potential privacy risk)
            common_cols = list(set(synthetic_df.columns) & set(reference_df.columns))
            if common_cols:
                # Sample check for exact row matches
                sample_size = min(1000, len(synthetic_df))
                synthetic_sample = synthetic_df[common_cols].sample(n=sample_size, random_state=42)
                
                matches = 0
                for _, row in synthetic_sample.iterrows():
                    # Check if this exact row exists in reference data
                    match_mask = (reference_df[common_cols] == row).all(axis=1)
                    if match_mask.any():
                        matches += 1
                
                disclosure_risk = matches / sample_size
                privacy_score = 1 - disclosure_risk
        
        logger.debug(f"Privacy evaluation - Score: {privacy_score:.3f}, Risk: {disclosure_risk:.3f}")
        return {
            "privacy_score": privacy_score,
            "disclosure_risk": disclosure_risk
        }
    
    def _calculate_overall_score(self, metrics: DatasetQualityMetrics) -> float:
        """Calculate overall quality score."""
        # Weighted combination of different quality aspects
        weights = {
            'completeness': 0.2,
            'consistency': 0.15,
            'uniqueness': 0.15,
            'validity': 0.15,
            'statistical_fidelity': 0.15,
            'ml_utility': 0.2
        }
        
        scores = [
            metrics.completeness_score * weights['completeness'],
            metrics.consistency_score * weights['consistency'],
            metrics.uniqueness_score * weights['uniqueness'],
            metrics.validity_score * weights['validity'],
            metrics.statistical_fidelity * weights['statistical_fidelity'],
            metrics.ml_utility_score * weights['ml_utility']
        ]
        
        overall_score = sum(scores)
        logger.debug(f"Overall quality score: {overall_score:.3f}")
        return overall_score
    
    def _generate_report(
        self, 
        metrics: DatasetQualityMetrics, 
        synthetic_df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame]
    ) -> None:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_report": {
                "timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "synthetic_records": len(synthetic_df),
                    "synthetic_features": len(synthetic_df.columns),
                    "reference_records": len(reference_df) if reference_df is not None else None,
                    "reference_features": len(reference_df.columns) if reference_df is not None else None
                },
                "quality_metrics": metrics.model_dump(),
                "recommendations": self._generate_recommendations(metrics)
            }
        }
        
        # Save report
        report_path = Path("benchmark_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Benchmark report saved to: {report_path}")
    
    def _generate_recommendations(self, metrics: DatasetQualityMetrics) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if metrics.completeness_score < 0.9:
            recommendations.append("Consider improving data completeness by reducing missing values")
        
        if metrics.consistency_score < 0.8:
            recommendations.append("Review data consistency and standardize data formats")
        
        if metrics.uniqueness_score < 0.7:
            recommendations.append("Address duplicate records and improve data uniqueness")
        
        if metrics.validity_score < 0.9:
            recommendations.append("Validate data formats and fix invalid entries")
        
        if metrics.statistical_fidelity < 0.8:
            recommendations.append("Improve statistical similarity to reference data")
        
        if metrics.ml_utility_score < 0.7:
            recommendations.append("Enhance machine learning utility through better feature representation")
        
        if metrics.overall_quality_score < 0.8:
            recommendations.append("Overall data quality needs improvement across multiple dimensions")
        
        return recommendations
    
    def _create_visualizations(
        self, 
        synthetic_df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame]
    ) -> None:
        """Create visualization plots for benchmark results."""
        if not PLOTTING_AVAILABLE:
            return
        
        try:
            # Create output directory
            viz_dir = Path("benchmark_visualizations")
            viz_dir.mkdir(exist_ok=True)
            
            # Quality metrics radar chart
            self._create_quality_radar_chart(viz_dir)
            
            # Distribution comparison plots
            if reference_df is not None:
                self._create_distribution_plots(synthetic_df, reference_df, viz_dir)
            
            # Correlation heatmaps
            self._create_correlation_plots(synthetic_df, reference_df, viz_dir)
            
            logger.info(f"Visualizations saved to: {viz_dir}")
        
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
    
    def _create_quality_radar_chart(self, output_dir: Path) -> None:
        """Create radar chart of quality metrics."""
        if "metrics" not in self.results:
            return
        
        metrics = self.results["metrics"]
        
        categories = ['Completeness', 'Consistency', 'Uniqueness', 'Validity', 'Statistical Fidelity', 'ML Utility']
        values = [
            metrics.completeness_score,
            metrics.consistency_score,
            metrics.uniqueness_score,
            metrics.validity_score,
            metrics.statistical_fidelity,
            metrics.ml_utility_score
        ]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Quality Scores')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title('Dataset Quality Metrics', y=1.08, fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_plots(
        self, 
        synthetic_df: pd.DataFrame, 
        reference_df: pd.DataFrame, 
        output_dir: Path
    ) -> None:
        """Create distribution comparison plots."""
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
        common_numeric = [col for col in numeric_cols if col in reference_df.columns]
        
        if not common_numeric:
            return
        
        # Plot distributions for first few numeric columns
        n_cols = min(4, len(common_numeric))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(common_numeric[:n_cols]):
            ax = axes[i]
            
            synthetic_data = synthetic_df[col].dropna()
            reference_data = reference_df[col].dropna()
            
            ax.hist(reference_data, bins=30, alpha=0.7, label='Reference', density=True)
            ax.hist(synthetic_data, bins=30, alpha=0.7, label='Synthetic', density=True)
            ax.set_title(f'Distribution: {col}')
            ax.legend()
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_plots(
        self, 
        synthetic_df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame], 
        output_dir: Path
    ) -> None:
        """Create correlation heatmap plots."""
        numeric_df = synthetic_df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return
        
        if reference_df is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Synthetic data correlation
            corr_synthetic = numeric_df.corr()
            sns.heatmap(corr_synthetic, annot=True, cmap='coolwarm', center=0, ax=ax1)
            ax1.set_title('Synthetic Data Correlations')
            
            # Reference data correlation
            numeric_ref = reference_df.select_dtypes(include=[np.number])
            common_cols = list(set(numeric_df.columns) & set(numeric_ref.columns))
            if common_cols:
                corr_reference = numeric_ref[common_cols].corr()
                sns.heatmap(corr_reference, annot=True, cmap='coolwarm', center=0, ax=ax2)
                ax2.set_title('Reference Data Correlations')
            else:
                ax2.axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            corr_synthetic = numeric_df.corr()
            sns.heatmap(corr_synthetic, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Synthetic Data Correlations')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()


class BenchmarkSuite:
    """Suite of benchmarks for comprehensive dataset evaluation."""
    
    def __init__(self) -> None:
        """Initialize benchmark suite."""
        self.benchmarks = {}
        
    def add_benchmark(self, name: str, benchmark: SyntheticDatasetBenchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks[name] = benchmark
        
    def run_all_benchmarks(
        self,
        datasets: Dict[str, pd.DataFrame],
        reference_data: Optional[pd.DataFrame] = None,
        target_columns: Optional[Dict[str, str]] = None
    ) -> Dict[str, DatasetQualityMetrics]:
        """Run all benchmarks on multiple datasets."""
        results = {}
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"Running benchmarks for dataset: {dataset_name}")
            
            target_col = target_columns.get(dataset_name) if target_columns else None
            
            for benchmark_name, benchmark in self.benchmarks.items():
                try:
                    metrics = benchmark.benchmark_dataset(
                        dataset, reference_data, target_col
                    )
                    results[f"{dataset_name}_{benchmark_name}"] = metrics
                    
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_name} failed for {dataset_name}: {e}")
                    continue
        
        # Create comparative report
        self._create_comparative_report(results)
        
        return results
    
    def _create_comparative_report(self, results: Dict[str, DatasetQualityMetrics]) -> None:
        """Create comparative report across all benchmarks."""
        comparison_data = []
        
        for name, metrics in results.items():
            comparison_data.append({
                'dataset_benchmark': name,
                'overall_quality': metrics.overall_quality_score,
                'completeness': metrics.completeness_score,
                'consistency': metrics.consistency_score,
                'uniqueness': metrics.uniqueness_score,
                'validity': metrics.validity_score,
                'statistical_fidelity': metrics.statistical_fidelity,
                'ml_utility': metrics.ml_utility_score
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparative report
        report_path = Path("comparative_benchmark_report.csv")
        df.to_csv(report_path, index=False)
        
        # Create summary statistics
        summary = df.describe()
        summary_path = Path("benchmark_summary_statistics.csv")
        summary.to_csv(summary_path)
        
        logger.info(f"Comparative reports saved to: {report_path} and {summary_path}") 