"""Advanced scientific data generation strategies for OpenSynthetics."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import GenerationError
from opensynthetics.datagen.engine import GenerationStrategy
from opensynthetics.data_ops.arxiv_client import ArxivClient, ArxivSearchQuery
from opensynthetics.data_ops.pdf_processor import ScientificPDFProcessor
from opensynthetics.data_ops.pubmed_client import PubMedClient, PubMedSearchQuery
from opensynthetics.training.llm_trainer import (
    FineTuningConfig,
    LLMTrainer,
    TrainingDataFormat,
)


class ScientificLiteratureParams(BaseModel):
    """Parameters for scientific literature data generation."""
    
    # Data sources
    use_arxiv: bool = Field(True, description="Include arXiv papers")
    use_pubmed: bool = Field(True, description="Include PubMed papers")
    use_pdf_upload: bool = Field(True, description="Allow PDF uploads")
    
    # Search parameters
    arxiv_categories: List[str] = Field(
        default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL"],
        description="arXiv categories to search"
    )
    pubmed_subjects: List[str] = Field(
        default_factory=lambda: ["Artificial Intelligence", "Machine Learning"],
        description="PubMed subject areas"
    )
    search_queries: List[str] = Field(
        default_factory=lambda: ["machine learning", "artificial intelligence"],
        description="Additional search queries"
    )
    
    # Data processing
    max_papers: int = Field(50, description="Maximum papers per source", ge=1, le=500)
    include_full_text: bool = Field(True, description="Download and process full PDFs")
    include_abstracts_only: bool = Field(False, description="Include abstract-only papers")
    days_back: int = Field(30, description="Days to look back for recent papers", ge=1, le=365)
    
    # Training data generation
    generate_training_data: bool = Field(True, description="Generate LLM training data")
    training_formats: List[str] = Field(
        default_factory=lambda: ["alpaca", "instruction"],
        description="Training data formats to generate"
    )
    create_qa_pairs: bool = Field(True, description="Create Q&A pairs")
    create_summaries: bool = Field(True, description="Create summary tasks")
    create_explanations: bool = Field(True, description="Create explanation tasks")
    
    # LLM fine-tuning setup
    setup_training_pipeline: bool = Field(False, description="Setup complete training pipeline")
    base_model: str = Field("microsoft/DialoGPT-medium", description="Base model for fine-tuning")
    output_dir: str = Field("./scientific_llm_training", description="Training output directory")


class ScientificDatasetParams(BaseModel):
    """Parameters for comprehensive scientific dataset generation."""
    
    # Dataset composition
    research_domains: List[str] = Field(
        default_factory=lambda: ["computer_science", "medicine", "physics", "biology"],
        description="Research domains to include"
    )
    dataset_size: int = Field(1000, description="Target dataset size", ge=100, le=10000)
    train_test_split: float = Field(0.8, description="Training/test split ratio", gt=0.0, lt=1.0)
    
    # Data quality
    min_abstract_length: int = Field(100, description="Minimum abstract length", ge=50)
    min_full_text_length: int = Field(1000, description="Minimum full text length", ge=500)
    require_doi: bool = Field(False, description="Require DOI for papers")
    language_filter: List[str] = Field(
        default_factory=lambda: ["english"],
        description="Language filters"
    )
    
    # Advanced features
    include_citations: bool = Field(True, description="Include citation analysis")
    include_author_networks: bool = Field(True, description="Include author collaboration networks")
    include_temporal_analysis: bool = Field(True, description="Include temporal publication analysis")
    
    # Output formats
    output_formats: List[str] = Field(
        default_factory=lambda: ["json", "jsonl", "csv", "parquet"],
        description="Output data formats"
    )
    create_embeddings: bool = Field(False, description="Generate text embeddings")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model name")


class ScientificLiteratureStrategy(GenerationStrategy):
    """Strategy for generating scientific literature training data."""
    
    parameter_model = ScientificLiteratureParams
    
    def __init__(self, parameters: Dict[str, Any], config: Config) -> None:
        """Initialize strategy."""
        super().__init__(parameters, config)
        
        # Initialize clients
        self.arxiv_client = ArxivClient() if self.parameters.get("use_arxiv") else None
        self.pubmed_client = PubMedClient() if self.parameters.get("use_pubmed") else None
        self.pdf_processor = ScientificPDFProcessor()
        
        # Initialize trainer if needed
        self.llm_trainer = None
        if self.parameters.get("setup_training_pipeline"):
            self._setup_llm_trainer()
    
    def _setup_llm_trainer(self) -> None:
        """Setup LLM trainer for fine-tuning pipeline."""
        try:
            training_config = FineTuningConfig(
                model_name=self.parameters.get("base_model", "microsoft/DialoGPT-medium"),
                output_dir=self.parameters.get("output_dir", "./scientific_llm_training"),
                data_format=TrainingDataFormat(format_type="alpaca")
            )
            self.llm_trainer = LLMTrainer(training_config)
            logger.info("LLM trainer initialized for scientific literature")
        except Exception as e:
            logger.warning(f"Failed to setup LLM trainer: {e}")
            self.llm_trainer = None
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate scientific literature training data."""
        logger.info("Starting scientific literature data generation")
        
        all_data = []
        generation_metadata = {
            "generated_at": datetime.now().isoformat(),
            "parameters": self.parameters,
            "sources": [],
            "total_papers": 0,
            "total_training_segments": 0
        }
        
        try:
            # Generate from arXiv
            if self.arxiv_client:
                arxiv_data = self._generate_from_arxiv()
                all_data.extend(arxiv_data)
                generation_metadata["sources"].append("arxiv")
                logger.info(f"Generated {len(arxiv_data)} items from arXiv")
            
            # Generate from PubMed
            if self.pubmed_client:
                pubmed_data = self._generate_from_pubmed()
                all_data.extend(pubmed_data)
                generation_metadata["sources"].append("pubmed")
                logger.info(f"Generated {len(pubmed_data)} items from PubMed")
            
            # Process uploaded PDFs if enabled
            if self.parameters.get("use_pdf_upload"):
                pdf_data = self._process_uploaded_pdfs()
                all_data.extend(pdf_data)
                generation_metadata["sources"].append("uploaded_pdfs")
                logger.info(f"Processed {len(pdf_data)} uploaded PDFs")
            
            # Generate training data
            if self.parameters.get("generate_training_data"):
                training_data = self._create_training_data(all_data)
                all_data.extend(training_data)
                generation_metadata["total_training_segments"] = len(training_data)
                logger.info(f"Generated {len(training_data)} training segments")
            
            # Setup training pipeline if requested
            if self.llm_trainer and all_data:
                pipeline_info = self._setup_training_pipeline(all_data)
                generation_metadata["training_pipeline"] = pipeline_info
                logger.info("Training pipeline setup completed")
            
            generation_metadata["total_papers"] = len([item for item in all_data if item.get("type") == "paper"])
            
            # Add metadata to results
            all_data.append({
                "type": "generation_metadata",
                "metadata": generation_metadata
            })
            
            logger.info(f"Scientific literature generation completed: {len(all_data)} total items")
            return all_data
            
        except Exception as e:
            logger.error(f"Error in scientific literature generation: {e}")
            raise GenerationError(f"Failed to generate scientific literature data: {e}")
    
    def _generate_from_arxiv(self) -> List[Dict[str, Any]]:
        """Generate data from arXiv papers."""
        if not self.arxiv_client:
            return []
        
        results = []
        categories = self.parameters.get("arxiv_categories", ["cs.AI"])
        max_papers = self.parameters.get("max_papers", 50)
        include_full_text = self.parameters.get("include_full_text", True)
        
        try:
            # Get recent papers from categories
            recent_papers = self.arxiv_client.get_recent_papers(
                categories=categories,
                days_back=self.parameters.get("days_back", 30),
                max_results=max_papers
            )
            
            # Process papers
            for paper in recent_papers[:max_papers]:
                try:
                    paper_data = {
                        "type": "paper",
                        "source": "arxiv",
                        "arxiv_id": paper.id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "categories": paper.categories,
                        "published": paper.published.isoformat() if paper.published else None,
                        "doi": paper.doi,
                        "pdf_url": paper.pdf_url
                    }
                    
                    # Download and process PDF if requested
                    if include_full_text:
                        try:
                            processed_pdf = self.arxiv_client.download_pdf(paper)
                            paper_data["processed_pdf"] = {
                                "sections": [s.dict() for s in processed_pdf.sections],
                                "full_text": processed_pdf.full_text,
                                "figures_count": processed_pdf.figures_count,
                                "tables_count": processed_pdf.tables_count,
                                "references_count": processed_pdf.references_count
                            }
                            
                            # Extract training segments
                            training_segments = self.pdf_processor.extract_training_segments(processed_pdf)
                            paper_data["training_segments"] = training_segments
                            
                        except Exception as e:
                            logger.warning(f"Failed to process PDF for {paper.id}: {e}")
                    
                    results.append(paper_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process arXiv paper {paper.id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating from arXiv: {e}")
            return []
    
    def _generate_from_pubmed(self) -> List[Dict[str, Any]]:
        """Generate data from PubMed papers."""
        if not self.pubmed_client:
            return []
        
        results = []
        subjects = self.parameters.get("pubmed_subjects", ["Artificial Intelligence"])
        max_papers = self.parameters.get("max_papers", 50)
        include_full_text = self.parameters.get("include_full_text", True)
        
        try:
            # Get recent papers from subject areas
            recent_papers = self.pubmed_client.get_recent_papers(
                subject_areas=subjects,
                days_back=self.parameters.get("days_back", 30),
                max_results=max_papers
            )
            
            # Process papers
            for paper in recent_papers[:max_papers]:
                try:
                    paper_data = {
                        "type": "paper",
                        "source": "pubmed",
                        "pmid": paper.pmid,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "journal": paper.journal,
                        "mesh_terms": paper.mesh_terms,
                        "publication_types": paper.publication_types,
                        "published": paper.publication_date.isoformat() if paper.publication_date else None,
                        "doi": paper.doi,
                        "pmc_id": paper.pmc_id
                    }
                    
                    # Try to download full text if available
                    if include_full_text and paper.pmc_id:
                        try:
                            processed_pdf = self.pubmed_client.download_full_text(paper)
                            if processed_pdf:
                                paper_data["processed_pdf"] = {
                                    "sections": [s.dict() for s in processed_pdf.sections],
                                    "full_text": processed_pdf.full_text,
                                    "figures_count": processed_pdf.figures_count,
                                    "tables_count": processed_pdf.tables_count,
                                    "references_count": processed_pdf.references_count
                                }
                                
                                # Extract training segments
                                training_segments = self.pdf_processor.extract_training_segments(processed_pdf)
                                paper_data["training_segments"] = training_segments
                        except Exception as e:
                            logger.warning(f"Failed to process full text for {paper.pmid}: {e}")
                    
                    # Create training segments from abstract if no full text
                    if "training_segments" not in paper_data and paper.abstract:
                        paper_data["training_segments"] = self._create_abstract_training_segments(paper)
                    
                    results.append(paper_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process PubMed paper {paper.pmid}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating from PubMed: {e}")
            return []
    
    def _process_uploaded_pdfs(self) -> List[Dict[str, Any]]:
        """Process uploaded PDF files."""
        # This would integrate with a file upload system
        # For now, we'll look for PDFs in a specific directory
        results = []
        
        upload_dir = Path("./uploads/pdfs")
        if not upload_dir.exists():
            return results
        
        try:
            pdf_files = list(upload_dir.glob("*.pdf"))
            
            for pdf_file in pdf_files:
                try:
                    processed_pdf = self.pdf_processor.process_pdf_file(pdf_file)
                    
                    paper_data = {
                        "type": "paper",
                        "source": "uploaded",
                        "filename": pdf_file.name,
                        "title": processed_pdf.metadata.title,
                        "authors": processed_pdf.metadata.authors,
                        "abstract": processed_pdf.metadata.abstract,
                        "doi": processed_pdf.metadata.doi,
                        "arxiv_id": processed_pdf.metadata.arxiv_id,
                        "pubmed_id": processed_pdf.metadata.pubmed_id,
                        "processed_pdf": {
                            "sections": [s.dict() for s in processed_pdf.sections],
                            "full_text": processed_pdf.full_text,
                            "figures_count": processed_pdf.figures_count,
                            "tables_count": processed_pdf.tables_count,
                            "references_count": processed_pdf.references_count
                        }
                    }
                    
                    # Extract training segments
                    training_segments = self.pdf_processor.extract_training_segments(processed_pdf)
                    paper_data["training_segments"] = training_segments
                    
                    results.append(paper_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process uploaded PDF {pdf_file}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing uploaded PDFs: {e}")
            return []
    
    def _create_abstract_training_segments(self, paper) -> List[Dict[str, Any]]:
        """Create training segments from abstract only."""
        segments = []
        
        if hasattr(paper, 'abstract') and paper.abstract:
            # Abstract summary segment
            segments.append({
                "type": "abstract",
                "title": paper.title if hasattr(paper, 'title') else "",
                "content": paper.abstract,
                "metadata": {
                    "source": "pubmed" if hasattr(paper, 'pmid') else "arxiv",
                    "paper_id": getattr(paper, 'pmid', getattr(paper, 'id', '')),
                    "authors": getattr(paper, 'authors', [])
                }
            })
        
        return segments
    
    def _create_training_data(self, papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create comprehensive training data from papers."""
        training_data = []
        
        paper_items = [item for item in papers_data if item.get("type") == "paper"]
        
        for paper in paper_items:
            try:
                # Extract basic information
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                authors = paper.get("authors", [])
                
                if not title or not abstract:
                    continue
                
                # Create Q&A pairs if requested
                if self.parameters.get("create_qa_pairs", True):
                    qa_pairs = self._create_qa_pairs(paper)
                    training_data.extend(qa_pairs)
                
                # Create summaries if requested
                if self.parameters.get("create_summaries", True):
                    summaries = self._create_summary_tasks(paper)
                    training_data.extend(summaries)
                
                # Create explanations if requested
                if self.parameters.get("create_explanations", True):
                    explanations = self._create_explanation_tasks(paper)
                    training_data.extend(explanations)
                
                # Add training segments from full text if available
                if "training_segments" in paper:
                    for segment in paper["training_segments"]:
                        training_data.append({
                            "type": "training_segment",
                            "source_paper": paper.get("title", ""),
                            "segment_type": segment.get("type", ""),
                            "content": segment
                        })
                
            except Exception as e:
                logger.warning(f"Failed to create training data for paper: {e}")
                continue
        
        return training_data
    
    def _create_qa_pairs(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Q&A pairs from paper."""
        qa_pairs = []
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Basic Q&A pairs
        questions = [
            f"What is the main contribution of the paper '{title}'?",
            f"Summarize the research described in '{title}'.",
            f"What problem does this paper address: {title}?",
            f"What are the key findings of this research?",
        ]
        
        for question in questions:
            qa_pairs.append({
                "type": "qa_training",
                "question": question,
                "answer": abstract,
                "metadata": {
                    "source_paper": title,
                    "authors": paper.get("authors", []),
                    "synthetic": True
                }
            })
        
        return qa_pairs
    
    def _create_summary_tasks(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create summary tasks from paper."""
        summaries = []
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Different summary formats
        summary_tasks = [
            {
                "instruction": "Provide a concise summary of this research paper.",
                "input": f"Title: {title}",
                "output": abstract
            },
            {
                "instruction": "Create an executive summary for this scientific paper.",
                "input": f"Research: {title}",
                "output": f"This research paper presents: {abstract}"
            }
        ]
        
        for task in summary_tasks:
            summaries.append({
                "type": "summary_training",
                "task": task,
                "metadata": {
                    "source_paper": title,
                    "format": "summary"
                }
            })
        
        return summaries
    
    def _create_explanation_tasks(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create explanation tasks from paper."""
        explanations = []
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Explanation tasks
        explanation_tasks = [
            {
                "instruction": "Explain the methodology used in this research.",
                "input": f"Paper: {title}",
                "output": f"The methodology involves: {abstract}"
            },
            {
                "instruction": "What are the implications of this research?",
                "input": f"Study: {title}",
                "output": f"The implications include: {abstract}"
            }
        ]
        
        for task in explanation_tasks:
            explanations.append({
                "type": "explanation_training",
                "task": task,
                "metadata": {
                    "source_paper": title,
                    "format": "explanation"
                }
            })
        
        return explanations
    
    def _setup_training_pipeline(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Setup complete LLM training pipeline."""
        if not self.llm_trainer:
            return {"error": "LLM trainer not available"}
        
        try:
            # Prepare scientific dataset for training
            scientific_dataset = {
                "papers": [item for item in data if item.get("type") == "paper"],
                "training_segments": [item.get("content", item) for item in data if item.get("type") == "training_segment"]
            }
            
            # Create training pipeline
            pipeline_result = self.llm_trainer.create_training_pipeline(
                scientific_dataset,
                run_training=False
            )
            
            return {
                "status": "success",
                "output_directory": pipeline_result["output_dir"],
                "files_created": pipeline_result,
                "model_base": self.llm_trainer.config.model_name,
                "training_method": "QLoRA" if self.llm_trainer.config.use_qlora else "Full Fine-tuning"
            }
            
        except Exception as e:
            logger.error(f"Failed to setup training pipeline: {e}")
            return {"error": str(e)}


class ScientificDatasetStrategy(GenerationStrategy):
    """Strategy for generating comprehensive scientific datasets."""
    
    parameter_model = ScientificDatasetParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate comprehensive scientific dataset."""
        logger.info("Starting comprehensive scientific dataset generation")
        
        try:
            # Initialize components
            dataset_params = self.parameters
            arxiv_client = ArxivClient()
            pubmed_client = PubMedClient()
            
            # Generate dataset by domain
            dataset = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "parameters": dataset_params,
                    "target_size": dataset_params.get("dataset_size", 1000),
                    "domains": dataset_params.get("research_domains", [])
                },
                "papers": [],
                "statistics": {},
                "embeddings": None
            }
            
            domains = dataset_params.get("research_domains", ["computer_science"])
            target_size = dataset_params.get("dataset_size", 1000)
            papers_per_domain = target_size // len(domains)
            
            for domain in domains:
                try:
                    domain_papers = self._generate_domain_data(
                        domain, papers_per_domain, arxiv_client, pubmed_client
                    )
                    dataset["papers"].extend(domain_papers)
                    
                    logger.info(f"Generated {len(domain_papers)} papers for domain: {domain}")
                    
                except Exception as e:
                    logger.error(f"Error generating data for domain {domain}: {e}")
                    continue
            
            # Generate statistics
            dataset["statistics"] = self._compute_dataset_statistics(dataset["papers"])
            
            # Generate embeddings if requested
            if dataset_params.get("create_embeddings", False):
                dataset["embeddings"] = self._generate_embeddings(dataset["papers"])
            
            # Export in multiple formats
            export_results = self._export_dataset(dataset, dataset_params)
            
            # Return summary
            result = [{
                "type": "dataset_generation_result",
                "total_papers": len(dataset["papers"]),
                "domains": domains,
                "statistics": dataset["statistics"],
                "export_files": export_results,
                "metadata": dataset["metadata"]
            }]
            
            logger.info(f"Scientific dataset generation completed: {len(dataset['papers'])} papers")
            return result
            
        except Exception as e:
            logger.error(f"Error in scientific dataset generation: {e}")
            raise GenerationError(f"Failed to generate scientific dataset: {e}")
    
    def _generate_domain_data(
        self, 
        domain: str, 
        count: int, 
        arxiv_client: ArxivClient, 
        pubmed_client: PubMedClient
    ) -> List[Dict[str, Any]]:
        """Generate data for a specific research domain."""
        papers = []
        
        # Domain-specific search configuration
        domain_config = self._get_domain_config(domain)
        
        # Get papers from arXiv
        if domain_config.get("arxiv_categories"):
            try:
                query = ArxivSearchQuery(
                    query=f"cat:{' OR cat:'.join(domain_config['arxiv_categories'])}",
                    max_results=count // 2
                )
                arxiv_papers = arxiv_client.search(query)
                
                for paper in arxiv_papers:
                    papers.append({
                        "source": "arxiv",
                        "domain": domain,
                        "id": paper.id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "categories": paper.categories,
                        "published": paper.published.isoformat() if paper.published else None,
                        "doi": paper.doi
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to get arXiv papers for {domain}: {e}")
        
        # Get papers from PubMed
        if domain_config.get("pubmed_subjects"):
            try:
                query = PubMedSearchQuery(
                    query=" OR ".join([f'"{subject}"[MeSH Terms]' for subject in domain_config["pubmed_subjects"]]),
                    max_results=count // 2
                )
                pubmed_papers = pubmed_client.search(query)
                
                for paper in pubmed_papers:
                    papers.append({
                        "source": "pubmed",
                        "domain": domain,
                        "id": paper.pmid,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "mesh_terms": paper.mesh_terms,
                        "journal": paper.journal,
                        "published": paper.publication_date.isoformat() if paper.publication_date else None,
                        "doi": paper.doi
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to get PubMed papers for {domain}: {e}")
        
        return papers[:count]
    
    def _get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get search configuration for a research domain."""
        configs = {
            "computer_science": {
                "arxiv_categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"],
                "pubmed_subjects": ["Artificial Intelligence", "Machine Learning", "Natural Language Processing"]
            },
            "medicine": {
                "arxiv_categories": ["q-bio.QM", "physics.med-ph"],
                "pubmed_subjects": ["Medicine", "Clinical Medicine", "Biomedical Research"]
            },
            "physics": {
                "arxiv_categories": ["physics.gen-ph", "quant-ph", "physics.comp-ph"],
                "pubmed_subjects": ["Physics", "Biophysics"]
            },
            "biology": {
                "arxiv_categories": ["q-bio.BM", "q-bio.CB", "q-bio.GN"],
                "pubmed_subjects": ["Biology", "Molecular Biology", "Cell Biology", "Genetics"]
            }
        }
        
        return configs.get(domain, {"arxiv_categories": [], "pubmed_subjects": []})
    
    def _compute_dataset_statistics(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for the dataset."""
        stats = {
            "total_papers": len(papers),
            "by_source": {},
            "by_domain": {},
            "by_year": {},
            "abstract_lengths": [],
            "author_counts": []
        }
        
        for paper in papers:
            # Source distribution
            source = paper.get("source", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            
            # Domain distribution
            domain = paper.get("domain", "unknown")
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
            
            # Year distribution
            published = paper.get("published")
            if published:
                try:
                    year = datetime.fromisoformat(published.replace('Z', '+00:00')).year
                    stats["by_year"][str(year)] = stats["by_year"].get(str(year), 0) + 1
                except:
                    pass
            
            # Abstract length
            abstract = paper.get("abstract", "")
            if abstract:
                stats["abstract_lengths"].append(len(abstract))
            
            # Author count
            authors = paper.get("authors", [])
            stats["author_counts"].append(len(authors))
        
        # Compute summary statistics
        if stats["abstract_lengths"]:
            stats["avg_abstract_length"] = sum(stats["abstract_lengths"]) / len(stats["abstract_lengths"])
            stats["median_abstract_length"] = sorted(stats["abstract_lengths"])[len(stats["abstract_lengths"]) // 2]
        
        if stats["author_counts"]:
            stats["avg_authors_per_paper"] = sum(stats["author_counts"]) / len(stats["author_counts"])
        
        return stats
    
    def _generate_embeddings(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for papers (placeholder for now)."""
        # This would integrate with embedding models
        logger.info("Embedding generation requested but not implemented in this demo")
        return {
            "model": self.parameters.get("embedding_model", "all-MiniLM-L6-v2"),
            "status": "not_implemented",
            "papers_count": len(papers)
        }
    
    def _export_dataset(self, dataset: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, str]:
        """Export dataset in multiple formats."""
        output_formats = params.get("output_formats", ["json"])
        export_files = {}
        
        # Create output directory
        output_dir = Path("./scientific_datasets")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in output_formats:
            try:
                if format_type == "json":
                    file_path = output_dir / f"scientific_dataset_{timestamp}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)
                    export_files["json"] = str(file_path)
                
                elif format_type == "jsonl":
                    file_path = output_dir / f"scientific_dataset_{timestamp}.jsonl"
                    with open(file_path, "w", encoding="utf-8") as f:
                        for paper in dataset["papers"]:
                            f.write(json.dumps(paper, ensure_ascii=False, default=str) + "\n")
                    export_files["jsonl"] = str(file_path)
                
                # Additional formats would be implemented here
                
            except Exception as e:
                logger.error(f"Failed to export in format {format_type}: {e}")
                continue
        
        return export_files 