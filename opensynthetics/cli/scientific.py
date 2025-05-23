"""CLI commands for scientific data generation and LLM training."""

import json
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace
from opensynthetics.datagen.engine import Engine

app = typer.Typer(name="scientific", help="Scientific data generation and LLM training")
console = Console()

# Create alias for import
scientific = app


@app.command()
def literature(
    workspace_path: str = typer.Argument(help="Path to workspace"),
    output_dataset: str = typer.Option("scientific_literature", help="Output dataset name"),
    arxiv_categories: List[str] = typer.Option(
        ["cs.AI", "cs.LG"],
        help="arXiv categories to search"
    ),
    pubmed_subjects: List[str] = typer.Option(
        ["Artificial Intelligence", "Machine Learning"],
        help="PubMed subject areas"
    ),
    max_papers: int = typer.Option(50, help="Maximum papers per source"),
    include_full_text: bool = typer.Option(True, help="Download and process full PDFs"),
    setup_training: bool = typer.Option(False, help="Setup LLM training pipeline"),
    base_model: str = typer.Option(
        "microsoft/DialoGPT-medium",
        help="Base model for fine-tuning"
    ),
    output_dir: str = typer.Option(
        "./scientific_llm_training",
        help="Training output directory"
    ),
) -> None:
    """Generate scientific literature training data from arXiv and PubMed."""
    
    console.print("[bold blue]Starting scientific literature generation...[/bold blue]")
    
    try:
        # Load workspace
        workspace = Workspace.load(Path(workspace_path))
        engine = Engine(workspace)
        
        # Configure parameters
        parameters = {
            "use_arxiv": True,
            "use_pubmed": True,
            "use_pdf_upload": True,
            "arxiv_categories": arxiv_categories,
            "pubmed_subjects": pubmed_subjects,
            "max_papers": max_papers,
            "include_full_text": include_full_text,
            "days_back": 30,
            "generate_training_data": True,
            "training_formats": ["alpaca", "instruction"],
            "create_qa_pairs": True,
            "create_summaries": True,
            "create_explanations": True,
            "setup_training_pipeline": setup_training,
            "base_model": base_model,
            "output_dir": output_dir,
        }
        
        # Generate data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating scientific literature data...", total=None)
            
            result = engine.generate(
                strategy="scientific_literature",
                parameters=parameters,
                output_dataset=output_dataset,
            )
        
        # Display results
        console.print("[bold green]✓ Scientific literature generation completed![/bold green]")
        console.print(f"Generated {result['count']} items")
        console.print(f"Output dataset: {result['output_dataset']}")
        console.print(f"Workspace: {result['workspace']}")
        
        if setup_training and result.get("sample_items"):
            metadata_item = next(
                (item for item in result["sample_items"] if item.get("type") == "generation_metadata"),
                None
            )
            if metadata_item and "training_pipeline" in metadata_item.get("metadata", {}):
                pipeline_info = metadata_item["metadata"]["training_pipeline"]
                if pipeline_info.get("status") == "success":
                    console.print("\n[bold green]✓ Training pipeline setup completed![/bold green]")
                    console.print(f"Output directory: {pipeline_info['output_directory']}")
                    console.print(f"Base model: {pipeline_info['model_base']}")
                    console.print(f"Training method: {pipeline_info['training_method']}")
                    
                    # Show created files
                    table = Table(title="Generated Training Files")
                    table.add_column("File Type", style="cyan")
                    table.add_column("Path", style="green")
                    
                    for file_type, path in pipeline_info.get("files_created", {}).items():
                        table.add_row(file_type.title(), str(path))
                    
                    console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def dataset(
    workspace_path: str = typer.Argument(help="Path to workspace"),
    output_dataset: str = typer.Option("comprehensive_scientific", help="Output dataset name"),
    research_domains: List[str] = typer.Option(
        ["computer_science", "medicine", "physics", "biology"],
        help="Research domains to include"
    ),
    dataset_size: int = typer.Option(1000, help="Target dataset size"),
    output_formats: List[str] = typer.Option(
        ["json", "jsonl"],
        help="Output data formats"
    ),
    create_embeddings: bool = typer.Option(False, help="Generate text embeddings"),
) -> None:
    """Generate comprehensive scientific datasets from multiple sources."""
    
    console.print("[bold blue]Starting comprehensive scientific dataset generation...[/bold blue]")
    
    try:
        # Load workspace
        workspace = Workspace.load(Path(workspace_path))
        engine = Engine(workspace)
        
        # Configure parameters
        parameters = {
            "research_domains": research_domains,
            "dataset_size": dataset_size,
            "train_test_split": 0.8,
            "min_abstract_length": 100,
            "min_full_text_length": 1000,
            "require_doi": False,
            "language_filter": ["english"],
            "include_citations": True,
            "include_author_networks": True,
            "include_temporal_analysis": True,
            "output_formats": output_formats,
            "create_embeddings": create_embeddings,
            "embedding_model": "all-MiniLM-L6-v2",
        }
        
        # Generate dataset
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating comprehensive scientific dataset...", total=None)
            
            result = engine.generate(
                strategy="scientific_dataset",
                parameters=parameters,
                output_dataset=output_dataset,
            )
        
        # Display results
        console.print("[bold green]✓ Scientific dataset generation completed![/bold green]")
        console.print(f"Generated {result['count']} items")
        
        if result.get("sample_items"):
            dataset_result = result["sample_items"][0]
            
            # Show statistics
            stats_table = Table(title="Dataset Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats = dataset_result.get("statistics", {})
            stats_table.add_row("Total Papers", str(stats.get("total_papers", 0)))
            stats_table.add_row("Domains", ", ".join(dataset_result.get("domains", [])))
            
            if "avg_abstract_length" in stats:
                stats_table.add_row("Avg Abstract Length", f"{stats['avg_abstract_length']:.1f} chars")
            if "avg_authors_per_paper" in stats:
                stats_table.add_row("Avg Authors per Paper", f"{stats['avg_authors_per_paper']:.1f}")
            
            console.print(stats_table)
            
            # Show export files
            export_files = dataset_result.get("export_files", {})
            if export_files:
                files_table = Table(title="Export Files")
                files_table.add_column("Format", style="cyan")
                files_table.add_column("Path", style="green")
                
                for format_type, path in export_files.items():
                    files_table.add_row(format_type.upper(), str(path))
                
                console.print(files_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def process_pdf(
    pdf_path: str = typer.Argument(help="Path to PDF file"),
    output_file: Optional[str] = typer.Option(None, help="Output JSON file path"),
    extract_training: bool = typer.Option(True, help="Extract training segments"),
) -> None:
    """Process a scientific PDF and extract training data."""
    
    console.print(f"[bold blue]Processing PDF: {pdf_path}[/bold blue]")
    
    try:
        from opensynthetics.data_ops.pdf_processor import ScientificPDFProcessor
        
        processor = ScientificPDFProcessor()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing PDF...", total=None)
            
            # Process PDF
            processed_pdf = processor.process_pdf_file(pdf_path)
            
            # Extract training segments if requested
            training_segments = []
            if extract_training:
                training_segments = processor.extract_training_segments(processed_pdf)
        
        # Prepare output data
        output_data = {
            "metadata": processed_pdf.metadata.dict(),
            "sections": [section.dict() for section in processed_pdf.sections],
            "statistics": {
                "page_count": processed_pdf.metadata.page_count,
                "figures_count": processed_pdf.figures_count,
                "tables_count": processed_pdf.tables_count,
                "references_count": processed_pdf.references_count,
                "processing_time": processed_pdf.processing_time,
            },
            "training_segments": training_segments,
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"Results saved to: {output_file}")
        
        # Display summary
        console.print("[bold green]✓ PDF processing completed![/bold green]")
        
        summary_table = Table(title="PDF Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Title", processed_pdf.metadata.title or "Not found")
        summary_table.add_row("Authors", ", ".join(processed_pdf.metadata.authors[:3]) or "Not found")
        summary_table.add_row("Pages", str(processed_pdf.metadata.page_count))
        summary_table.add_row("Sections", str(len(processed_pdf.sections)))
        summary_table.add_row("Figures", str(processed_pdf.figures_count))
        summary_table.add_row("Tables", str(processed_pdf.tables_count))
        summary_table.add_row("References", str(processed_pdf.references_count))
        summary_table.add_row("Training Segments", str(len(training_segments)))
        summary_table.add_row("Processing Time", f"{processed_pdf.processing_time:.2f}s")
        
        console.print(summary_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def arxiv_search(
    query: str = typer.Argument(help="Search query"),
    max_results: int = typer.Option(10, help="Maximum number of results"),
    download_pdfs: bool = typer.Option(False, help="Download and process PDFs"),
    output_file: Optional[str] = typer.Option(None, help="Output JSON file path"),
) -> None:
    """Search arXiv papers and optionally download PDFs."""
    
    console.print(f"[bold blue]Searching arXiv: {query}[/bold blue]")
    
    try:
        from opensynthetics.data_ops.arxiv_client import ArxivClient, ArxivSearchQuery
        
        client = ArxivClient()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching arXiv...", total=None)
            
            # Search papers
            search_query = ArxivSearchQuery(
                query=query,
                max_results=max_results
            )
            
            if download_pdfs:
                results = client.search_and_download(search_query, download_pdfs=True)
            else:
                papers = client.search(search_query)
                results = [{"paper": paper, "processed_pdf": None, "training_segments": None} for paper in papers]
        
        # Save results if requested
        if output_file:
            # Convert to JSON-serializable format
            json_results = []
            for result in results:
                json_result = {
                    "paper": result["paper"].dict(),
                    "has_processed_pdf": result["processed_pdf"] is not None,
                    "training_segments_count": len(result["training_segments"]) if result["training_segments"] else 0
                }
                if result["training_segments"]:
                    json_result["training_segments"] = result["training_segments"]
                json_results.append(json_result)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"Results saved to: {output_file}")
        
        # Display results
        console.print(f"[bold green]✓ Found {len(results)} papers[/bold green]")
        
        results_table = Table(title="arXiv Search Results")
        results_table.add_column("ID", style="cyan")
        results_table.add_column("Title", style="white", max_width=50)
        results_table.add_column("Authors", style="green", max_width=30)
        results_table.add_column("Categories", style="yellow")
        results_table.add_column("PDF", style="red")
        
        for result in results[:10]:  # Show first 10 results
            paper = result["paper"]
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += f" et al. ({len(paper.authors)} total)"
            
            pdf_status = "✓" if result["processed_pdf"] else "✗"
            
            results_table.add_row(
                paper.id,
                paper.title[:47] + "..." if len(paper.title) > 50 else paper.title,
                authors_str,
                ", ".join(paper.categories[:2]),
                pdf_status
            )
        
        console.print(results_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def pubmed_search(
    query: str = typer.Argument(help="Search query"),
    max_results: int = typer.Option(10, help="Maximum number of results"),
    download_full_text: bool = typer.Option(False, help="Download full text when available"),
    output_file: Optional[str] = typer.Option(None, help="Output JSON file path"),
) -> None:
    """Search PubMed papers and optionally download full text."""
    
    console.print(f"[bold blue]Searching PubMed: {query}[/bold blue]")
    
    try:
        from opensynthetics.data_ops.pubmed_client import PubMedClient, PubMedSearchQuery
        
        client = PubMedClient()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching PubMed...", total=None)
            
            # Search papers
            search_query = PubMedSearchQuery(
                query=query,
                max_results=max_results
            )
            
            results = client.search_and_process(search_query, download_full_text=download_full_text)
        
        # Save results if requested
        if output_file:
            # Convert to JSON-serializable format
            json_results = []
            for result in results:
                json_result = {
                    "paper": result["paper"].dict(),
                    "has_processed_pdf": result["processed_pdf"] is not None,
                    "training_segments_count": len(result["training_segments"]) if result["training_segments"] else 0
                }
                if result["training_segments"]:
                    json_result["training_segments"] = result["training_segments"]
                json_results.append(json_result)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"Results saved to: {output_file}")
        
        # Display results
        console.print(f"[bold green]✓ Found {len(results)} papers[/bold green]")
        
        results_table = Table(title="PubMed Search Results")
        results_table.add_column("PMID", style="cyan")
        results_table.add_column("Title", style="white", max_width=50)
        results_table.add_column("Authors", style="green", max_width=30)
        results_table.add_column("Journal", style="yellow", max_width=20)
        results_table.add_column("Full Text", style="red")
        
        for result in results[:10]:  # Show first 10 results
            paper = result["paper"]
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += f" et al. ({len(paper.authors)} total)"
            
            full_text_status = "✓" if result["processed_pdf"] else "✗"
            
            results_table.add_row(
                paper.pmid,
                paper.title[:47] + "..." if len(paper.title) > 50 else paper.title,
                authors_str,
                paper.journal[:17] + "..." if len(paper.journal) > 20 else paper.journal,
                full_text_status
            )
        
        console.print(results_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def create_training_pipeline(
    data_file: str = typer.Argument(help="Path to scientific data JSON file"),
    base_model: str = typer.Option("microsoft/DialoGPT-medium", help="Base model for fine-tuning"),
    output_dir: str = typer.Option("./llm_training", help="Training output directory"),
    use_qlora: bool = typer.Option(True, help="Use QLoRA for parameter-efficient fine-tuning"),
    format_type: str = typer.Option("alpaca", help="Training data format"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Training batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    run_training: bool = typer.Option(False, help="Start training immediately"),
) -> None:
    """Create a complete LLM training pipeline from scientific data."""
    
    console.print("[bold blue]Creating LLM training pipeline...[/bold blue]")
    
    try:
        from opensynthetics.training.llm_trainer import (
            FineTuningConfig,
            LLMTrainer,
            TrainingDataFormat,
        )
        
        # Load scientific data
        with open(data_file, "r", encoding="utf-8") as f:
            scientific_data = json.load(f)
        
        # Create training configuration
        training_config = FineTuningConfig(
            model_name=base_model,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            use_qlora=use_qlora,
            data_format=TrainingDataFormat(format_type=format_type),
        )
        
        # Initialize trainer
        trainer = LLMTrainer(training_config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Setting up training pipeline...", total=None)
            
            # Create training pipeline
            pipeline_result = trainer.create_training_pipeline(
                scientific_data,
                run_training=run_training
            )
        
        # Display results
        console.print("[bold green]✓ Training pipeline created successfully![/bold green]")
        console.print(f"Output directory: {pipeline_result['output_dir']}")
        console.print(f"Base model: {base_model}")
        console.print(f"Training method: {'QLoRA' if use_qlora else 'Full Fine-tuning'}")
        
        # Show created files
        files_table = Table(title="Generated Files")
        files_table.add_column("File Type", style="cyan")
        files_table.add_column("Path", style="green")
        
        for file_type, path in pipeline_result.items():
            if file_type != "output_dir":
                files_table.add_row(file_type.replace("_", " ").title(), str(path))
        
        console.print(files_table)
        
        if run_training:
            console.print("\n[bold yellow]Training started in background...[/bold yellow]")
        else:
            console.print(f"\n[bold cyan]To start training, run:[/bold cyan]")
            console.print(f"python {pipeline_result['script']}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 