"""Command-line interface for OpenSynthetics."""

import json
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

from opensynthetics.core.config import Config
from opensynthetics.core.workspace import Workspace, WorkspaceError, DatasetError
from opensynthetics.cli.scientific import scientific

# Import synthetic commands from the Typer wrapper
try:
    from opensynthetics.cli.synthetic_typer import app as synthetic
    SYNTHETIC_AVAILABLE = True
except ImportError:
    SYNTHETIC_AVAILABLE = False

# Import only when actually starting the API to avoid circular imports
# and unnecessary dependencies loading when just using the CLI
api_app = None

app = typer.Typer(
    name="opensynthetics",
    help="OpenSynthetics: Advanced Synthetic Data Generation Platform",
)

console = Console()

# Create subcommands
config_app = typer.Typer(name="config", help="Configuration management")
generate_app = typer.Typer(name="generate", help="Data generation")
data_app = typer.Typer(name="data", help="Data management")
api_cli_app = typer.Typer(name="api", help="API server")
datasette_app = typer.Typer(name="datasette", help="Datasette integration")
agent_app = typer.Typer(name="agent", help="Agent interaction")

app.add_typer(config_app)
app.add_typer(generate_app)
app.add_typer(data_app)
app.add_typer(api_cli_app)
app.add_typer(datasette_app)
app.add_typer(agent_app)

# Add command groups
app.add_typer(scientific, name="scientific")

if SYNTHETIC_AVAILABLE:
    app.add_typer(synthetic, name="synthetic")


@app.command("init")
def init_project(
    name: str = typer.Argument(..., help="Project name"),
    path: Optional[Path] = typer.Option(None, help="Project path"),
    description: str = typer.Option("", help="Project description"),
) -> None:
    """Initialize a new project workspace."""
    try:
        workspace = Workspace.create(
            name=name,
            path=path,
            description=description,
        )
        console.print(f"[green]Project {name} initialized at {workspace.path}[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing project: {e}[/red]")
        sys.exit(1)


@app.command("version")
def version() -> None:
    """Show OpenSynthetics version."""
    console.print("[blue]OpenSynthetics v0.1.0[/blue]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key (dot notation, e.g. 'api_keys.openai')"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """Set a configuration value."""
    try:
        config = Config.load()
        
        # Special case for api keys which is a common use case
        if key.startswith("api_keys."):
            provider = key.split(".", 1)[1]
            api_keys = config.settings.setdefault("api_keys", {})
            api_keys[provider] = value
            config.save()
            console.print(f"[green]API key for {provider} set successfully[/green]")
            return
            
        config.set_value(key, value)
        console.print(f"[green]Configuration value '{key}' set to '{value}'[/green]")
    except Exception as e:
        console.print(f"[red]Error setting configuration value: {e}[/red]")
        sys.exit(1)


@config_app.command("get")
def config_get(
    key: Optional[str] = typer.Argument(None, help="Configuration key to retrieve (dot notation)"),
) -> None:
    """Get configuration value(s)."""
    try:
        config = Config.load()
        
        if key is None:
            # Display all config values
            console.print(Panel.fit(
                Syntax(json.dumps(config.settings, indent=2, default=str), "json"),
                title="Current Configuration",
                border_style="blue"
            ))
            return
            
        value = config.get_value(key)
        if value is None:
            console.print(f"[yellow]Configuration key '{key}' not found[/yellow]")
            return
            
        # Format based on type
        if isinstance(value, dict):
            console.print(Panel.fit(
                Syntax(json.dumps(value, indent=2, default=str), "json"),
                title=f"Configuration: {key}",
                border_style="blue"
            ))
        else:
            console.print(f"[blue]{key}:[/blue] {value}")
    except Exception as e:
        console.print(f"[red]Error retrieving configuration: {e}[/red]")
        sys.exit(1)


@generate_app.command("run")
def generate_run(
    workspace_path: Path = typer.Argument(..., help="Path to workspace"),
    strategy: str = typer.Option(..., "--strategy", "-s", help="Generation strategy"),
    parameters_file: Optional[Path] = typer.Option(None, "--parameters-file", "-p", help="JSON file with parameters"),
    parameters_json: Optional[str] = typer.Option(None, "--parameters-json", "-j", help="JSON string with parameters"),
    output_dataset: str = typer.Option(..., "--output-dataset", "-o", help="Dataset name for output"),
    count: Optional[int] = typer.Option(None, "--count", "-c", help="Number of items to generate"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite dataset if it exists"),
) -> None:
    """Run a data generation job."""
    try:
        # Load workspace
        workspace = Workspace.load(workspace_path)
        console.print(f"[blue]Loaded workspace: {workspace.name}[/blue]")
        
        # Parse parameters
        parameters = {}
        if parameters_file:
            with open(parameters_file, "r") as f:
                parameters = json.load(f)
        elif parameters_json:
            parameters = json.loads(parameters_json)
        
        if count is not None:
            parameters["count"] = count
            
        # Import here to avoid circular imports when just using the CLI
        from opensynthetics.datagen.engine import Engine
        
        # Create generation engine
        engine = Engine(workspace)
        
        # Check if dataset exists
        dataset_exists = False
        try:
            workspace.get_dataset(output_dataset)
            dataset_exists = True
        except WorkspaceError:
            pass
            
        if dataset_exists and not overwrite:
            console.print(f"[red]Dataset '{output_dataset}' already exists. Use --overwrite to replace it.[/red]")
            sys.exit(1)
        elif dataset_exists:
            console.print(f"[yellow]Dataset '{output_dataset}' will be overwritten.[/yellow]")
            workspace.remove_dataset(output_dataset, confirm=True)
            
        # Run the generation
        with console.status(f"Generating data using strategy '{strategy}'..."):
            result = engine.generate(strategy=strategy, parameters=parameters, output_dataset=output_dataset)
        
        console.print(f"[green]Successfully generated {result.get('count', 'N/A')} items in dataset '{output_dataset}'[/green]")
        
        # Display table of sample items if available
        if result.get("sample_items"):
            console.print("[blue]Sample items:[/blue]")
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            
            # Dynamically create columns based on first item
            if result["sample_items"]:
                first_item = result["sample_items"][0]
                for key in first_item.keys():
                    table.add_column(key)
                
                # Add rows
                for item in result["sample_items"]:
                    row_values = [str(item.get(key, "")) for key in first_item.keys()]
                    table.add_row(*row_values)
                
                console.print(table)
    except Exception as e:
        console.print(f"[red]Error generating data: {e}[/red]")
        sys.exit(1)


@data_app.command("list")
def data_list(
    workspace_path: Path = typer.Argument(..., help="Path to workspace"),
) -> None:
    """List available datasets in the workspace."""
    try:
        workspace = Workspace.load(workspace_path)
        datasets = workspace.list_datasets()
        
        if not datasets:
            console.print("[yellow]No datasets found in workspace.[/yellow]")
            return
            
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Tags")
        table.add_column("Created At")
        
        for ds in datasets:
            table.add_row(
                ds["name"],
                ds["description"] or "-",
                ", ".join(ds["tags"]) if ds["tags"] else "-",
                ds["created_at"]
            )
            
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing datasets: {e}[/red]")
        sys.exit(1)


@data_app.command("describe")
def data_describe(
    workspace_path: Path = typer.Argument(..., help="Path to workspace"),
    dataset_name: str = typer.Argument(..., help="Dataset name"),
) -> None:
    """Show detailed information about a dataset."""
    try:
        workspace = Workspace.load(workspace_path)
        dataset = workspace.get_dataset(dataset_name)
        stats = dataset.get_stats()
        
        # Display dataset metadata
        console.print(Panel(f"[bold]Name:[/bold] {stats['name']}\n"
                            f"[bold]Description:[/bold] {stats.get('description', '-')}\n"
                            f"[bold]Tags:[/bold] {', '.join(stats.get('tags', []))}\n"
                            f"[bold]Created:[/bold] {stats.get('created_at', 'Unknown')}\n"
                            f"[bold]Updated:[/bold] {stats.get('updated_at', 'Unknown')}",
                      title="Dataset Information",
                      border_style="blue"))
        
        # Display tables
        tables = stats.get("tables", {})
        if not tables:
            console.print("[yellow]No tables found in dataset.[/yellow]")
            return
            
        for table_name, table_info in tables.items():
            if "error" in table_info:
                console.print(f"[red]Error with table '{table_name}': {table_info['error']}[/red]")
                continue
                
            console.print(f"\n[bold magenta]Table: {table_name}[/bold magenta]")
            console.print(f"Row count: {table_info.get('row_count', 'Unknown')}")
            
            # Display columns
            columns = table_info.get("columns", {})
            if columns:
                col_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
                col_table.add_column("Column")
                col_table.add_column("Type")
                
                for col_name, col_type in columns.items():
                    col_table.add_row(col_name, col_type)
                    
                console.print(col_table)
                
            # Sample data
            try:
                sample_rows = dataset.query(f'SELECT * FROM "{table_name}" LIMIT 5')
                if sample_rows:
                    console.print("[bold]Sample data:[/bold]")
                    sample_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
                    
                    # Add columns
                    for col in sample_rows[0].keys():
                        sample_table.add_column(col)
                        
                    # Add rows
                    for row in sample_rows:
                        sample_table.add_row(*[str(val)[:50] + ('...' if len(str(val)) > 50 else '') for val in row.values()])
                        
                    console.print(sample_table)
            except Exception as e:
                console.print(f"[yellow]Could not get sample data: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error describing dataset: {e}[/red]")
        sys.exit(1)


@datasette_app.command("serve")
def datasette_serve(
    workspace_path: Path = typer.Argument(..., help="Path to workspace"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8001, "--port", "-p", help="Port to bind to"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser", help="Open browser automatically"),
) -> None:
    """Start Datasette server to explore datasets."""
    try:
        workspace = Workspace.load(workspace_path)
        console.print(f"[blue]Starting Datasette server for workspace: {workspace.name}[/blue]")
        
        # This calls the implementation in Workspace class
        workspace.serve_datasette(host=host, port=port, open_browser=open_browser)
    except Exception as e:
        console.print(f"[red]Error starting Datasette server: {e}[/red]")
        sys.exit(1)


@api_cli_app.command("serve")
def api_serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start the API server."""
    try:
        console.print(f"[blue]Starting API server at http://{host}:{port}[/blue]")
        
        # Import the app here to avoid circular imports
        # and unnecessary dependencies when just using CLI commands
        global api_app
        if api_app is None:
            from opensynthetics.api.main import app as imported_app
            api_app = imported_app
        
        # Start Uvicorn server
        uvicorn.run(
            "opensynthetics.api.main:app",
            host=host,
            port=port,
            reload=reload,
        )
    except Exception as e:
        console.print(f"[red]Error starting API server: {e}[/red]")
        sys.exit(1)


@data_app.command("remove")
def data_remove(
    workspace_path: Path = typer.Argument(..., help="Path to workspace"),
    dataset_name: str = typer.Argument(..., help="Dataset name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation"),
) -> None:
    """Remove a dataset from the workspace."""
    try:
        workspace = Workspace.load(workspace_path)
        
        if not force:
            confirm = typer.confirm(f"Are you sure you want to remove dataset '{dataset_name}'? This cannot be undone.")
            if not confirm:
                console.print("[yellow]Operation cancelled.[/yellow]")
                return
                
        workspace.remove_dataset(dataset_name, confirm=True)
        console.print(f"[green]Dataset '{dataset_name}' successfully removed.[/green]")
    except Exception as e:
        console.print(f"[red]Error removing dataset: {e}[/red]")
        sys.exit(1)


@agent_app.command("chat")
def agent_chat(
    workspace_path: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Path to workspace"),
) -> None:
    """Start an interactive chat with an agent."""
    try:
        # This is a placeholder for the agent chat functionality
        # In a full implementation, you would:
        # 1. Import the agent module
        # 2. Create an agent instance
        # 3. Start an interactive session
        
        if workspace_path:
            workspace = Workspace.load(workspace_path)
            console.print(f"[blue]Using workspace: {workspace.name}[/blue]")
        
        console.print("[yellow]Agent chat functionality is not yet implemented.[/yellow]")
        console.print("[blue]This will be an interactive chat interface with an AI agent.[/blue]")
    except Exception as e:
        console.print(f"[red]Error starting agent chat: {e}[/red]")
        sys.exit(1)


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()