# Getting Started with OpenSynthetics

This guide will help you get started with OpenSynthetics for generating synthetic data.

![OpenSynthetics UI](../OpenSyntheticsUI.png)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Installing from PyPI

```bash
pip install opensynthetics
```

### Installing from source

```bash
git clone https://github.com/llamasearchai/opensynthetics.git
cd opensynthetics
pip install -e .
```

## Configuration

Before using OpenSynthetics, you need to configure your API keys for the LLM providers you want to use.

```bash
opensynthetics config set api_keys.openai your-openai-api-key
```

## Creating a Workspace

A workspace is a container for your synthetic data projects. Create a new workspace:

```bash
opensynthetics init my_workspace --description "My first synthetic data workspace"
```

This creates a new workspace in the default location (`~/opensynthetics_data/my_workspace`).

## Generating Data

OpenSynthetics supports various data generation strategies. Here's how to generate engineering problems:

```bash
opensynthetics generate run ~/opensynthetics_data/my_workspace \
  --strategy engineering_problems \
  --parameters-file params.json \
  --output-dataset mechanical_problems
```

Where `params.json` contains:

```json
{
  "domain": "mechanical",
  "count": 10,
  "difficulty": 5,
  "constraints": "The problems should involve static equilibrium and stress analysis."
}
```

## Using the Web UI

For a more user-friendly experience, OpenSynthetics provides a modern web interface.

### Starting the UI server

```bash
opensynthetics api serve --host 0.0.0.0 --port 8000
```

### Accessing the UI

Open your browser and navigate to http://localhost:8000/ui

### UI Features

The OpenSynthetics UI allows you to:

1. **Manage Workspaces**: Create, view, and organize your workspaces
2. **Generate Data**: Configure and run data generation jobs with various strategies
3. **Explore Datasets**: Browse and analyze your generated synthetic data
4. **Configure Settings**: Manage API keys and system settings

## Exploring Generated Data

### Using the CLI

List datasets in a workspace:

```bash
opensynthetics list datasets ~/opensynthetics_data/my_workspace
```

Export a dataset to CSV:

```bash
opensynthetics export dataset ~/opensynthetics_data/my_workspace mechanical_problems --format csv --output exported_data.csv
```

### Using Datasette

OpenSynthetics integrates with Datasette for a web-based data exploration experience:

```bash
opensynthetics datasette serve ~/opensynthetics_data/my_workspace
```

This will start a Datasette server, typically at http://localhost:8001, where you can explore your data using SQL queries.

## Next Steps

- Explore different [generation strategies](api-reference/strategies.md)
- Learn about [data formats and schemas](api-reference/schemas.md)
- Set up [automated generation workflows](development/automation.md)

## Troubleshooting

If you encounter issues:

- Check that your API keys are correctly configured
- Ensure you have sufficient permissions for the workspace directory
- Look at the logs using `opensynthetics logs`

For more help, open an issue on the [GitHub repository](https://github.com/llamasearchai/opensynthetics/issues).