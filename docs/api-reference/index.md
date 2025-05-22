# OpenSynthetics API Reference

This section provides detailed documentation for the OpenSynthetics API endpoints.

## Base URL

All API endpoints are accessible at:

```
http://localhost:8000/api/v1/
```

## Authentication

API requests require authentication using an API key. Add the key to your requests using the `X-API-Key` header.

## Endpoints

### Workspaces

- `GET /api/v1/workspaces` - List all workspaces
- `POST /api/v1/workspaces` - Create a new workspace
- `GET /api/v1/workspaces/{workspace_path}` - Get workspace details

### Datasets

- `GET /api/v1/workspaces/{workspace_path}/datasets` - List datasets in a workspace
- `GET /api/v1/workspaces/{workspace_path}/datasets/{dataset_name}` - Get dataset details

### Generation

- `GET /api/v1/strategies` - List available generation strategies
- `POST /api/v1/generate/jobs` - Create a generation job

### Configuration

- `GET /api/v1/config` - Get current configuration
- `POST /api/v1/config/api_keys/{provider}` - Set API key for a provider

## Response Format

All API responses are in JSON format and follow a consistent structure:

```json
{
  "data": { ... },  // The main response payload
  "error": null     // Error message, if any
}
```

For detailed information about each endpoint, refer to the specific endpoint documentation.
