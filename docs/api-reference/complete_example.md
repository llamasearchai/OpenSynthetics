# OpenSynthetics: Complete Working Example

This guide provides a full, step-by-step example for using OpenSynthetics via the API, CLI, Python, and Web UI.

---

## 1. Create a Workspace (API)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/workspaces" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{
    "name": "demo_workspace",
    "description": "Demo workspace for OpenSynthetics",
    "tags": ["demo", "example"]
  }'
```
**Expected Response:**
```json
{
  "name": "demo_workspace",
  "description": "Demo workspace for OpenSynthetics",
  "created_at": "2024-05-22T12:00:00Z",
  "updated_at": "2024-05-22T12:00:00Z"
}
```

---

## 2. Generate a Dataset (API)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{
    "workspace": "demo_workspace",
    "strategy": "tabular_random",
    "parameters": {"num_rows": 100, "num_columns": 5},
    "dataset": "demo_dataset"
  }'
```
**Expected Response:**
```json
{
  "count": 100,
  "strategy": "tabular_random",
  "output_dataset": "demo_dataset",
  "workspace": "demo_workspace",
  "timestamp": 1716384000,
  "sample_items": [
    {"col1": 42, "col2": "foo", ...},
    ...
  ]
}
```

---

## 3. Visualize in the Web UI

- Open [http://localhost:8000/ui/](http://localhost:8000/ui/)
- Go to **Workspaces** to see your workspace.
- Go to **Datasets** to see your dataset.
- Use **Visualize**/**Analytics** for charts and 3D views.

---

## 4. Python Example

```python
from opensynthetics.core.workspace import Workspace
ws = Workspace.create(name="demo_workspace", description="Demo workspace")
ds = ws.create_dataset(name="demo_dataset", description="Demo dataset")
data = [
    {"id": 1, "name": "Alice", "score": 95},
    {"id": 2, "name": "Bob", "score": 88},
    {"id": 3, "name": "Charlie", "score": 92},
]
ds.add_data(data)
print(ws.list_datasets())
```

---

## 5. CLI Example

```bash
opensynthetics generate --template tabular_random --output demo.csv --rows 100 --columns 5
```

---

## 6. UI Demo Walkthrough

1. Click **Create Workspace** on Dashboard/Workspaces.
2. Enter `demo_workspace` and description, submit.
3. Go to **Generate**, create a dataset in your workspace.
4. View/analyze in **Datasets**/**Visualize**.

---

## Troubleshooting
- If you see `Config file not found`, run `opensynthetics config set` or use the web UI to configure.
- For 401 errors, check your API key.
- For 404 on datasets, ensure you created the workspace first.

---

## More
- [API Reference](../api-reference/)
- [Web UI Guide](../web-ui/)
- [CLI Reference](../cli/) 