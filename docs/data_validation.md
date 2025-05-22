# DataValidator

The `DataValidator` class is a core component of OpenSynthetics, responsible for managing and validating data against JSON schemas.

## Initialization

```python
from opensynthetics.data_ops.validation import DataValidator

validator = DataValidator()
```

## Managing Schemas

### Registering a Schema

You can register a schema directly as a Python dictionary:

```python
my_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name", "age"]
}
validator.register_schema("user_profile", my_schema)
```

### Loading a Schema from a File

Load a JSON schema from a `.json` file:

```python
validator.load_schema_from_file("product_schema", "/path/to/your/product_schema.json")
```
*   **Error Handling**: Raises `ValidationError` if the file cannot be loaded or is not valid JSON.

### Loading Schemas from a Directory

Load all `*.json` (or other specified extension) files from a directory. Schema names are derived from the filenames (without the extension).

```python
validator.load_schemas_from_directory("/path/to/schemas_directory")
# To load files with a different extension, e.g., .schema
validator.load_schemas_from_directory("/path/to/schemas_directory", file_extension=".schema")
```
*   **Error Handling**: Raises `ValidationError` if the directory path is invalid. Logs errors for individual file load failures but attempts to load other schemas.

### Listing Registered Schemas

Get a list of all currently registered schema names:

```python
schema_names = validator.list_schemas()
print(schema_names)
# Output: ['user_profile', 'product_schema', ...]
```

### Getting a Specific Schema

Retrieve a registered schema by its name:

```python
try:
    schema_content = validator.get_schema("user_profile")
    # print(json.dumps(schema_content, indent=2))
except ValidationError as e:
    print(e) # Schema not found
```

### Unregistering a Schema

Remove a schema from the validator:

```python
try:
    validator.unregister_schema("product_schema")
except ValidationError as e:
    print(e) # Schema not found
```

## Validating Data

### Validating a Single Data Record

Validate a Python dictionary against a registered schema:

```python
data_to_validate = {"name": "Alice", "age": 30}
is_valid, error_message = validator.validate(data_to_validate, "user_profile")

if is_valid:
    print("Data is valid.")
else:
    print(f"Data is invalid: {error_message}")
    # Example error_message: "Validation error at path 'age': -5 is less than the minimum of 0"
```
*   **Returns**: A tuple `(bool, Optional[str])`. The first element is `True` if valid, `False` otherwise. The second element is `None` if valid, or an error message string if invalid.
*   **Error Handling**: Raises `ValidationError` if the specified `schema_name` is not found.

### Validating a Batch of Data Records

Validate a list of Python dictionaries against a registered schema:

```python
data_batch = [
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": -5}, # Invalid age
    {"name": "David"} # Missing age
]

results = validator.validate_batch(data_batch, "user_profile")

for i, (is_valid, error_message) in enumerate(results):
    if is_valid:
        print(f"Record {i} is valid.")
    else:
        print(f"Record {i} is invalid: {error_message}")
```
*   **Returns**: A list of tuples, where each tuple is `(bool, Optional[str])` corresponding to each data record in the input list.
*   **Error Handling**: Raises `ValidationError` if the specified `schema_name` is not found.

## Exception Handling

*   `opensynthetics.core.exceptions.ValidationError`: This custom exception is raised for issues like schema not found, failure to load a schema, or invalid directory paths. 