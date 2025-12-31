from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_CACHE: dict[str, Any] | None = None


def load_schema(version: str = "v1_0", path: str = "schemas/v1_0.json") -> dict[str, Any]:
    # Try different resolution strategies
    # 1. Absolute path
    schema_path = Path(path)
    if schema_path.is_absolute():
        if not schema_path.exists():
             schema_path = Path(__file__).resolve().parents[2] / path
    
    # 2. Relative to Repo Root
    else:
        repo_root = Path(__file__).resolve().parents[2]
        schema_path = repo_root / path

    # 3. Fallback to default
    if not schema_path.exists():
        fallback = Path(__file__).resolve().parents[2] / "schemas" / f"{version}.json"
        if fallback.exists():
            schema_path = fallback
            
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found. Checked: {schema_path}")
        
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    global _SCHEMA_CACHE
    _SCHEMA_CACHE = data
    return data


def required_cols(schema: dict[str, Any], table_name: str) -> list[str]:
    try:
        columns = schema["tables"][table_name]["required_columns"]
    except KeyError as exc:
        raise ValueError(f"Unknown table in schema: {table_name}") from exc
    return [col["name"] for col in columns]


def optional_cols(schema: dict[str, Any], table_name: str) -> list[str]:
    try:
        columns = schema["tables"][table_name].get("optional_columns", [])
    except KeyError as exc:
        raise ValueError(f"Unknown table in schema: {table_name}") from exc
    return [col["name"] for col in columns]


def require_columns(df, cols: list[str], table_name: str, strict: bool = True) -> bool:
    missing = [col for col in cols if col not in df.columns]
    if not missing:
        return True
    message = f"{table_name} missing required columns: {', '.join(missing)}"
    if strict:
        raise ValueError(message)
    print(f"WARNING: {message}")
    return False


def validate_table(df, table_name: str, strict: bool = True) -> bool:
    schema = _SCHEMA_CACHE or load_schema()
    cols = required_cols(schema, table_name)
    return require_columns(df, cols, table_name, strict=strict)
