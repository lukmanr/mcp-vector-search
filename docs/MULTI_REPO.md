# Multi-Repository Support

MCP Vector Search now supports indexing and searching across multiple code repositories. This is particularly useful for applications like Velocity PPM that need to manage semantic search indexes for multiple products/projects.

## Overview

Multi-repo mode provides:

- **Central registry**: A single registry file tracks all registered repositories
- **Isolated collections**: Each repository gets its own ChromaDB collection
- **Cross-repo search**: Search across all repositories simultaneously
- **Persistent state**: Repository configurations and index stats are persisted

## Enabling Multi-Repo Mode

Multi-repo mode can be enabled in several ways:

### 1. Environment Variable

```bash
export MCP_MULTI_REPO_MODE=true
```

### 2. Command Line Flag

```bash
python -m mcp_vector_search.mcp.server --multi-repo
```

### 3. Programmatic

```python
from mcp_vector_search.mcp.server import MCPVectorSearchServer

server = MCPVectorSearchServer(multi_repo_mode=True)
```

## Central Index Location

By default, the central index is stored at `~/.velocity/vector-search/`. This can be customized:

```bash
export MCP_CENTRAL_INDEX_PATH=/path/to/custom/index
```

The central index contains:
- `registry.json` - Repository metadata and configuration
- `chroma/` - Shared ChromaDB database with per-repo collections

## MCP Tools

### Repository Management

#### `register_repo`
Register a new repository for semantic search.

```json
{
  "repo_path": "/path/to/repo",
  "display_name": "My Project",
  "repo_id": "my-project",  // optional, auto-generated if not provided
  "file_extensions": [".py", ".js"],  // optional
  "set_as_default": true  // optional
}
```

#### `unregister_repo`
Remove a repository from the registry.

```json
{
  "repo_id": "my-project",
  "delete_index": true  // also remove indexed data
}
```

#### `list_repos`
List all registered repositories.

```json
{}
```

#### `set_default_repo`
Set the default repository for operations.

```json
{
  "repo_id": "my-project"
}
```

#### `get_repo_status`
Get status and statistics for a repository.

```json
{
  "repo_id": "my-project"  // optional, uses default if not specified
}
```

### Indexing

#### `index_repo`
Index or reindex a specific repository.

```json
{
  "repo_id": "my-project",  // optional, uses default if not specified
  "force": true  // force reindex even if already indexed
}
```

### Search

#### `search_code`
Search within a specific repository (repo_id is now supported).

```json
{
  "query": "authentication logic",
  "repo_id": "my-project",  // optional, uses default if not specified
  "limit": 10,
  "similarity_threshold": 0.3
}
```

#### `search_all_repos`
Search across all indexed repositories.

```json
{
  "query": "error handling",
  "limit": 5,  // per repository
  "similarity_threshold": 0.3
}
```

## Registry Structure

The registry is stored as JSON:

```json
{
  "version": "1.0.0",
  "repos": {
    "repo1_abc123": {
      "repo_id": "repo1_abc123",
      "repo_path": "/path/to/repo1",
      "display_name": "Repository 1",
      "collection_name": "code_search_repo1_abc123",
      "is_indexed": true,
      "last_indexed": 1703012345.678,
      "file_extensions": [".py", ".js"],
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "chunk_count": 1500,
      "file_count": 100,
      "languages": ["python", "javascript"],
      "created_at": 1703000000.0
    }
  },
  "default_repo_id": "repo1_abc123",
  "central_index_path": "/Users/user/.velocity/vector-search"
}
```

## Usage with Velocity

For Velocity PPM integration:

1. Set environment variables in the Electron app:
   ```
   MCP_MULTI_REPO_MODE=true
   MCP_CENTRAL_INDEX_PATH=~/.velocity/vector-search
   ```

2. When a product is opened, register its repository:
   ```
   register_repo(repo_path="/path/to/product/repo", display_name="Product Name")
   ```

3. Index the repository:
   ```
   index_repo(repo_id="product_xxx")
   ```

4. Search within the product:
   ```
   search_code(query="feature implementation", repo_id="product_xxx")
   ```

## Backwards Compatibility

Single-repo mode (the default) continues to work exactly as before. Multi-repo mode is opt-in and does not affect existing single-repo configurations.

## Python API

```python
from mcp_vector_search.core.registry import RepoRegistry, get_central_index_path

# Load or create registry
registry = RepoRegistry.load()

# Register a repository
repo_info = registry.register_repo(
    repo_path=Path("/path/to/repo"),
    display_name="My Project",
    file_extensions=[".py", ".js", ".ts"],
    set_as_default=True,
)

# List repositories
for repo in registry.list_repos():
    print(f"{repo.display_name}: {repo.chunk_count} chunks indexed")

# Update stats after indexing
registry.update_repo_stats(
    repo_id=repo_info.repo_id,
    chunk_count=1500,
    file_count=100,
    languages=["python", "javascript"],
    is_indexed=True,
)
```

