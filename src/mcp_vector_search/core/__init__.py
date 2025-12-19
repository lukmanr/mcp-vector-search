"""Core functionality for MCP Vector Search."""

from mcp_vector_search.core.git import (
    GitError,
    GitManager,
    GitNotAvailableError,
    GitNotRepoError,
    GitReferenceError,
)
from mcp_vector_search.core.registry import (
    MultiRepoContext,
    RepoInfo,
    RepoRegistry,
    get_central_index_path,
)

__all__ = [
    "GitError",
    "GitManager",
    "GitNotAvailableError",
    "GitNotRepoError",
    "GitReferenceError",
    "MultiRepoContext",
    "RepoInfo",
    "RepoRegistry",
    "get_central_index_path",
]
