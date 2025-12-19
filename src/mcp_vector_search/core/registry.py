"""Multi-repository registry for MCP Vector Search.

This module provides a central registry for managing multiple code repositories
with their own vector search indices. Each repository gets its own ChromaDB
collection, allowing for isolated indexing and cross-repo search.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from .exceptions import (
    ConfigurationError,
    ProjectNotFoundError,
)


def get_central_index_path() -> Path:
    """Get the central index path for multi-repo mode.

    Priority:
    1. MCP_CENTRAL_INDEX_PATH environment variable
    2. ~/.velocity/vector-search/ (default)

    Returns:
        Path to central index directory
    """
    env_path = os.getenv("MCP_CENTRAL_INDEX_PATH")
    if env_path:
        return Path(env_path).resolve()
    return Path.home() / ".velocity" / "vector-search"


class RepoInfo(BaseModel):
    """Information about a registered repository."""

    repo_id: str = Field(..., description="Unique identifier for the repository")
    repo_path: Path = Field(..., description="Absolute path to the repository root")
    display_name: str = Field(..., description="Human-readable name for the repository")
    collection_name: str = Field(..., description="ChromaDB collection name")
    is_indexed: bool = Field(default=False, description="Whether the repo has been indexed")
    last_indexed: float | None = Field(default=None, description="Unix timestamp of last indexing")
    file_extensions: list[str] = Field(
        default=[".py", ".js", ".ts", ".jsx", ".tsx"],
        description="File extensions to index",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    chunk_count: int = Field(default=0, description="Number of indexed chunks")
    file_count: int = Field(default=0, description="Number of indexed files")
    languages: list[str] = Field(default=[], description="Detected programming languages")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "repo_id": self.repo_id,
            "repo_path": str(self.repo_path),
            "display_name": self.display_name,
            "collection_name": self.collection_name,
            "is_indexed": self.is_indexed,
            "last_indexed": self.last_indexed,
            "file_extensions": self.file_extensions,
            "embedding_model": self.embedding_model,
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
            "languages": self.languages,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoInfo":
        """Create from dictionary."""
        data = data.copy()
        data["repo_path"] = Path(data["repo_path"])
        return cls(**data)


class RepoRegistry(BaseModel):
    """Central registry for managing multiple repositories.

    The registry is stored as a JSON file in the central index directory.
    Each repository gets a unique collection in the shared ChromaDB instance.
    """

    version: str = Field(default="1.0.0", description="Registry schema version")
    repos: dict[str, RepoInfo] = Field(default={}, description="Registered repositories")
    default_repo_id: str | None = Field(
        default=None, description="Default repository for operations"
    )
    central_index_path: Path = Field(
        default_factory=get_central_index_path,
        description="Central index directory path",
    )

    class Config:
        arbitrary_types_allowed = True

    def get_registry_file_path(self) -> Path:
        """Get the path to the registry JSON file."""
        return self.central_index_path / "registry.json"

    def get_chroma_persist_path(self) -> Path:
        """Get the path to the ChromaDB persistence directory."""
        return self.central_index_path / "chroma"

    def save(self) -> None:
        """Save the registry to disk."""
        self.central_index_path.mkdir(parents=True, exist_ok=True)
        registry_file = self.get_registry_file_path()

        data = {
            "version": self.version,
            "repos": {repo_id: info.to_dict() for repo_id, info in self.repos.items()},
            "default_repo_id": self.default_repo_id,
            "central_index_path": str(self.central_index_path),
        }

        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved registry to {registry_file}")

    @classmethod
    def load(cls, central_index_path: Path | None = None) -> "RepoRegistry":
        """Load the registry from disk.

        Args:
            central_index_path: Override for central index path. If None, uses default.

        Returns:
            Loaded or new RepoRegistry instance
        """
        if central_index_path is None:
            central_index_path = get_central_index_path()

        registry_file = central_index_path / "registry.json"

        if not registry_file.exists():
            logger.info(f"No existing registry found at {registry_file}, creating new one")
            return cls(central_index_path=central_index_path)

        try:
            with open(registry_file) as f:
                data = json.load(f)

            repos = {
                repo_id: RepoInfo.from_dict(info)
                for repo_id, info in data.get("repos", {}).items()
            }

            return cls(
                version=data.get("version", "1.0.0"),
                repos=repos,
                default_repo_id=data.get("default_repo_id"),
                central_index_path=central_index_path,
            )

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise ConfigurationError(f"Failed to load registry: {e}") from e

    def register_repo(
        self,
        repo_path: Path,
        display_name: str | None = None,
        repo_id: str | None = None,
        file_extensions: list[str] | None = None,
        embedding_model: str | None = None,
        set_as_default: bool = False,
    ) -> RepoInfo:
        """Register a new repository.

        Args:
            repo_path: Path to the repository root
            display_name: Human-readable name (defaults to directory name)
            repo_id: Unique ID (defaults to sanitized path hash)
            file_extensions: File extensions to index
            embedding_model: Embedding model to use
            set_as_default: Set this repo as the default

        Returns:
            Created RepoInfo
        """
        repo_path = repo_path.resolve()

        if not repo_path.exists():
            raise ProjectNotFoundError(f"Repository path does not exist: {repo_path}")

        # Generate repo_id from path if not provided
        if repo_id is None:
            import hashlib

            path_hash = hashlib.sha256(str(repo_path).encode()).hexdigest()[:12]
            repo_id = f"{repo_path.name}_{path_hash}"

        # Sanitize repo_id for use as collection name
        sanitized_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in repo_id)

        if display_name is None:
            display_name = repo_path.name

        # Create collection name (ChromaDB requires specific format)
        collection_name = f"code_search_{sanitized_id}"

        repo_info = RepoInfo(
            repo_id=repo_id,
            repo_path=repo_path,
            display_name=display_name,
            collection_name=collection_name,
            file_extensions=file_extensions or [".py", ".js", ".ts", ".jsx", ".tsx"],
            embedding_model=embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.repos[repo_id] = repo_info

        if set_as_default or len(self.repos) == 1:
            self.default_repo_id = repo_id

        self.save()
        logger.info(f"Registered repository: {display_name} ({repo_id}) at {repo_path}")

        return repo_info

    def unregister_repo(self, repo_id: str) -> bool:
        """Unregister a repository.

        Args:
            repo_id: Repository ID to unregister

        Returns:
            True if unregistered, False if not found
        """
        if repo_id not in self.repos:
            return False

        del self.repos[repo_id]

        if self.default_repo_id == repo_id:
            self.default_repo_id = next(iter(self.repos.keys()), None)

        self.save()
        logger.info(f"Unregistered repository: {repo_id}")

        return True

    def get_repo(self, repo_id: str | None = None) -> RepoInfo | None:
        """Get a repository by ID.

        Args:
            repo_id: Repository ID. If None, returns default repo.

        Returns:
            RepoInfo or None if not found
        """
        if repo_id is None:
            repo_id = self.default_repo_id

        if repo_id is None:
            return None

        return self.repos.get(repo_id)

    def get_repo_by_path(self, repo_path: Path) -> RepoInfo | None:
        """Get a repository by its path.

        Args:
            repo_path: Repository path

        Returns:
            RepoInfo or None if not found
        """
        repo_path = repo_path.resolve()
        for repo_info in self.repos.values():
            if repo_info.repo_path == repo_path:
                return repo_info
        return None

    def list_repos(self) -> list[RepoInfo]:
        """List all registered repositories.

        Returns:
            List of RepoInfo objects
        """
        return list(self.repos.values())

    def update_repo_stats(
        self,
        repo_id: str,
        chunk_count: int | None = None,
        file_count: int | None = None,
        languages: list[str] | None = None,
        is_indexed: bool | None = None,
    ) -> None:
        """Update repository statistics after indexing.

        Args:
            repo_id: Repository ID
            chunk_count: Number of indexed chunks
            file_count: Number of indexed files
            languages: Detected languages
            is_indexed: Whether indexing is complete
        """
        if repo_id not in self.repos:
            return

        repo = self.repos[repo_id]

        if chunk_count is not None:
            repo.chunk_count = chunk_count
        if file_count is not None:
            repo.file_count = file_count
        if languages is not None:
            repo.languages = languages
        if is_indexed is not None:
            repo.is_indexed = is_indexed
            if is_indexed:
                repo.last_indexed = time.time()

        self.save()

    def set_default_repo(self, repo_id: str) -> bool:
        """Set the default repository.

        Args:
            repo_id: Repository ID to set as default

        Returns:
            True if set, False if repo not found
        """
        if repo_id not in self.repos:
            return False

        self.default_repo_id = repo_id
        self.save()
        return True


@dataclass
class MultiRepoContext:
    """Context for multi-repo operations.

    Tracks the current repository context and provides
    convenient access to repo-specific resources.
    """

    registry: RepoRegistry
    current_repo_id: str | None = None
    _databases: dict = field(default_factory=dict)

    def get_current_repo(self) -> RepoInfo | None:
        """Get the current repository info."""
        return self.registry.get_repo(self.current_repo_id)

    def switch_repo(self, repo_id: str) -> bool:
        """Switch to a different repository.

        Args:
            repo_id: Repository ID to switch to

        Returns:
            True if switched, False if repo not found
        """
        if repo_id not in self.registry.repos:
            return False

        self.current_repo_id = repo_id
        return True

    def get_collection_name(self, repo_id: str | None = None) -> str | None:
        """Get the ChromaDB collection name for a repository.

        Args:
            repo_id: Repository ID. If None, uses current repo.

        Returns:
            Collection name or None if repo not found
        """
        repo = self.registry.get_repo(repo_id or self.current_repo_id)
        return repo.collection_name if repo else None

