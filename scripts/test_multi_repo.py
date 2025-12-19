#!/usr/bin/env python3
"""Test script for multi-repo functionality.

This script tests the multi-repo registry and database management
without needing a full MCP server connection.

Usage:
    uv run python scripts/test_multi_repo.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.registry import RepoInfo, RepoRegistry, get_central_index_path


async def test_registry():
    """Test the RepoRegistry functionality."""
    print("=" * 60)
    print("Testing Multi-Repo Registry")
    print("=" * 60)

    # Use a temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        central_path = Path(tmpdir) / "vector-search"

        print(f"\n1. Creating registry at: {central_path}")
        registry = RepoRegistry(central_index_path=central_path)

        # Test registering repos
        print("\n2. Registering test repositories...")

        # Create some fake repo directories
        repo1_path = Path(tmpdir) / "repo1"
        repo2_path = Path(tmpdir) / "repo2"
        repo1_path.mkdir()
        repo2_path.mkdir()

        # Create some marker files
        (repo1_path / "package.json").write_text('{"name": "repo1"}')
        (repo2_path / "pyproject.toml").write_text('[project]\nname = "repo2"')

        repo1 = registry.register_repo(
            repo_path=repo1_path,
            display_name="Test Repository 1",
            file_extensions=[".js", ".ts"],
            set_as_default=True,
        )
        print(f"   Registered: {repo1.display_name} (ID: {repo1.repo_id})")
        print(f"   Collection: {repo1.collection_name}")

        repo2 = registry.register_repo(
            repo_path=repo2_path,
            display_name="Test Repository 2",
            file_extensions=[".py"],
        )
        print(f"   Registered: {repo2.display_name} (ID: {repo2.repo_id})")
        print(f"   Collection: {repo2.collection_name}")

        # Test listing repos
        print("\n3. Listing repositories...")
        repos = registry.list_repos()
        print(f"   Found {len(repos)} repositories:")
        for repo in repos:
            default_marker = " (default)" if repo.repo_id == registry.default_repo_id else ""
            print(f"   - {repo.display_name}{default_marker}")

        # Test saving and loading
        print("\n4. Testing persistence...")
        registry.save()
        print(f"   Saved registry to: {registry.get_registry_file_path()}")

        # Load in a new instance
        loaded_registry = RepoRegistry.load(central_path)
        print(f"   Loaded registry with {len(loaded_registry.repos)} repositories")

        # Verify data
        assert len(loaded_registry.repos) == 2
        assert loaded_registry.default_repo_id == repo1.repo_id
        print("   ✓ Persistence verified")

        # Test getting repo by path
        print("\n5. Testing repo lookup by path...")
        found_repo = loaded_registry.get_repo_by_path(repo1_path)
        assert found_repo is not None
        assert found_repo.repo_id == repo1.repo_id
        print(f"   ✓ Found repo by path: {found_repo.display_name}")

        # Test updating stats
        print("\n6. Testing stats update...")
        loaded_registry.update_repo_stats(
            repo_id=repo1.repo_id,
            chunk_count=100,
            file_count=10,
            languages=["javascript", "typescript"],
            is_indexed=True,
        )
        updated_repo = loaded_registry.get_repo(repo1.repo_id)
        assert updated_repo.chunk_count == 100
        assert updated_repo.is_indexed is True
        print(f"   ✓ Stats updated: {updated_repo.chunk_count} chunks, {updated_repo.file_count} files")

        # Test unregistering
        print("\n7. Testing unregister...")
        success = loaded_registry.unregister_repo(repo2.repo_id)
        assert success is True
        assert len(loaded_registry.repos) == 1
        print(f"   ✓ Unregistered repo2, {len(loaded_registry.repos)} repo(s) remaining")

        # Test setting default
        print("\n8. Testing set_default_repo...")
        # Since we unregistered repo2, repo1 should still be default
        assert loaded_registry.default_repo_id == repo1.repo_id
        print(f"   ✓ Default repo is still: {repo1.display_name}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


async def test_central_index_path():
    """Test the central index path detection."""
    print("\n" + "=" * 60)
    print("Testing Central Index Path Detection")
    print("=" * 60)

    import os

    # Test default path
    print("\n1. Default path (no env var):")
    default_path = get_central_index_path()
    print(f"   {default_path}")
    assert "velocity" in str(default_path).lower() or ".velocity" in str(default_path)

    # Test with env var
    print("\n2. Custom path via MCP_CENTRAL_INDEX_PATH:")
    os.environ["MCP_CENTRAL_INDEX_PATH"] = "/tmp/custom-index"
    custom_path = get_central_index_path()
    print(f"   {custom_path}")
    # On macOS, /tmp is a symlink to /private/tmp, so compare resolved paths
    assert custom_path.resolve() == Path("/tmp/custom-index").resolve()

    # Cleanup
    del os.environ["MCP_CENTRAL_INDEX_PATH"]

    print("\n" + "=" * 60)
    print("Path detection tests passed! ✓")
    print("=" * 60)


async def main():
    """Run all tests."""
    await test_central_index_path()
    await test_registry()


if __name__ == "__main__":
    asyncio.run(main())

