"""MCP server implementation for MCP Vector Search."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ServerCapabilities,
    TextContent,
    Tool,
)

from ..analysis import (
    ProjectMetrics,
    SmellDetector,
    SmellSeverity,
)
from ..config.thresholds import ThresholdConfig
from ..core.database import ChromaVectorDatabase
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError
from ..core.indexer import SemanticIndexer
from ..core.project import ProjectManager
from ..core.registry import RepoInfo, RepoRegistry, get_central_index_path
from ..core.search import SemanticSearchEngine
from ..core.watcher import FileWatcher
from ..parsers.registry import ParserRegistry


class MCPVectorSearchServer:
    """MCP server for vector search functionality.

    Supports two modes:
    1. Single-repo mode (legacy): Works with a single project_root
    2. Multi-repo mode: Manages multiple repositories with a central registry

    Multi-repo mode is enabled when:
    - MCP_MULTI_REPO_MODE=true environment variable is set
    - OR no project_root is provided and MCP_PROJECT_ROOT is not set
    """

    def __init__(
        self,
        project_root: Path | None = None,
        enable_file_watching: bool | None = None,
        multi_repo_mode: bool | None = None,
        central_index_path: Path | None = None,
    ):
        """Initialize the MCP server.

        Args:
            project_root: Project root directory. If None in single-repo mode,
                         will auto-detect from environment or current directory.
            enable_file_watching: Enable file watching for automatic reindexing.
                                  If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
            multi_repo_mode: Enable multi-repo mode. If None, checks MCP_MULTI_REPO_MODE env var.
            central_index_path: Override for central index path in multi-repo mode.
        """
        # Determine if multi-repo mode should be enabled
        if multi_repo_mode is None:
            env_value = os.getenv("MCP_MULTI_REPO_MODE", "false").lower()
            multi_repo_mode = env_value in ("true", "1", "yes", "on")

        self.multi_repo_mode = multi_repo_mode
        self.central_index_path = central_index_path or get_central_index_path()

        # Multi-repo mode resources
        self.registry: RepoRegistry | None = None
        self._repo_databases: dict[str, ChromaVectorDatabase] = {}
        self._repo_search_engines: dict[str, SemanticSearchEngine] = {}
        self._repo_indexers: dict[str, SemanticIndexer] = {}
        self._embedding_function = None

        # Single-repo mode resources (legacy)
        self.project_root: Path | None = None
        self.project_manager: ProjectManager | None = None
        self.search_engine: SemanticSearchEngine | None = None
        self.file_watcher: FileWatcher | None = None
        self.indexer: SemanticIndexer | None = None
        self.database: ChromaVectorDatabase | None = None

        if not multi_repo_mode:
            # Auto-detect project root from environment or current directory
            if project_root is None:
                # Priority 1: MCP_PROJECT_ROOT (new standard)
                # Priority 2: PROJECT_ROOT (legacy)
                # Priority 3: Current working directory
                env_project_root = os.getenv("MCP_PROJECT_ROOT") or os.getenv(
                    "PROJECT_ROOT"
                )
                if env_project_root:
                    project_root = Path(env_project_root).resolve()
                    logger.info(f"Using project root from environment: {project_root}")
                else:
                    project_root = Path.cwd()
                    logger.info(f"Using current directory as project root: {project_root}")

            self.project_root = project_root
            self.project_manager = ProjectManager(self.project_root)

        self._initialized = False

        # Determine if file watching should be enabled
        if enable_file_watching is None:
            # Check environment variable, default to True
            env_value = os.getenv("MCP_ENABLE_FILE_WATCHING", "true").lower()
            self.enable_file_watching = env_value in ("true", "1", "yes", "on")
        else:
            self.enable_file_watching = enable_file_watching

        if multi_repo_mode:
            logger.info(f"Multi-repo mode enabled. Central index: {self.central_index_path}")
        else:
            logger.info(f"Single-repo mode. Project root: {self.project_root}")

    async def initialize(self) -> None:
        """Initialize the server.

        In multi-repo mode, this loads the registry and shared embedding function.
        In single-repo mode, this initializes the search engine and database.
        """
        if self._initialized:
            return

        if self.multi_repo_mode:
            await self._initialize_multi_repo()
        else:
            await self._initialize_single_repo()

    async def _initialize_multi_repo(self) -> None:
        """Initialize multi-repo mode resources."""
        try:
            # Load or create registry
            self.registry = RepoRegistry.load(self.central_index_path)
            logger.info(f"Loaded registry with {len(self.registry.repos)} repositories")

            # Create shared embedding function
            self._embedding_function, _ = create_embedding_function(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            self._initialized = True
            logger.info("Multi-repo MCP server initialized")

        except Exception as e:
            logger.error(f"Failed to initialize multi-repo server: {e}")
            raise

    async def _initialize_single_repo(self) -> None:
        """Initialize single-repo mode resources (legacy)."""
        try:
            # Load project configuration
            config = self.project_manager.load_config()

            # Setup embedding function
            embedding_function, _ = create_embedding_function(
                model_name=config.embedding_model
            )

            # Setup database
            self.database = ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
            )

            # Initialize database
            await self.database.__aenter__()

            # Setup search engine
            self.search_engine = SemanticSearchEngine(
                database=self.database, project_root=self.project_root
            )

            # Setup indexer for file watching
            if self.enable_file_watching:
                self.indexer = SemanticIndexer(
                    database=self.database,
                    project_root=self.project_root,
                    config=config,
                )

                # Setup file watcher
                self.file_watcher = FileWatcher(
                    project_root=self.project_root,
                    config=config,
                    indexer=self.indexer,
                    database=self.database,
                )

                # Start file watching
                await self.file_watcher.start()
                logger.info("File watching enabled for automatic reindexing")
            else:
                logger.info("File watching disabled")

            self._initialized = True
            logger.info(f"MCP server initialized for project: {self.project_root}")

        except ProjectNotFoundError:
            logger.error(f"Project not initialized at {self.project_root}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise

    async def _get_repo_database(self, repo_id: str) -> ChromaVectorDatabase | None:
        """Get or create a database for a specific repository.

        Args:
            repo_id: Repository ID

        Returns:
            ChromaVectorDatabase for the repo, or None if repo not found
        """
        if not self.multi_repo_mode or not self.registry:
            return None

        # Return cached database if available
        if repo_id in self._repo_databases:
            return self._repo_databases[repo_id]

        # Get repo info
        repo_info = self.registry.get_repo(repo_id)
        if not repo_info:
            return None

        # Create database for this repo
        chroma_path = self.registry.get_chroma_persist_path()
        database = ChromaVectorDatabase(
            persist_directory=chroma_path,
            embedding_function=self._embedding_function,
            collection_name=repo_info.collection_name,
        )

        # Initialize database
        await database.__aenter__()

        self._repo_databases[repo_id] = database
        return database

    async def _get_repo_search_engine(
        self, repo_id: str
    ) -> tuple[SemanticSearchEngine | None, RepoInfo | None]:
        """Get or create a search engine for a specific repository.

        Args:
            repo_id: Repository ID

        Returns:
            Tuple of (SemanticSearchEngine, RepoInfo) or (None, None) if not found
        """
        if not self.multi_repo_mode or not self.registry:
            return None, None

        repo_info = self.registry.get_repo(repo_id)
        if not repo_info:
            return None, None

        # Return cached search engine if available
        if repo_id in self._repo_search_engines:
            return self._repo_search_engines[repo_id], repo_info

        # Get database
        database = await self._get_repo_database(repo_id)
        if not database:
            return None, None

        # Create search engine
        search_engine = SemanticSearchEngine(
            database=database,
            project_root=repo_info.repo_path,
        )

        self._repo_search_engines[repo_id] = search_engine
        return search_engine, repo_info

    def _resolve_repo_id(self, args: dict[str, Any]) -> str | None:
        """Resolve repo_id from arguments, falling back to default.

        Args:
            args: Tool arguments that may contain repo_id

        Returns:
            Resolved repo_id or None if not in multi-repo mode
        """
        if not self.multi_repo_mode or not self.registry:
            return None

        repo_id = args.get("repo_id")
        if repo_id:
            return repo_id

        return self.registry.default_repo_id

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.multi_repo_mode:
            # Cleanup all repo databases
            for repo_id, database in self._repo_databases.items():
                if hasattr(database, "__aexit__"):
                    await database.__aexit__(None, None, None)
                    logger.debug(f"Closed database for repo: {repo_id}")

            self._repo_databases.clear()
            self._repo_search_engines.clear()
            self._repo_indexers.clear()
        else:
            # Stop file watcher if running
            if self.file_watcher and self.file_watcher.is_running:
                logger.info("Stopping file watcher...")
                await self.file_watcher.stop()
                self.file_watcher = None

            # Cleanup database connection
            if self.database and hasattr(self.database, "__aexit__"):
                await self.database.__aexit__(None, None, None)
                self.database = None

            # Clear references
            self.search_engine = None
            self.indexer = None

        self._initialized = False
        logger.info("MCP server cleanup completed")

    def get_tools(self) -> list[Tool]:
        """Get available MCP tools."""
        # Common repo_id property for multi-repo tools
        repo_id_prop = {
            "type": "string",
            "description": "Repository ID (multi-repo mode only). Uses default repo if not specified.",
        }

        tools = [
            Tool(
                name="search_code",
                description="Search for code using semantic similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant code",
                        },
                        "repo_id": repo_id_prop,
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by file extensions (e.g., ['.py', '.js'])",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by programming language",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Filter by function name",
                        },
                        "class_name": {
                            "type": "string",
                            "description": "Filter by class name",
                        },
                        "files": {
                            "type": "string",
                            "description": "Filter by file patterns (e.g., '*.py' or 'src/*.js')",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search_similar",
                description="Find code similar to a specific file or function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to find similar code for",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Optional function name within the file",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="search_context",
                description="Search for code based on contextual description",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Contextual description of what you're looking for",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Areas to focus on (e.g., ['security', 'authentication'])",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["description"],
                },
            ),
            Tool(
                name="get_project_status",
                description="Get project indexing status and statistics",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="index_project",
                description="Index or reindex the project codebase",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force reindexing even if index exists",
                            "default": False,
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to index (e.g., ['.py', '.js'])",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="analyze_project",
                description="Returns project-wide metrics summary",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threshold_preset": {
                            "type": "string",
                            "description": "Threshold preset: 'strict', 'standard', or 'relaxed'",
                            "enum": ["strict", "standard", "relaxed"],
                            "default": "standard",
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Output format: 'summary' or 'detailed'",
                            "enum": ["summary", "detailed"],
                            "default": "summary",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="analyze_file",
                description="Returns file-level metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to analyze (relative or absolute)",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="find_smells",
                description="Returns list of code smells",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "smell_type": {
                            "type": "string",
                            "description": "Filter by smell type: 'Long Method', 'Deep Nesting', 'Long Parameter List', 'God Class', 'Complex Method'",
                            "enum": [
                                "Long Method",
                                "Deep Nesting",
                                "Long Parameter List",
                                "God Class",
                                "Complex Method",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "description": "Filter by severity level",
                            "enum": ["info", "warning", "error"],
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="get_complexity_hotspots",
                description="Returns top N most complex functions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of hotspots to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="check_circular_dependencies",
                description="Returns circular dependency cycles",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="interpret_analysis",
                description="Interpret analysis results with natural language explanations and recommendations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_json": {
                            "type": "string",
                            "description": "JSON string from analyze command with --include-context",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus area: 'summary', 'recommendations', or 'priorities'",
                            "enum": ["summary", "recommendations", "priorities"],
                            "default": "summary",
                        },
                        "verbosity": {
                            "type": "string",
                            "description": "Verbosity level: 'brief', 'normal', or 'detailed'",
                            "enum": ["brief", "normal", "detailed"],
                            "default": "normal",
                        },
                    },
                    "required": ["analysis_json"],
                },
            ),
        ]

        # Add multi-repo management tools (only in multi-repo mode)
        if self.multi_repo_mode:
            tools.extend(
                [
                    Tool(
                        name="register_repo",
                        description="Register a new repository for semantic code search",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repo_path": {
                                    "type": "string",
                                    "description": "Absolute path to the repository root",
                                },
                                "display_name": {
                                    "type": "string",
                                    "description": "Human-readable name for the repository",
                                },
                                "repo_id": {
                                    "type": "string",
                                    "description": "Unique identifier (auto-generated if not provided)",
                                },
                                "file_extensions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "File extensions to index (e.g., ['.py', '.js'])",
                                },
                                "set_as_default": {
                                    "type": "boolean",
                                    "description": "Set as the default repository",
                                    "default": False,
                                },
                            },
                            "required": ["repo_path"],
                        },
                    ),
                    Tool(
                        name="unregister_repo",
                        description="Unregister a repository and remove its index",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repo_id": {
                                    "type": "string",
                                    "description": "Repository ID to unregister",
                                },
                                "delete_index": {
                                    "type": "boolean",
                                    "description": "Also delete the indexed data",
                                    "default": True,
                                },
                            },
                            "required": ["repo_id"],
                        },
                    ),
                    Tool(
                        name="list_repos",
                        description="List all registered repositories",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    ),
                    Tool(
                        name="set_default_repo",
                        description="Set the default repository for operations",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repo_id": {
                                    "type": "string",
                                    "description": "Repository ID to set as default",
                                },
                            },
                            "required": ["repo_id"],
                        },
                    ),
                    Tool(
                        name="get_repo_status",
                        description="Get status and statistics for a specific repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repo_id": repo_id_prop,
                            },
                            "required": [],
                        },
                    ),
                    Tool(
                        name="index_repo",
                        description="Index or reindex a specific repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repo_id": repo_id_prop,
                                "force": {
                                    "type": "boolean",
                                    "description": "Force reindexing even if index exists",
                                    "default": False,
                                },
                            },
                            "required": [],
                        },
                    ),
                    Tool(
                        name="search_all_repos",
                        description="Search across all registered repositories",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find relevant code",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum results per repository",
                                    "default": 5,
                                    "minimum": 1,
                                    "maximum": 20,
                                },
                                "similarity_threshold": {
                                    "type": "number",
                                    "description": "Minimum similarity threshold (0.0-1.0)",
                                    "default": 0.3,
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                ]
            )

        return tools

    def get_capabilities(self) -> ServerCapabilities:
        """Get server capabilities."""
        return ServerCapabilities(tools={"listChanged": True}, logging={})

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        # Tools that don't require initialization
        no_init_tools = {"interpret_analysis", "register_repo", "list_repos"}

        if request.params.name not in no_init_tools and not self._initialized:
            await self.initialize()

        try:
            # Standard tools
            if request.params.name == "search_code":
                return await self._search_code(request.params.arguments)
            elif request.params.name == "search_similar":
                return await self._search_similar(request.params.arguments)
            elif request.params.name == "search_context":
                return await self._search_context(request.params.arguments)
            elif request.params.name == "get_project_status":
                return await self._get_project_status(request.params.arguments)
            elif request.params.name == "index_project":
                return await self._index_project(request.params.arguments)
            elif request.params.name == "analyze_project":
                return await self._analyze_project(request.params.arguments)
            elif request.params.name == "analyze_file":
                return await self._analyze_file(request.params.arguments)
            elif request.params.name == "find_smells":
                return await self._find_smells(request.params.arguments)
            elif request.params.name == "get_complexity_hotspots":
                return await self._get_complexity_hotspots(request.params.arguments)
            elif request.params.name == "check_circular_dependencies":
                return await self._check_circular_dependencies(request.params.arguments)
            elif request.params.name == "interpret_analysis":
                return await self._interpret_analysis(request.params.arguments)
            # Multi-repo management tools
            elif request.params.name == "register_repo":
                return await self._register_repo(request.params.arguments)
            elif request.params.name == "unregister_repo":
                return await self._unregister_repo(request.params.arguments)
            elif request.params.name == "list_repos":
                return await self._list_repos(request.params.arguments)
            elif request.params.name == "set_default_repo":
                return await self._set_default_repo(request.params.arguments)
            elif request.params.name == "get_repo_status":
                return await self._get_repo_status(request.params.arguments)
            elif request.params.name == "index_repo":
                return await self._index_repo(request.params.arguments)
            elif request.params.name == "search_all_repos":
                return await self._search_all_repos(request.params.arguments)
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"Unknown tool: {request.params.name}"
                        )
                    ],
                    isError=True,
                )
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Tool execution failed: {str(e)}")
                ],
                isError=True,
            )

    async def _search_code(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_code tool call."""
        query = args.get("query", "")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)
        file_extensions = args.get("file_extensions")
        language = args.get("language")
        function_name = args.get("function_name")
        class_name = args.get("class_name")
        files = args.get("files")

        if not query:
            return CallToolResult(
                content=[TextContent(type="text", text="Query parameter is required")],
                isError=True,
            )

        # Build filters
        filters = {}
        if file_extensions:
            filters["file_extension"] = {"$in": file_extensions}
        if language:
            filters["language"] = language
        if function_name:
            filters["function_name"] = function_name
        if class_name:
            filters["class_name"] = class_name
        if files:
            # Convert file pattern to filter (simplified)
            filters["file_pattern"] = files

        # Get the appropriate search engine
        if self.multi_repo_mode:
            repo_id = self._resolve_repo_id(args)
            if not repo_id:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No repository specified and no default repository set. "
                            "Use repo_id parameter or set a default with set_default_repo.",
                        )
                    ],
                    isError=True,
                )

            search_engine, repo_info = await self._get_repo_search_engine(repo_id)
            if not search_engine:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Repository not found or not indexed: {repo_id}",
                        )
                    ],
                    isError=True,
                )
        else:
            search_engine = self.search_engine
            repo_info = None

        # Perform search
        results = await search_engine.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            filters=filters,
        )

        # Format results
        if not results:
            response_text = f"No results found for query: '{query}'"
        else:
            response_lines = [f"Found {len(results)} results for query: '{query}'\n"]

            for i, result in enumerate(results, 1):
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                response_lines.append(result.content)
                response_lines.append("```\n")

            response_text = "\n".join(response_lines)

        return CallToolResult(content=[TextContent(type="text", text=response_text)])

    async def _get_project_status(self, args: dict[str, Any]) -> CallToolResult:
        """Handle get_project_status tool call."""
        try:
            config = self.project_manager.load_config()

            # Get database stats
            if self.search_engine:
                stats = await self.search_engine.database.get_stats()

                status_info = {
                    "project_root": str(config.project_root),
                    "index_path": str(config.index_path),
                    "file_extensions": config.file_extensions,
                    "embedding_model": config.embedding_model,
                    "languages": config.languages,
                    "total_chunks": stats.total_chunks,
                    "total_files": stats.total_files,
                    "index_size": (
                        f"{stats.index_size_mb:.2f} MB"
                        if hasattr(stats, "index_size_mb")
                        else "Unknown"
                    ),
                }
            else:
                status_info = {
                    "project_root": str(config.project_root),
                    "index_path": str(config.index_path),
                    "file_extensions": config.file_extensions,
                    "embedding_model": config.embedding_model,
                    "languages": config.languages,
                    "status": "Not indexed",
                }

            response_text = "# Project Status\n\n"
            response_text += f"**Project Root:** {status_info['project_root']}\n"
            response_text += f"**Index Path:** {status_info['index_path']}\n"
            response_text += (
                f"**File Extensions:** {', '.join(status_info['file_extensions'])}\n"
            )
            response_text += f"**Embedding Model:** {status_info['embedding_model']}\n"
            response_text += f"**Languages:** {', '.join(status_info['languages'])}\n"

            if "total_chunks" in status_info:
                response_text += f"**Total Chunks:** {status_info['total_chunks']}\n"
                response_text += f"**Total Files:** {status_info['total_files']}\n"
                response_text += f"**Index Size:** {status_info['index_size']}\n"
            else:
                response_text += f"**Status:** {status_info['status']}\n"

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except ProjectNotFoundError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first.",
                    )
                ],
                isError=True,
            )

    async def _index_project(self, args: dict[str, Any]) -> CallToolResult:
        """Handle index_project tool call."""
        force = args.get("force", False)
        file_extensions = args.get("file_extensions")

        try:
            # Import indexing functionality
            from ..cli.commands.index import run_indexing

            # Run indexing
            await run_indexing(
                project_root=self.project_root,
                force_reindex=force,
                extensions=file_extensions,
                show_progress=False,  # Disable progress for MCP
            )

            # Reinitialize search engine after indexing
            await self.cleanup()
            await self.initialize()

            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text="Project indexing completed successfully!"
                    )
                ]
            )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Indexing failed: {str(e)}")],
                isError=True,
            )

    async def _search_similar(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_similar tool call."""
        file_path = args.get("file_path", "")
        function_name = args.get("function_name")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)

        if not file_path:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="file_path parameter is required")
                ],
                isError=True,
            )

        try:
            from pathlib import Path

            # Convert to Path object
            file_path_obj = Path(file_path)
            if not file_path_obj.is_absolute():
                file_path_obj = self.project_root / file_path_obj

            if not file_path_obj.exists():
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"File not found: {file_path}")
                    ],
                    isError=True,
                )

            # Run similar search
            results = await self.search_engine.search_similar(
                file_path=file_path_obj,
                function_name=function_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"No similar code found for {file_path}"
                        )
                    ]
                )

            response_lines = [
                f"Found {len(results)} similar code snippets for {file_path}\n"
            ]

            for i, result in enumerate(results, 1):
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Show more of the content for similar search
                content_preview = (
                    result.content[:500]
                    if len(result.content) > 500
                    else result.content
                )
                response_lines.append(
                    content_preview + ("..." if len(result.content) > 500 else "")
                )
                response_lines.append("```\n")

            result_text = "\n".join(response_lines)

            return CallToolResult(content=[TextContent(type="text", text=result_text)])

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Similar search failed: {str(e)}")
                ],
                isError=True,
            )

    async def _search_context(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_context tool call."""
        description = args.get("description", "")
        focus_areas = args.get("focus_areas")
        limit = args.get("limit", 10)

        if not description:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="description parameter is required")
                ],
                isError=True,
            )

        try:
            # Perform context search
            results = await self.search_engine.search_by_context(
                context_description=description, focus_areas=focus_areas, limit=limit
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"No contextually relevant code found for: {description}",
                        )
                    ]
                )

            response_lines = [
                f"Found {len(results)} contextually relevant code snippets"
            ]
            if focus_areas:
                response_lines[0] += f" (focus: {', '.join(focus_areas)})"
            response_lines[0] += f" for: {description}\n"

            for i, result in enumerate(results, 1):
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Show more of the content for context search
                content_preview = (
                    result.content[:500]
                    if len(result.content) > 500
                    else result.content
                )
                response_lines.append(
                    content_preview + ("..." if len(result.content) > 500 else "")
                )
                response_lines.append("```\n")

            result_text = "\n".join(response_lines)

            return CallToolResult(content=[TextContent(type="text", text=result_text)])

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Context search failed: {str(e)}")
                ],
                isError=True,
            )

    async def _analyze_project(self, args: dict[str, Any]) -> CallToolResult:
        """Handle analyze_project tool call."""
        threshold_preset = args.get("threshold_preset", "standard")
        output_format = args.get("output_format", "summary")

        try:
            # Load threshold configuration based on preset
            threshold_config = self._get_threshold_config(threshold_preset)

            # Run analysis using CLI analyze logic
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            if not files_to_analyze:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No analyzable files found in project",
                        )
                    ],
                    isError=True,
                )

            # Analyze files
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception as e:
                    logger.debug(f"Failed to analyze {file_path}: {e}")
                    continue

            project_metrics.compute_aggregates()

            # Detect code smells
            smell_detector = SmellDetector(thresholds=threshold_config)
            all_smells = []
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

            # Format response
            if output_format == "detailed":
                # Return full JSON output
                import json

                output = project_metrics.to_summary()
                output["smells"] = {
                    "total": len(all_smells),
                    "by_severity": {
                        "error": sum(
                            1 for s in all_smells if s.severity == SmellSeverity.ERROR
                        ),
                        "warning": sum(
                            1 for s in all_smells if s.severity == SmellSeverity.WARNING
                        ),
                        "info": sum(
                            1 for s in all_smells if s.severity == SmellSeverity.INFO
                        ),
                    },
                }
                response_text = json.dumps(output, indent=2)
            else:
                # Return summary
                summary = project_metrics.to_summary()
                response_lines = [
                    "# Project Analysis Summary\n",
                    f"**Project Root:** {summary['project_root']}",
                    f"**Total Files:** {summary['total_files']}",
                    f"**Total Functions:** {summary['total_functions']}",
                    f"**Total Classes:** {summary['total_classes']}",
                    f"**Average File Complexity:** {summary['avg_file_complexity']}\n",
                    "## Complexity Distribution",
                ]

                dist = summary["complexity_distribution"]
                for grade in ["A", "B", "C", "D", "F"]:
                    response_lines.append(f"- Grade {grade}: {dist[grade]} chunks")

                response_lines.extend(
                    [
                        "\n## Health Metrics",
                        f"- Average Health Score: {summary['health_metrics']['avg_health_score']:.2f}",
                        f"- Files Needing Attention: {summary['health_metrics']['files_needing_attention']}",
                        "\n## Code Smells",
                        f"- Total: {len(all_smells)}",
                        f"- Errors: {sum(1 for s in all_smells if s.severity == SmellSeverity.ERROR)}",
                        f"- Warnings: {sum(1 for s in all_smells if s.severity == SmellSeverity.WARNING)}",
                        f"- Info: {sum(1 for s in all_smells if s.severity == SmellSeverity.INFO)}",
                    ]
                )

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Project analysis failed: {str(e)}")
                ],
                isError=True,
            )

    async def _analyze_file(self, args: dict[str, Any]) -> CallToolResult:
        """Handle analyze_file tool call."""
        file_path_str = args.get("file_path", "")

        if not file_path_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="file_path parameter is required")
                ],
                isError=True,
            )

        try:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                file_path = self.project_root / file_path

            if not file_path.exists():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"File not found: {file_path_str}"
                        )
                    ],
                    isError=True,
                )

            # Analyze single file
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )
            from ..cli.commands.analyze import _analyze_file

            parser_registry = ParserRegistry()
            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            file_metrics = await _analyze_file(file_path, parser_registry, collectors)

            if not file_metrics:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Unable to analyze file: {file_path_str}",
                        )
                    ],
                    isError=True,
                )

            # Detect smells
            smell_detector = SmellDetector()
            smells = smell_detector.detect_all(file_metrics, str(file_path))

            # Format response
            response_lines = [
                f"# File Analysis: {file_path.name}\n",
                f"**Path:** {file_path}",
                f"**Total Lines:** {file_metrics.total_lines}",
                f"**Code Lines:** {file_metrics.code_lines}",
                f"**Comment Lines:** {file_metrics.comment_lines}",
                f"**Functions:** {file_metrics.function_count}",
                f"**Classes:** {file_metrics.class_count}",
                f"**Methods:** {file_metrics.method_count}\n",
                "## Complexity Metrics",
                f"- Total Complexity: {file_metrics.total_complexity}",
                f"- Average Complexity: {file_metrics.avg_complexity:.2f}",
                f"- Max Complexity: {file_metrics.max_complexity}",
                f"- Health Score: {file_metrics.health_score:.2f}\n",
            ]

            if smells:
                response_lines.append(f"## Code Smells ({len(smells)})\n")
                for smell in smells[:10]:  # Show top 10
                    response_lines.append(
                        f"- [{smell.severity.value.upper()}] {smell.name}: {smell.description}"
                    )
                if len(smells) > 10:
                    response_lines.append(f"\n... and {len(smells) - 10} more")
            else:
                response_lines.append("## Code Smells\n- None detected")

            response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"File analysis failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"File analysis failed: {str(e)}")
                ],
                isError=True,
            )

    async def _find_smells(self, args: dict[str, Any]) -> CallToolResult:
        """Handle find_smells tool call."""
        smell_type_filter = args.get("smell_type")
        severity_filter = args.get("severity")

        try:
            # Run full project analysis
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception:  # nosec B112 - intentional skip of unparseable files
                    continue

            # Detect all smells
            smell_detector = SmellDetector()
            all_smells = []
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

            # Apply filters
            filtered_smells = all_smells

            if smell_type_filter:
                filtered_smells = [
                    s for s in filtered_smells if s.name == smell_type_filter
                ]

            if severity_filter:
                severity_enum = SmellSeverity(severity_filter)
                filtered_smells = [
                    s for s in filtered_smells if s.severity == severity_enum
                ]

            # Format response
            if not filtered_smells:
                filter_desc = []
                if smell_type_filter:
                    filter_desc.append(f"type={smell_type_filter}")
                if severity_filter:
                    filter_desc.append(f"severity={severity_filter}")
                filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
                response_text = f"No code smells found{filter_str}"
            else:
                response_lines = [f"# Code Smells Found: {len(filtered_smells)}\n"]

                # Group by severity
                by_severity = {
                    "error": [
                        s for s in filtered_smells if s.severity == SmellSeverity.ERROR
                    ],
                    "warning": [
                        s
                        for s in filtered_smells
                        if s.severity == SmellSeverity.WARNING
                    ],
                    "info": [
                        s for s in filtered_smells if s.severity == SmellSeverity.INFO
                    ],
                }

                for severity_level in ["error", "warning", "info"]:
                    smells = by_severity[severity_level]
                    if smells:
                        response_lines.append(
                            f"## {severity_level.upper()} ({len(smells)})\n"
                        )
                        for smell in smells[:20]:  # Show top 20 per severity
                            response_lines.append(
                                f"- **{smell.name}** at `{smell.location}`"
                            )
                            response_lines.append(f"  {smell.description}")
                            if smell.suggestion:
                                response_lines.append(
                                    f"  *Suggestion: {smell.suggestion}*"
                                )
                            response_lines.append("")

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Smell detection failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Smell detection failed: {str(e)}")
                ],
                isError=True,
            )

    async def _get_complexity_hotspots(self, args: dict[str, Any]) -> CallToolResult:
        """Handle get_complexity_hotspots tool call."""
        limit = args.get("limit", 10)

        try:
            # Run full project analysis
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception:  # nosec B112 - intentional skip of unparseable files
                    continue

            # Get top N complex files
            hotspots = project_metrics.get_hotspots(limit=limit)

            # Format response
            if not hotspots:
                response_text = "No complexity hotspots found"
            else:
                response_lines = [f"# Top {len(hotspots)} Complexity Hotspots\n"]

                for i, file_metrics in enumerate(hotspots, 1):
                    response_lines.extend(
                        [
                            f"## {i}. {Path(file_metrics.file_path).name}",
                            f"**Path:** `{file_metrics.file_path}`",
                            f"**Average Complexity:** {file_metrics.avg_complexity:.2f}",
                            f"**Max Complexity:** {file_metrics.max_complexity}",
                            f"**Total Complexity:** {file_metrics.total_complexity}",
                            f"**Functions:** {file_metrics.function_count}",
                            f"**Health Score:** {file_metrics.health_score:.2f}\n",
                        ]
                    )

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Hotspot detection failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Hotspot detection failed: {str(e)}")
                ],
                isError=True,
            )

    async def _check_circular_dependencies(
        self, args: dict[str, Any]
    ) -> CallToolResult:
        """Handle check_circular_dependencies tool call."""
        try:
            # Find analyzable files to build import graph
            from ..cli.commands.analyze import _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            if not files_to_analyze:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No analyzable files found in project",
                        )
                    ],
                    isError=True,
                )

            # Import circular dependency detection
            from ..analysis.collectors.coupling import build_import_graph

            # Build import graph for the project (reverse dependency graph)
            import_graph = build_import_graph(
                self.project_root, files_to_analyze, language="python"
            )

            # Convert to forward dependency graph for cycle detection
            # import_graph maps: module -> set of files that import it (reverse)
            # We need: file -> list of files it imports (forward)
            forward_graph: dict[str, list[str]] = {}

            # Build forward graph by reading imports from files
            for file_path in files_to_analyze:
                file_str = str(file_path.relative_to(self.project_root))
                if file_str not in forward_graph:
                    forward_graph[file_str] = []

                # For each module in import_graph, if this file imports it, add edge
                for module, importers in import_graph.items():
                    for importer in importers:
                        importer_str = str(
                            Path(importer).relative_to(self.project_root)
                            if Path(importer).is_absolute()
                            else importer
                        )
                        if importer_str == file_str:
                            # This file imports the module, add forward edge
                            if module not in forward_graph[file_str]:
                                forward_graph[file_str].append(module)

            # Detect circular dependencies using DFS
            def find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
                """Find all cycles in the import graph using DFS."""
                cycles = []
                visited = set()
                rec_stack = set()

                def dfs(node: str, path: list[str]) -> None:
                    visited.add(node)
                    rec_stack.add(node)
                    path.append(node)

                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            dfs(neighbor, path.copy())
                        elif neighbor in rec_stack:
                            # Found a cycle
                            try:
                                cycle_start = path.index(neighbor)
                                cycle = path[cycle_start:] + [neighbor]
                                # Normalize cycle representation to avoid duplicates
                                cycle_tuple = tuple(sorted(cycle))
                                if not any(
                                    tuple(sorted(c)) == cycle_tuple for c in cycles
                                ):
                                    cycles.append(cycle)
                            except ValueError:
                                pass

                    rec_stack.remove(node)

                for node in graph:
                    if node not in visited:
                        dfs(node, [])

                return cycles

            cycles = find_cycles(forward_graph)

            # Format response
            if not cycles:
                response_text = "No circular dependencies detected"
            else:
                response_lines = [f"# Circular Dependencies Found: {len(cycles)}\n"]

                for i, cycle in enumerate(cycles, 1):
                    response_lines.append(f"## Cycle {i}")
                    response_lines.append("```")
                    for j, node in enumerate(cycle):
                        if j < len(cycle) - 1:
                            response_lines.append(f"{node}")
                            response_lines.append("  ↓")
                        else:
                            response_lines.append(f"{node} (back to {cycle[0]})")
                    response_lines.append("```\n")

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Circular dependency check failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Circular dependency check failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def _interpret_analysis(self, args: dict[str, Any]) -> CallToolResult:
        """Handle interpret_analysis tool call."""
        analysis_json_str = args.get("analysis_json", "")
        focus = args.get("focus", "summary")
        verbosity = args.get("verbosity", "normal")

        if not analysis_json_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="analysis_json parameter is required")
                ],
                isError=True,
            )

        try:
            import json

            from ..analysis.interpretation import AnalysisInterpreter, LLMContextExport

            # Parse JSON input
            analysis_data = json.loads(analysis_json_str)

            # Convert to LLMContextExport
            export = LLMContextExport(**analysis_data)

            # Create interpreter and generate interpretation
            interpreter = AnalysisInterpreter()
            interpretation = interpreter.interpret(
                export, focus=focus, verbosity=verbosity
            )

            return CallToolResult(
                content=[TextContent(type="text", text=interpretation)]
            )

        except json.JSONDecodeError as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid JSON input: {str(e)}",
                    )
                ],
                isError=True,
            )
        except Exception as e:
            logger.error(f"Analysis interpretation failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Interpretation failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    # =========================================================================
    # Multi-repo management tools
    # =========================================================================

    async def _register_repo(self, args: dict[str, Any]) -> CallToolResult:
        """Handle register_repo tool call."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="register_repo is only available in multi-repo mode. "
                        "Set MCP_MULTI_REPO_MODE=true to enable.",
                    )
                ],
                isError=True,
            )

        repo_path_str = args.get("repo_path", "")
        display_name = args.get("display_name")
        repo_id = args.get("repo_id")
        file_extensions = args.get("file_extensions")
        set_as_default = args.get("set_as_default", False)

        if not repo_path_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="repo_path parameter is required")
                ],
                isError=True,
            )

        try:
            repo_path = Path(repo_path_str).resolve()

            if not repo_path.exists():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Repository path does not exist: {repo_path}",
                        )
                    ],
                    isError=True,
                )

            # Ensure registry is loaded
            if not self.registry:
                self.registry = RepoRegistry.load(self.central_index_path)

            # Check if already registered
            existing = self.registry.get_repo_by_path(repo_path)
            if existing:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Repository already registered as '{existing.display_name}' "
                            f"(ID: {existing.repo_id})",
                        )
                    ]
                )

            # Register the repository
            repo_info = self.registry.register_repo(
                repo_path=repo_path,
                display_name=display_name,
                repo_id=repo_id,
                file_extensions=file_extensions,
                set_as_default=set_as_default,
            )

            response_lines = [
                "# Repository Registered Successfully\n",
                f"**ID:** {repo_info.repo_id}",
                f"**Name:** {repo_info.display_name}",
                f"**Path:** {repo_info.repo_path}",
                f"**Collection:** {repo_info.collection_name}",
                f"**Extensions:** {', '.join(repo_info.file_extensions)}",
                "",
                "To index this repository, use the `index_repo` tool.",
            ]

            if set_as_default or self.registry.default_repo_id == repo_info.repo_id:
                response_lines.insert(-1, "**Default:** Yes")

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(response_lines))]
            )

        except Exception as e:
            logger.error(f"Failed to register repository: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Failed to register repository: {str(e)}"
                    )
                ],
                isError=True,
            )

    async def _unregister_repo(self, args: dict[str, Any]) -> CallToolResult:
        """Handle unregister_repo tool call."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="unregister_repo is only available in multi-repo mode.",
                    )
                ],
                isError=True,
            )

        repo_id = args.get("repo_id", "")
        delete_index = args.get("delete_index", True)

        if not repo_id:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="repo_id parameter is required")
                ],
                isError=True,
            )

        try:
            if not self.registry:
                self.registry = RepoRegistry.load(self.central_index_path)

            repo_info = self.registry.get_repo(repo_id)
            if not repo_info:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"Repository not found: {repo_id}"
                        )
                    ],
                    isError=True,
                )

            # Close database if open
            if repo_id in self._repo_databases:
                db = self._repo_databases[repo_id]
                await db.__aexit__(None, None, None)
                del self._repo_databases[repo_id]

            if repo_id in self._repo_search_engines:
                del self._repo_search_engines[repo_id]

            # Delete index data if requested
            if delete_index:
                database = await self._get_repo_database(repo_id)
                if database:
                    try:
                        await database.reset()
                    except Exception as e:
                        logger.warning(f"Failed to reset database for {repo_id}: {e}")

            # Unregister
            display_name = repo_info.display_name
            self.registry.unregister_repo(repo_id)

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Repository '{display_name}' ({repo_id}) unregistered successfully.",
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Failed to unregister repository: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Failed to unregister repository: {str(e)}"
                    )
                ],
                isError=True,
            )

    async def _list_repos(self, args: dict[str, Any]) -> CallToolResult:
        """Handle list_repos tool call."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="list_repos is only available in multi-repo mode.",
                    )
                ],
                isError=True,
            )

        try:
            if not self.registry:
                self.registry = RepoRegistry.load(self.central_index_path)

            repos = self.registry.list_repos()

            if not repos:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No repositories registered. Use `register_repo` to add one.",
                        )
                    ]
                )

            response_lines = [f"# Registered Repositories ({len(repos)})\n"]

            for repo in repos:
                is_default = repo.repo_id == self.registry.default_repo_id
                default_marker = " ⭐ (default)" if is_default else ""

                response_lines.extend(
                    [
                        f"## {repo.display_name}{default_marker}",
                        f"**ID:** {repo.repo_id}",
                        f"**Path:** {repo.repo_path}",
                        f"**Indexed:** {'Yes' if repo.is_indexed else 'No'}",
                    ]
                )

                if repo.is_indexed:
                    response_lines.extend(
                        [
                            f"**Chunks:** {repo.chunk_count}",
                            f"**Files:** {repo.file_count}",
                            f"**Languages:** {', '.join(repo.languages) or 'Unknown'}",
                        ]
                    )

                response_lines.append("")

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(response_lines))]
            )

        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Failed to list repositories: {str(e)}"
                    )
                ],
                isError=True,
            )

    async def _set_default_repo(self, args: dict[str, Any]) -> CallToolResult:
        """Handle set_default_repo tool call."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="set_default_repo is only available in multi-repo mode.",
                    )
                ],
                isError=True,
            )

        repo_id = args.get("repo_id", "")

        if not repo_id:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="repo_id parameter is required")
                ],
                isError=True,
            )

        try:
            if not self.registry:
                self.registry = RepoRegistry.load(self.central_index_path)

            if not self.registry.set_default_repo(repo_id):
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"Repository not found: {repo_id}"
                        )
                    ],
                    isError=True,
                )

            repo_info = self.registry.get_repo(repo_id)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Default repository set to '{repo_info.display_name}' ({repo_id})",
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Failed to set default repository: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Failed to set default repository: {str(e)}"
                    )
                ],
                isError=True,
            )

    async def _get_repo_status(self, args: dict[str, Any]) -> CallToolResult:
        """Handle get_repo_status tool call."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="get_repo_status is only available in multi-repo mode.",
                    )
                ],
                isError=True,
            )

        repo_id = self._resolve_repo_id(args)

        if not repo_id:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="No repository specified and no default repository set.",
                    )
                ],
                isError=True,
            )

        try:
            repo_info = self.registry.get_repo(repo_id)
            if not repo_info:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"Repository not found: {repo_id}"
                        )
                    ],
                    isError=True,
                )

            is_default = repo_id == self.registry.default_repo_id

            response_lines = [
                f"# Repository Status: {repo_info.display_name}\n",
                f"**ID:** {repo_info.repo_id}",
                f"**Path:** {repo_info.repo_path}",
                f"**Default:** {'Yes' if is_default else 'No'}",
                f"**Collection:** {repo_info.collection_name}",
                f"**Embedding Model:** {repo_info.embedding_model}",
                f"**Extensions:** {', '.join(repo_info.file_extensions)}",
                "",
                "## Index Status",
                f"**Indexed:** {'Yes' if repo_info.is_indexed else 'No'}",
            ]

            if repo_info.is_indexed:
                import datetime

                last_indexed = (
                    datetime.datetime.fromtimestamp(repo_info.last_indexed).isoformat()
                    if repo_info.last_indexed
                    else "Unknown"
                )
                response_lines.extend(
                    [
                        f"**Last Indexed:** {last_indexed}",
                        f"**Total Chunks:** {repo_info.chunk_count}",
                        f"**Total Files:** {repo_info.file_count}",
                        f"**Languages:** {', '.join(repo_info.languages) or 'Unknown'}",
                    ]
                )

                # Try to get live stats from database
                try:
                    database = await self._get_repo_database(repo_id)
                    if database:
                        stats = await database.get_stats()
                        response_lines.extend(
                            [
                                "",
                                "## Live Statistics",
                                f"**Index Size:** {stats.index_size_mb:.2f} MB",
                            ]
                        )
                except Exception as e:
                    logger.debug(f"Could not get live stats: {e}")

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(response_lines))]
            )

        except Exception as e:
            logger.error(f"Failed to get repository status: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Failed to get repository status: {str(e)}"
                    )
                ],
                isError=True,
            )

    async def _index_repo(self, args: dict[str, Any]) -> CallToolResult:
        """Handle index_repo tool call."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="index_repo is only available in multi-repo mode. "
                        "Use index_project for single-repo mode.",
                    )
                ],
                isError=True,
            )

        repo_id = self._resolve_repo_id(args)
        force = args.get("force", False)

        if not repo_id:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="No repository specified and no default repository set.",
                    )
                ],
                isError=True,
            )

        try:
            repo_info = self.registry.get_repo(repo_id)
            if not repo_info:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"Repository not found: {repo_id}"
                        )
                    ],
                    isError=True,
                )

            # Get or create database for this repo
            database = await self._get_repo_database(repo_id)
            if not database:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Failed to initialize database for repository: {repo_id}",
                        )
                    ],
                    isError=True,
                )

            # Reset if forcing reindex
            if force and repo_info.is_indexed:
                await database.reset()
                # Reinitialize after reset
                database = await self._get_repo_database(repo_id)

            # Create indexer
            from ..config.settings import ProjectConfig

            config = ProjectConfig(
                project_root=repo_info.repo_path,
                index_path=self.registry.get_chroma_persist_path(),
                file_extensions=repo_info.file_extensions,
                embedding_model=repo_info.embedding_model,
            )

            indexer = SemanticIndexer(
                database=database,
                project_root=repo_info.repo_path,
                config=config,
            )

            # Run indexing
            stats = await indexer.index_project(force_reindex=force)

            # Update registry with stats
            self.registry.update_repo_stats(
                repo_id=repo_id,
                chunk_count=stats.total_chunks if hasattr(stats, "total_chunks") else 0,
                file_count=stats.files_indexed if hasattr(stats, "files_indexed") else 0,
                languages=list(stats.languages.keys()) if hasattr(stats, "languages") else [],
                is_indexed=True,
            )

            response_lines = [
                f"# Indexing Complete: {repo_info.display_name}\n",
                f"**Repository ID:** {repo_id}",
                f"**Files Indexed:** {stats.files_indexed if hasattr(stats, 'files_indexed') else 'N/A'}",
                f"**Total Chunks:** {stats.total_chunks if hasattr(stats, 'total_chunks') else 'N/A'}",
            ]

            if hasattr(stats, "languages") and stats.languages:
                response_lines.append(
                    f"**Languages:** {', '.join(stats.languages.keys())}"
                )

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(response_lines))]
            )

        except Exception as e:
            logger.error(f"Failed to index repository: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Failed to index repository: {str(e)}"
                    )
                ],
                isError=True,
            )

    async def _search_all_repos(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_all_repos tool call - search across all repositories."""
        if not self.multi_repo_mode:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="search_all_repos is only available in multi-repo mode.",
                    )
                ],
                isError=True,
            )

        query = args.get("query", "")
        limit = args.get("limit", 5)
        similarity_threshold = args.get("similarity_threshold", 0.3)

        if not query:
            return CallToolResult(
                content=[TextContent(type="text", text="Query parameter is required")],
                isError=True,
            )

        try:
            if not self.registry:
                self.registry = RepoRegistry.load(self.central_index_path)

            repos = self.registry.list_repos()
            indexed_repos = [r for r in repos if r.is_indexed]

            if not indexed_repos:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No indexed repositories found. Use `index_repo` to index a repository first.",
                        )
                    ]
                )

            all_results = []

            for repo in indexed_repos:
                try:
                    search_engine, _ = await self._get_repo_search_engine(repo.repo_id)
                    if not search_engine:
                        continue

                    results = await search_engine.search(
                        query=query,
                        limit=limit,
                        similarity_threshold=similarity_threshold,
                    )

                    for result in results:
                        all_results.append((repo, result))

                except Exception as e:
                    logger.warning(f"Search failed for repo {repo.repo_id}: {e}")
                    continue

            if not all_results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"No results found across {len(indexed_repos)} repositories for query: '{query}'",
                        )
                    ]
                )

            # Sort by similarity score
            all_results.sort(key=lambda x: x[1].similarity_score, reverse=True)

            # Format results
            response_lines = [
                f"# Cross-Repository Search Results\n",
                f"**Query:** {query}",
                f"**Repositories Searched:** {len(indexed_repos)}",
                f"**Total Results:** {len(all_results)}\n",
            ]

            for i, (repo, result) in enumerate(all_results[:20], 1):  # Limit to top 20
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**Repository:** {repo.display_name}")
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Truncate long content
                content = result.content[:500] if len(result.content) > 500 else result.content
                response_lines.append(content + ("..." if len(result.content) > 500 else ""))
                response_lines.append("```\n")

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(response_lines))]
            )

        except Exception as e:
            logger.error(f"Cross-repository search failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Cross-repository search failed: {str(e)}"
                    )
                ],
                isError=True,
            )

    def _get_threshold_config(self, preset: str) -> ThresholdConfig:
        """Get threshold configuration based on preset.

        Args:
            preset: Threshold preset ('strict', 'standard', or 'relaxed')

        Returns:
            ThresholdConfig instance
        """
        if preset == "strict":
            # Stricter thresholds
            config = ThresholdConfig()
            config.complexity.cognitive_a = 3
            config.complexity.cognitive_b = 7
            config.complexity.cognitive_c = 15
            config.complexity.cognitive_d = 20
            config.smells.long_method_lines = 30
            config.smells.high_complexity = 10
            config.smells.too_many_parameters = 3
            config.smells.deep_nesting_depth = 3
            return config
        elif preset == "relaxed":
            # More relaxed thresholds
            config = ThresholdConfig()
            config.complexity.cognitive_a = 7
            config.complexity.cognitive_b = 15
            config.complexity.cognitive_c = 25
            config.complexity.cognitive_d = 40
            config.smells.long_method_lines = 75
            config.smells.high_complexity = 20
            config.smells.too_many_parameters = 7
            config.smells.deep_nesting_depth = 5
            return config
        else:
            # Standard (default)
            return ThresholdConfig()


def create_mcp_server(
    project_root: Path | None = None,
    enable_file_watching: bool | None = None,
    multi_repo_mode: bool | None = None,
    central_index_path: Path | None = None,
) -> Server:
    """Create and configure the MCP server.

    Args:
        project_root: Project root directory. If None, will auto-detect.
        enable_file_watching: Enable file watching for automatic reindexing.
                              If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
        multi_repo_mode: Enable multi-repo mode. If None, checks MCP_MULTI_REPO_MODE env var.
        central_index_path: Override for central index path in multi-repo mode.
    """
    server = Server("mcp-vector-search")
    mcp_server = MCPVectorSearchServer(
        project_root=project_root,
        enable_file_watching=enable_file_watching,
        multi_repo_mode=multi_repo_mode,
        central_index_path=central_index_path,
    )

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        """List available tools."""
        return mcp_server.get_tools()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None):
        """Handle tool calls."""
        # Create a mock request object for compatibility
        from types import SimpleNamespace

        mock_request = SimpleNamespace()
        mock_request.params = SimpleNamespace()
        mock_request.params.name = name
        mock_request.params.arguments = arguments or {}

        result = await mcp_server.call_tool(mock_request)

        # Return the content from the result
        return result.content

    # Store reference for cleanup
    server._mcp_server = mcp_server

    return server


async def run_mcp_server(
    project_root: Path | None = None,
    enable_file_watching: bool | None = None,
    multi_repo_mode: bool | None = None,
    central_index_path: Path | None = None,
) -> None:
    """Run the MCP server using stdio transport.

    Args:
        project_root: Project root directory. If None, will auto-detect.
        enable_file_watching: Enable file watching for automatic reindexing.
                              If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
        multi_repo_mode: Enable multi-repo mode. If None, checks MCP_MULTI_REPO_MODE env var.
        central_index_path: Override for central index path in multi-repo mode.
    """
    server = create_mcp_server(
        project_root=project_root,
        enable_file_watching=enable_file_watching,
        multi_repo_mode=multi_repo_mode,
        central_index_path=central_index_path,
    )

    # Create initialization options with proper capabilities
    init_options = InitializationOptions(
        server_name="mcp-vector-search",
        server_version="0.5.0",  # Version bump for multi-repo support
        capabilities=ServerCapabilities(tools={"listChanged": True}, logging={}),
    )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(server, "_mcp_server"):
            logger.info("Performing server cleanup...")
            await server._mcp_server.cleanup()


if __name__ == "__main__":
    # Allow specifying project root as command line argument
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Check for file watching flag in command line args
    enable_file_watching = None
    multi_repo_mode = None

    if "--no-watch" in sys.argv:
        enable_file_watching = False
        sys.argv.remove("--no-watch")
    elif "--watch" in sys.argv:
        enable_file_watching = True
        sys.argv.remove("--watch")

    if "--multi-repo" in sys.argv:
        multi_repo_mode = True
        sys.argv.remove("--multi-repo")

    asyncio.run(
        run_mcp_server(
            project_root=project_root,
            enable_file_watching=enable_file_watching,
            multi_repo_mode=multi_repo_mode,
        )
    )
