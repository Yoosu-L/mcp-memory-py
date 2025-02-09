#!/usr/bin/env python3

import logging
import json
import time
import functools
from pathlib import Path
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from .knowledge_graph import KnowledgeGraphManager, Entity, Relation, KnowledgeGraph
from typing import List, Dict, Any, Callable, TypeVar, ParamSpec

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"mcp_server_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger("mcp_memory")

# Type variables for decorator
P = ParamSpec("P")
R = TypeVar("R")


def log_tool_call(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log tool calls with timing and results."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()
        tool_name = func.__name__

        # Log call
        try:
            call_args = {
                "args": [str(arg) for arg in args[1:]],  # Skip self
                "kwargs": {k: str(v) for k, v in kwargs.items()},
            }
            logger.info(f"Tool call: {tool_name} - Args: {json.dumps(call_args)}")

            # Execute tool
            result = await func(*args, **kwargs)

            # Log success
            execution_time = time.time() - start_time
            logger.info(
                f"Tool success: {tool_name} - "
                f"Time: {execution_time:.2f}s - "
                f"Result: {str(result)[:200]}..."  # Truncate long results
            )
            return result

        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.error(
                f"Tool error: {tool_name} - "
                f"Time: {execution_time:.2f}s - "
                f"Error: {str(e)}"
            )
            raise

    return wrapper


# Create FastMCP server
mcp = FastMCP("Memory Graph", dependencies=["mcp"])

# Initialize knowledge graph manager
knowledge_graph = KnowledgeGraphManager()


@mcp.tool()
@log_tool_call
async def create_entities(entities: List[Entity]) -> List[Entity]:
    """Create multiple new entities in the knowledge graph.

    Args:
        entities: List of entities to create, each with name, type and observations

    Returns:
        List of newly created entities (excluding any that already existed)
    """
    return await knowledge_graph.create_entities(entities)


@mcp.tool()
@log_tool_call
async def create_relations(relations: List[Relation]) -> List[Relation]:
    """Create new relations between entities in the knowledge graph.

    Args:
        relations: List of relations to create, each with from_, to and relationType

    Returns:
        List of newly created relations (excluding any duplicates)
    """
    return await knowledge_graph.create_relations(relations)


@mcp.tool()
@log_tool_call
async def add_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add new observations to existing entities.

    Args:
        observations: List of observations to add, each with entityName and contents

    Returns:
        List of results showing which observations were added to which entities
    """
    return await knowledge_graph.add_observations(observations)


@mcp.tool()
@log_tool_call
async def delete_entities(entity_names: List[str]) -> None:
    """Delete entities and their associated relations from the graph.

    Args:
        entity_names: List of entity names to delete
    """
    await knowledge_graph.delete_entities(entity_names)


@mcp.tool()
@log_tool_call
async def delete_observations(deletions: List[Dict[str, Any]]) -> None:
    """Delete specific observations from entities.

    Args:
        deletions: List of deletions, each with entityName and observations to remove
    """
    await knowledge_graph.delete_observations(deletions)


@mcp.tool()
@log_tool_call
async def delete_relations(relations: List[Relation]) -> None:
    """Delete specific relations from the graph.

    Args:
        relations: List of relations to delete
    """
    await knowledge_graph.delete_relations(relations)


@mcp.tool()
@log_tool_call
async def read_graph() -> KnowledgeGraph:
    """Read the entire knowledge graph.

    Returns:
        The complete knowledge graph with all entities and relations
    """
    return await knowledge_graph.read_graph()


@mcp.tool()
@log_tool_call
async def search_nodes(query: str) -> KnowledgeGraph:
    """Search for nodes and their relations matching a query string.

    Args:
        query: Search string to match against node names, types and observations

    Returns:
        Subgraph containing matching entities and their interconnecting relations
    """
    return await knowledge_graph.search_nodes(query)


@mcp.tool()
@log_tool_call
async def open_nodes(names: List[str]) -> KnowledgeGraph:
    """Get a subgraph containing specific nodes and their interconnecting relations.

    Args:
        names: List of entity names to include

    Returns:
        Subgraph containing the specified entities and relations between them
    """
    return await knowledge_graph.open_nodes(names)
