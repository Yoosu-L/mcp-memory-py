#!/usr/bin/env python3

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json
import os
from typing import List, Dict, Any, TypedDict


# Define types
class Entity(TypedDict):
    name: str
    entityType: str
    observations: List[str]


class Relation(TypedDict):
    from_: str  # Using from_ to avoid Python keyword
    to: str
    relationType: str


class KnowledgeGraph(TypedDict):
    entities: List[Entity]
    relations: List[Relation]


class KnowledgeGraphManager:
    def __init__(self):
        # Get script directory for relative paths
        script_dir = Path(__file__).parent
        default_memory_path = script_dir / "memory.json"

        # Handle memory file path from environment
        memory_path = os.getenv("MEMORY_FILE_PATH")
        if memory_path:
            self.memory_path = Path(memory_path)
            if not self.memory_path.is_absolute():
                self.memory_path = script_dir / memory_path
        else:
            self.memory_path = default_memory_path

    async def load_graph(self) -> KnowledgeGraph:
        try:
            with open(self.memory_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                graph: KnowledgeGraph = {"entities": [], "relations": []}
                for line in lines:
                    item = json.loads(line)
                    if item["type"] == "entity":
                        del item["type"]
                        graph["entities"].append(item)
                    elif item["type"] == "relation":
                        del item["type"]
                        graph["relations"].append(item)
                return graph
        except FileNotFoundError:
            return {"entities": [], "relations": []}

    async def save_graph(self, graph: KnowledgeGraph):
        lines = []
        for entity in graph["entities"]:
            lines.append(json.dumps({"type": "entity", **entity}))
        for relation in graph["relations"]:
            lines.append(json.dumps({"type": "relation", **relation}))

        with open(self.memory_path, "w") as f:
            f.write("\n".join(lines))

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        graph = await self.load_graph()
        new_entities = [
            e
            for e in entities
            if not any(
                existing.get("name") == e["name"] for existing in graph["entities"]
            )
        ]
        graph["entities"].extend(new_entities)
        await self.save_graph(graph)
        return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        graph = await self.load_graph()
        new_relations = [
            r
            for r in relations
            if not any(
                existing.get("from_") == r["from_"]
                and existing.get("to") == r["to"]
                and existing.get("relationType") == r["relationType"]
                for existing in graph["relations"]
            )
        ]
        graph["relations"].extend(new_relations)
        await self.save_graph(graph)
        return new_relations

    async def add_observations(
        self, observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        graph = await self.load_graph()
        results = []

        for obs in observations:
            entity = next(
                (e for e in graph["entities"] if e["name"] == obs["entityName"]), None
            )
            if not entity:
                raise ValueError(f"Entity with name {obs['entityName']} not found")

            new_observations = [
                content
                for content in obs["contents"]
                if content not in entity["observations"]
            ]
            entity["observations"].extend(new_observations)
            results.append(
                {"entityName": obs["entityName"], "addedObservations": new_observations}
            )

        await self.save_graph(graph)
        return results

    async def delete_entities(self, entity_names: List[str]) -> None:
        graph = await self.load_graph()
        graph["entities"] = [
            e for e in graph["entities"] if e["name"] not in entity_names
        ]
        graph["relations"] = [
            r
            for r in graph["relations"]
            if r["from_"] not in entity_names and r["to"] not in entity_names
        ]
        await self.save_graph(graph)

    async def delete_observations(self, deletions: List[Dict[str, Any]]) -> None:
        graph = await self.load_graph()
        for deletion in deletions:
            entity = next(
                (e for e in graph["entities"] if e["name"] == deletion["entityName"]),
                None,
            )
            if entity:
                entity["observations"] = [
                    obs
                    for obs in entity["observations"]
                    if obs not in deletion["observations"]
                ]
        await self.save_graph(graph)

    async def delete_relations(self, relations: List[Relation]) -> None:
        graph = await self.load_graph()
        graph["relations"] = [
            r
            for r in graph["relations"]
            if not any(
                r["from_"] == del_rel["from_"]
                and r["to"] == del_rel["to"]
                and r["relationType"] == del_rel["relationType"]
                for del_rel in relations
            )
        ]
        await self.save_graph(graph)

    async def read_graph(self) -> KnowledgeGraph:
        return await self.load_graph()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        graph = await self.load_graph()
        query = query.lower()

        filtered_entities = [
            e
            for e in graph["entities"]
            if query in e["name"].lower()
            or query in e["entityType"].lower()
            or any(query in obs.lower() for obs in e["observations"])
        ]

        filtered_entity_names = {e["name"] for e in filtered_entities}
        filtered_relations = [
            r
            for r in graph["relations"]
            if r["from_"] in filtered_entity_names and r["to"] in filtered_entity_names
        ]

        return {"entities": filtered_entities, "relations": filtered_relations}

    async def open_nodes(self, names: List[str]) -> KnowledgeGraph:
        graph = await self.load_graph()
        filtered_entities = [e for e in graph["entities"] if e["name"] in names]
        filtered_entity_names = {e["name"] for e in filtered_entities}

        filtered_relations = [
            r
            for r in graph["relations"]
            if r["from_"] in filtered_entity_names and r["to"] in filtered_entity_names
        ]

        return {"entities": filtered_entities, "relations": filtered_relations}


# Create FastMCP server
mcp = FastMCP("Memory Graph", dependencies=["mcp"])

# Initialize knowledge graph manager
knowledge_graph = KnowledgeGraphManager()


@mcp.tool()
async def create_entities(entities: List[Entity]) -> List[Entity]:
    """Create multiple new entities in the knowledge graph.

    Args:
        entities: List of entities to create, each with name, type and observations

    Returns:
        List of newly created entities (excluding any that already existed)
    """
    return await knowledge_graph.create_entities(entities)


@mcp.tool()
async def create_relations(relations: List[Relation]) -> List[Relation]:
    """Create new relations between entities in the knowledge graph.

    Args:
        relations: List of relations to create, each with from_, to and relationType

    Returns:
        List of newly created relations (excluding any duplicates)
    """
    return await knowledge_graph.create_relations(relations)


@mcp.tool()
async def add_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add new observations to existing entities.

    Args:
        observations: List of observations to add, each with entityName and contents

    Returns:
        List of results showing which observations were added to which entities
    """
    return await knowledge_graph.add_observations(observations)


@mcp.tool()
async def delete_entities(entity_names: List[str]) -> None:
    """Delete entities and their associated relations from the graph.

    Args:
        entity_names: List of entity names to delete
    """
    await knowledge_graph.delete_entities(entity_names)


@mcp.tool()
async def delete_observations(deletions: List[Dict[str, Any]]) -> None:
    """Delete specific observations from entities.

    Args:
        deletions: List of deletions, each with entityName and observations to remove
    """
    await knowledge_graph.delete_observations(deletions)


@mcp.tool()
async def delete_relations(relations: List[Relation]) -> None:
    """Delete specific relations from the graph.

    Args:
        relations: List of relations to delete
    """
    await knowledge_graph.delete_relations(relations)


@mcp.tool()
async def read_graph() -> KnowledgeGraph:
    """Read the entire knowledge graph.

    Returns:
        The complete knowledge graph with all entities and relations
    """
    return await knowledge_graph.read_graph()


@mcp.tool()
async def search_nodes(query: str) -> KnowledgeGraph:
    """Search for nodes and their relations matching a query string.

    Args:
        query: Search string to match against node names, types and observations

    Returns:
        Subgraph containing matching entities and their interconnecting relations
    """
    return await knowledge_graph.search_nodes(query)


@mcp.tool()
async def open_nodes(names: List[str]) -> KnowledgeGraph:
    """Get a subgraph containing specific nodes and their interconnecting relations.

    Args:
        names: List of entity names to include

    Returns:
        Subgraph containing the specified entities and relations between them
    """
    return await knowledge_graph.open_nodes(names)


async def initialize_test_data():
    # Create test person entities
    test_entities = [
        {
            "name": "Alice",
            "entityType": "person",
            "observations": ["Software engineer", "Likes hiking"],
        },
        {
            "name": "Bob",
            "entityType": "person",
            "observations": ["Data scientist", "Coffee enthusiast"],
        },
        {
            "name": "Carol",
            "entityType": "person",
            "observations": ["Product manager", "Avid reader"],
        },
    ]

    # Create test location entities
    location_entities = [
        {
            "name": "San Francisco",
            "entityType": "location",
            "observations": ["Tech hub", "Foggy weather"],
        },
        {
            "name": "New York",
            "entityType": "location",
            "observations": ["Financial center", "Busy streets"],
        },
    ]

    # Create all entities
    await knowledge_graph.create_entities(test_entities)
    await knowledge_graph.create_entities(location_entities)

    # Create test relations
    test_relations = [
        {"from_": "Alice", "to": "Bob", "relationType": "colleague"},
        {"from_": "Bob", "to": "Carol", "relationType": "friend"},
        {"from_": "Alice", "to": "San Francisco", "relationType": "lives_in"},
        {"from_": "Bob", "to": "New York", "relationType": "lives_in"},
    ]

    await knowledge_graph.create_relations(test_relations)


if __name__ == "__main__":
    import asyncio

    # Initialize test data before running the server
    asyncio.run(initialize_test_data())
    mcp.run()
