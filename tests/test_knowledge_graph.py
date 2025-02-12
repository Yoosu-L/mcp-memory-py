import pytest
import pytest_asyncio
import os
import tempfile

from mcp_memory_py.knowledge_graph import (
    KnowledgeGraphManager,
)


@pytest_asyncio.fixture
async def graph_manager():
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    # Set environment variable to use temp file
    os.environ["MEMORY_FILE_PATH"] = tmp_path
    manager = KnowledgeGraphManager()

    try:
        yield manager
    finally:
        # Cleanup
        os.unlink(tmp_path)
        del os.environ["MEMORY_FILE_PATH"]


@pytest.mark.asyncio
async def test_create_entities(graph_manager):
    # Test creating new entities
    entities = [
        {
            "name": "Alice",
            "entityType": "person",
            "observations": ["Software engineer"],
        },
        {"name": "Bob", "entityType": "person", "observations": ["Data scientist"]},
    ]

    new_entities = await graph_manager.create_entities(entities)
    assert len(new_entities) == 2
    assert new_entities[0]["name"] == "Alice"
    assert new_entities[1]["name"] == "Bob"

    # Test duplicate entity is not added
    duplicate = await graph_manager.create_entities([entities[0]])
    assert len(duplicate) == 0


@pytest.mark.asyncio
async def test_create_relations(graph_manager):
    # Create test entities first
    entities = [
        {"name": "Alice", "entityType": "person", "observations": []},
        {"name": "Bob", "entityType": "person", "observations": []},
    ]
    await graph_manager.create_entities(entities)

    # Test creating new relations
    relations = [{"from_": "Alice", "to": "Bob", "relationType": "friend"}]

    new_relations = await graph_manager.create_relations(relations)
    assert len(new_relations) == 1
    assert new_relations[0]["from_"] == "Alice"
    assert new_relations[0]["to"] == "Bob"

    # Test duplicate relation is not added
    duplicate = await graph_manager.create_relations(relations)
    assert len(duplicate) == 0


@pytest.mark.asyncio
async def test_add_observations(graph_manager):
    # Create test entity first
    await graph_manager.create_entities(
        [
            {
                "name": "Alice",
                "entityType": "person",
                "observations": ["Initial observation"],
            }
        ]
    )

    # Test adding new observations
    observations = [{"entityName": "Alice", "contents": ["New observation"]}]

    results = await graph_manager.add_observations(observations)
    assert len(results) == 1
    assert results[0]["entityName"] == "Alice"
    assert len(results[0]["addedObservations"]) == 1

    # Test adding duplicate observation
    duplicate = await graph_manager.add_observations(observations)
    assert len(duplicate[0]["addedObservations"]) == 0

    # Test adding to non-existent entity
    with pytest.raises(ValueError):
        await graph_manager.add_observations(
            [{"entityName": "NonExistent", "contents": ["Test"]}]
        )


@pytest.mark.asyncio
async def test_delete_entities(graph_manager):
    # Create test entities and relations
    entities = [
        {"name": "Alice", "entityType": "person", "observations": []},
        {"name": "Bob", "entityType": "person", "observations": []},
    ]
    await graph_manager.create_entities(entities)
    await graph_manager.create_relations(
        [{"from_": "Alice", "to": "Bob", "relationType": "friend"}]
    )

    # Test deleting entity
    await graph_manager.delete_entities(["Alice"])
    graph = await graph_manager.read_graph()

    assert len(graph["entities"]) == 1
    assert graph["entities"][0]["name"] == "Bob"
    assert len(graph["relations"]) == 0  # Relation should be deleted


@pytest.mark.asyncio
async def test_delete_observations(graph_manager):
    # Create test entity with observations
    await graph_manager.create_entities(
        [{"name": "Alice", "entityType": "person", "observations": ["Obs1", "Obs2"]}]
    )

    # Test deleting observation
    deletions = [{"entityName": "Alice", "observations": ["Obs1"]}]
    await graph_manager.delete_observations(deletions)

    graph = await graph_manager.read_graph()
    assert len(graph["entities"][0]["observations"]) == 1
    assert graph["entities"][0]["observations"][0] == "Obs2"


@pytest.mark.asyncio
async def test_delete_relations(graph_manager):
    # Create test entities and relations
    entities = [
        {"name": "Alice", "entityType": "person", "observations": []},
        {"name": "Bob", "entityType": "person", "observations": []},
    ]
    relations = [{"from_": "Alice", "to": "Bob", "relationType": "friend"}]

    await graph_manager.create_entities(entities)
    await graph_manager.create_relations(relations)

    # Test deleting relation
    await graph_manager.delete_relations(relations)
    graph = await graph_manager.read_graph()
    assert len(graph["relations"]) == 0


@pytest.mark.asyncio
async def test_search_nodes(graph_manager):
    # Create test data
    entities = [
        {
            "name": "Alice",
            "entityType": "person",
            "observations": ["Software engineer"],
        },
        {"name": "Bob", "entityType": "person", "observations": ["Data scientist"]},
        {"name": "Carol", "entityType": "person", "observations": ["Product manager"]},
    ]
    relations = [{"from_": "Alice", "to": "Bob", "relationType": "friend"}]

    await graph_manager.create_entities(entities)
    await graph_manager.create_relations(relations)

    # Test search by name
    result = await graph_manager.search_nodes("alice")
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "Alice"

    # Test search by observation
    result = await graph_manager.search_nodes("engineer")
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "Alice"

    # Test search by type
    result = await graph_manager.search_nodes("person")
    assert len(result["entities"]) == 3


@pytest.mark.asyncio
async def test_open_nodes(graph_manager):
    # Create test data
    entities = [
        {"name": "Alice", "entityType": "person", "observations": []},
        {"name": "Bob", "entityType": "person", "observations": []},
        {"name": "Carol", "entityType": "person", "observations": []},
    ]
    relations = [
        {"from_": "Alice", "to": "Bob", "relationType": "friend"},
        {"from_": "Bob", "to": "Carol", "relationType": "friend"},
    ]

    await graph_manager.create_entities(entities)
    await graph_manager.create_relations(relations)

    # Test opening specific nodes
    result = await graph_manager.open_nodes(["Alice", "Bob"])
    assert len(result["entities"]) == 2
    assert len(result["relations"]) == 1
    assert result["relations"][0]["from_"] == "Alice"
    assert result["relations"][0]["to"] == "Bob"


@pytest.mark.asyncio
async def test_persistence(graph_manager):
    # Create test data
    entities = [
        {"name": "Alice", "entityType": "person", "observations": ["Software engineer"]}
    ]
    await graph_manager.create_entities(entities)

    # Create new manager instance with same file
    new_manager = KnowledgeGraphManager()

    # Verify data persisted
    graph = await new_manager.read_graph()
    assert len(graph["entities"]) == 1
    assert graph["entities"][0]["name"] == "Alice"
    assert graph["entities"][0]["observations"] == ["Software engineer"]
