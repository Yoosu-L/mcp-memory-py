#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import List, Dict, Any, TypedDict
from modelcontextprotocol.server import Server, StdioServerTransport

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
        default_memory_path = script_dir / 'memory.json'
        
        # Handle memory file path from environment
        memory_path = os.getenv('MEMORY_FILE_PATH')
        if memory_path:
            self.memory_path = Path(memory_path)
            if not self.memory_path.is_absolute():
                self.memory_path = script_dir / memory_path
        else:
            self.memory_path = default_memory_path

    async def load_graph(self) -> KnowledgeGraph:
        try:
            with open(self.memory_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                graph: KnowledgeGraph = {'entities': [], 'relations': []}
                for line in lines:
                    item = json.loads(line)
                    if item['type'] == 'entity':
                        del item['type']
                        graph['entities'].append(item)
                    elif item['type'] == 'relation':
                        del item['type']
                        graph['relations'].append(item)
                return graph
        except FileNotFoundError:
            return {'entities': [], 'relations': []}

    async def save_graph(self, graph: KnowledgeGraph):
        lines = []
        for entity in graph['entities']:
            lines.append(json.dumps({'type': 'entity', **entity}))
        for relation in graph['relations']:
            lines.append(json.dumps({'type': 'relation', **relation}))
        
        with open(self.memory_path, 'w') as f:
            f.write('\n'.join(lines))

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        graph = await self.load_graph()
        new_entities = [
            e for e in entities 
            if not any(existing.get('name') == e['name'] for existing in graph['entities'])
        ]
        graph['entities'].extend(new_entities)
        await self.save_graph(graph)
        return new_entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        graph = await self.load_graph()
        new_relations = [
            r for r in relations
            if not any(
                existing.get('from_') == r['from_'] and
                existing.get('to') == r['to'] and
                existing.get('relationType') == r['relationType']
                for existing in graph['relations']
            )
        ]
        graph['relations'].extend(new_relations)
        await self.save_graph(graph)
        return new_relations

    async def add_observations(self, observations: List[Dict[str, Any]]):
        graph = await self.load_graph()
        results = []
        
        for obs in observations:
            entity = next((e for e in graph['entities'] if e['name'] == obs['entityName']), None)
            if not entity:
                raise ValueError(f"Entity with name {obs['entityName']} not found")
            
            new_observations = [
                content for content in obs['contents']
                if content not in entity['observations']
            ]
            entity['observations'].extend(new_observations)
            results.append({
                'entityName': obs['entityName'],
                'addedObservations': new_observations
            })
        
        await self.save_graph(graph)
        return results

    async def delete_entities(self, entity_names: List[str]):
        graph = await self.load_graph()
        graph['entities'] = [e for e in graph['entities'] if e['name'] not in entity_names]
        graph['relations'] = [
            r for r in graph['relations']
            if r['from_'] not in entity_names and r['to'] not in entity_names
        ]
        await self.save_graph(graph)

    async def delete_observations(self, deletions: List[Dict[str, Any]]):
        graph = await self.load_graph()
        for deletion in deletions:
            entity = next((e for e in graph['entities'] if e['name'] == deletion['entityName']), None)
            if entity:
                entity['observations'] = [
                    obs for obs in entity['observations']
                    if obs not in deletion['observations']
                ]
        await self.save_graph(graph)

    async def delete_relations(self, relations: List[Relation]):
        graph = await self.load_graph()
        graph['relations'] = [
            r for r in graph['relations']
            if not any(
                r['from_'] == del_rel['from_'] and
                r['to'] == del_rel['to'] and
                r['relationType'] == del_rel['relationType']
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
            e for e in graph['entities']
            if query in e['name'].lower() or
               query in e['entityType'].lower() or
               any(query in obs.lower() for obs in e['observations'])
        ]
        
        filtered_entity_names = {e['name'] for e in filtered_entities}
        filtered_relations = [
            r for r in graph['relations']
            if r['from_'] in filtered_entity_names and r['to'] in filtered_entity_names
        ]
        
        return {
            'entities': filtered_entities,
            'relations': filtered_relations
        }

    async def open_nodes(self, names: List[str]) -> KnowledgeGraph:
        graph = await self.load_graph()
        filtered_entities = [e for e in graph['entities'] if e['name'] in names]
        filtered_entity_names = {e['name'] for e in filtered_entities}
        
        filtered_relations = [
            r for r in graph['relations']
            if r['from_'] in filtered_entity_names and r['to'] in filtered_entity_names
        ]
        
        return {
            'entities': filtered_entities,
            'relations': filtered_relations
        }

def create_server():
    knowledge_graph_manager = KnowledgeGraphManager()
    
    server = Server(
        name="memory-server",
        version="1.0.0",
        capabilities={'tools': {}}
    )

    tools_config = [
        {
            "name": "create_entities",
            "description": "Create multiple new entities in the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "entityType": {"type": "string"},
                                "observations": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["name", "entityType", "observations"]
                        }
                    }
                },
                "required": ["entities"]
            }
        },
        # Add other tool configurations here...
    ]

    @server.list_tools
    async def list_tools():
        return {"tools": tools_config}

    @server.call_tool
    async def call_tool(name: str, arguments: Dict[str, Any]):
        if not arguments:
            raise ValueError(f"No arguments provided for tool: {name}")

        async def format_response(data):
            return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]}

        handlers = {
            "create_entities": lambda: knowledge_graph_manager.create_entities(arguments["entities"]),
            "create_relations": lambda: knowledge_graph_manager.create_relations(arguments["relations"]),
            "add_observations": lambda: knowledge_graph_manager.add_observations(arguments["observations"]),
            "delete_entities": lambda: knowledge_graph_manager.delete_entities(arguments["entityNames"]),
            "delete_observations": lambda: knowledge_graph_manager.delete_observations(arguments["deletions"]),
            "delete_relations": lambda: knowledge_graph_manager.delete_relations(arguments["relations"]),
            "read_graph": lambda: knowledge_graph_manager.read_graph(),
            "search_nodes": lambda: knowledge_graph_manager.search_nodes(arguments["query"]),
            "open_nodes": lambda: knowledge_graph_manager.open_nodes(arguments["names"])
        }

        if name not in handlers:
            raise ValueError(f"Unknown tool: {name}")

        result = await handlers[name]()
        return await format_response(result)

    return server

async def main():
    server = create_server()
    transport = StdioServerTransport()
    await server.connect(transport)
    print("Knowledge Graph MCP Server running on stdio", file=sys.stderr)

if __name__ == "__main__":
    import asyncio
    import sys
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error in main(): {e}", file=sys.stderr)
        sys.exit(1)