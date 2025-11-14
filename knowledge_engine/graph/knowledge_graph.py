"""
Knowledge graph implementation using Neo4j
"""
from typing import Any, Optional
from loguru import logger

from shared.models.base import IKnowledgeGraph
from shared.config.settings import settings


class Neo4jKnowledgeGraph(IKnowledgeGraph):
    """Neo4j-based knowledge graph implementation"""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j knowledge graph

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        from neo4j import AsyncGraphDatabase

        self.uri = uri or settings.neo4j.neo4j_uri
        self.user = user or settings.neo4j.neo4j_user
        self.password = password or settings.neo4j.neo4j_password
        self.database = database or settings.neo4j.neo4j_database

        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

        logger.info(f"Initialized Neo4j knowledge graph: {self.uri}")

    async def close(self) -> None:
        """Close database connection"""
        await self.driver.close()
        logger.info("Closed Neo4j connection")

    async def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any]
    ) -> None:
        """
        Add a node to the graph

        Args:
            node_id: Unique node identifier
            node_type: Node type/label
            properties: Node properties
        """
        async with self.driver.session(database=self.database) as session:
            query = f"""
            MERGE (n:{node_type} {{id: $node_id}})
            SET n += $properties
            RETURN n
            """

            await session.run(
                query,
                node_id=node_id,
                properties=properties
            )

            logger.debug(f"Added node: {node_type}:{node_id}")

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Add an edge between nodes

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type/relationship
            properties: Edge properties
        """
        async with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{edge_type}]->(target)
            SET r += $properties
            RETURN r
            """

            await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=properties or {}
            )

            logger.debug(f"Added edge: {source_id} -{edge_type}-> {target_id}")

    async def find_related(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """
        Find related nodes

        Args:
            node_id: Starting node ID
            edge_types: Filter by edge types (None = all types)
            max_depth: Maximum traversal depth

        Returns:
            List of related node dictionaries
        """
        async with self.driver.session(database=self.database) as session:
            # Build relationship pattern
            if edge_types:
                rel_pattern = "|".join(edge_types)
                rel_clause = f"[:{rel_pattern}*1..{max_depth}]"
            else:
                rel_clause = f"[*1..{max_depth}]"

            query = f"""
            MATCH (start {{id: $node_id}})-{rel_clause}-(related)
            RETURN DISTINCT related, labels(related) as labels
            """

            result = await session.run(query, node_id=node_id)

            related_nodes = []
            async for record in result:
                node = record["related"]
                labels = record["labels"]

                related_nodes.append({
                    "id": node.get("id"),
                    "type": labels[0] if labels else "Unknown",
                    "properties": dict(node)
                })

            logger.info(f"Found {len(related_nodes)} related nodes for {node_id}")
            return related_nodes

    async def traverse(
        self,
        start_id: str,
        query: str
    ) -> list[dict[str, Any]]:
        """
        Execute a graph traversal query

        Args:
            start_id: Starting node ID
            query: Cypher query pattern

        Returns:
            Query results
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, start_id=start_id)

            records = []
            async for record in result:
                records.append(dict(record))

            logger.info(f"Traversal query returned {len(records)} results")
            return records

    async def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """
        Get node by ID

        Args:
            node_id: Node ID

        Returns:
            Node data or None
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (n {id: $node_id})
            RETURN n, labels(n) as labels
            """

            result = await session.run(query, node_id=node_id)
            record = await result.single()

            if record:
                node = record["n"]
                labels = record["labels"]

                return {
                    "id": node.get("id"),
                    "type": labels[0] if labels else "Unknown",
                    "properties": dict(node)
                }

            return None

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> list[dict[str, Any]]:
        """
        Find shortest path between two nodes

        Args:
            start_id: Start node ID
            end_id: End node ID
            max_depth: Maximum path length

        Returns:
            List of nodes in path
        """
        async with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH path = shortestPath(
                (start {{id: $start_id}})-[*1..{max_depth}]-(end {{id: $end_id}})
            )
            RETURN nodes(path) as path_nodes
            """

            result = await session.run(
                query,
                start_id=start_id,
                end_id=end_id
            )
            record = await result.single()

            if record:
                nodes = record["path_nodes"]
                return [dict(node) for node in nodes]

            return []

    async def create_hierarchy(
        self,
        parent_id: str,
        child_id: str,
        level: int
    ) -> None:
        """
        Create hierarchical relationship

        Args:
            parent_id: Parent node ID
            child_id: Child node ID
            level: Hierarchy level
        """
        await self.add_edge(
            parent_id,
            child_id,
            "PARENT_OF",
            {"hierarchy_level": level}
        )

    async def find_prerequisites(self, concept_id: str) -> list[dict[str, Any]]:
        """
        Find prerequisites for a concept

        Args:
            concept_id: Concept node ID

        Returns:
            List of prerequisite concepts
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (concept {id: $concept_id})-[:REQUIRES]->(prereq)
            RETURN prereq, labels(prereq) as labels
            ORDER BY prereq.hierarchy_level
            """

            result = await session.run(query, concept_id=concept_id)

            prerequisites = []
            async for record in result:
                node = record["prereq"]
                labels = record["labels"]

                prerequisites.append({
                    "id": node.get("id"),
                    "type": labels[0] if labels else "Unknown",
                    "properties": dict(node)
                })

            return prerequisites

    async def find_examples(
        self,
        concept_id: str,
        asset_class: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Find examples for a concept

        Args:
            concept_id: Concept node ID
            asset_class: Filter by asset class

        Returns:
            List of example nodes
        """
        async with self.driver.session(database=self.database) as session:
            if asset_class:
                query = """
                MATCH (concept {id: $concept_id})-[:HAS_EXAMPLE]->(example)
                WHERE $asset_class IN example.asset_classes
                RETURN example
                """
                params = {"concept_id": concept_id, "asset_class": asset_class}
            else:
                query = """
                MATCH (concept {id: $concept_id})-[:HAS_EXAMPLE]->(example)
                RETURN example
                """
                params = {"concept_id": concept_id}

            result = await session.run(query, **params)

            examples = []
            async for record in result:
                examples.append(dict(record["example"]))

            return examples

    async def create_indexes(self) -> None:
        """Create database indexes for performance"""
        async with self.driver.session(database=self.database) as session:
            # Create indexes on common node types
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Strategy) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Example) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Topic) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.hierarchy_level)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Strategy) ON (n.asset_classes)",
            ]

            for index_query in indexes:
                await session.run(index_query)

            logger.info("Created Neo4j indexes")

    async def clear_graph(self) -> None:
        """Clear all nodes and relationships (use with caution!)"""
        async with self.driver.session(database=self.database) as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared entire knowledge graph")

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get graph statistics

        Returns:
            Statistics dictionary
        """
        async with self.driver.session(database=self.database) as session:
            # Count nodes
            node_result = await session.run("MATCH (n) RETURN count(n) as count")
            node_record = await node_result.single()
            num_nodes = node_record["count"] if node_record else 0

            # Count relationships
            rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_record = await rel_result.single()
            num_relationships = rel_record["count"] if rel_record else 0

            # Count by type
            type_result = await session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(*) as count
                ORDER BY count DESC
            """)

            node_types = {}
            async for record in type_result:
                node_types[record["type"]] = record["count"]

            return {
                "num_nodes": num_nodes,
                "num_relationships": num_relationships,
                "node_types": node_types
            }


class InMemoryKnowledgeGraph(IKnowledgeGraph):
    """In-memory knowledge graph for testing/development"""

    def __init__(self):
        """Initialize in-memory graph"""
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

        logger.info("Initialized in-memory knowledge graph")

    async def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any]
    ) -> None:
        """Add a node"""
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties
        }

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict[str, Any]] = None
    ) -> None:
        """Add an edge"""
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "properties": properties or {}
        })

    async def find_related(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """Find related nodes"""
        related = set()

        def traverse(current_id: str, depth: int) -> None:
            if depth > max_depth:
                return

            for edge in self.edges:
                if edge["source"] == current_id:
                    if not edge_types or edge["type"] in edge_types:
                        target_id = edge["target"]
                        if target_id in self.nodes:
                            related.add(target_id)
                            traverse(target_id, depth + 1)

        traverse(node_id, 1)
        return [self.nodes[nid] for nid in related if nid in self.nodes]

    async def traverse(
        self,
        start_id: str,
        query: str
    ) -> list[dict[str, Any]]:
        """Execute traversal (simplified)"""
        return await self.find_related(start_id)
