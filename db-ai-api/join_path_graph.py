"""Dynamic join path computation using graph algorithms.

This module provides automatic join path discovery between any tables
using the FK relationships from the database schema. Instead of hardcoding
join patterns for each table pair, we compute them dynamically.

Usage:
    graph = JoinPathGraph(schema)
    path = graph.find_join_path("Client", "Order")
    template = graph.get_join_template(["Client", "Order", "Product"])
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, Any
from functools import lru_cache

from loguru import logger


def normalize_table_name(name: str) -> str:
    """Unified table name normalization - use everywhere.

    Removes dbo. prefix and square brackets from table names.
    This function should be used across all components for consistency.

    Args:
        name: Table name (may include dbo., brackets, etc.)

    Returns:
        Normalized table name without prefix or brackets
    """
    return name.replace("dbo.", "").replace("[", "").replace("]", "").strip()


class JoinPathGraph:
    """Graph-based join path computation for any table combination.

    Builds a bidirectional graph from FK relationships and uses BFS
    to find optimal join paths between tables. Handles 312+ tables
    with 782+ relationships efficiently.

    Key features:
    - Automatic graph construction from schema FKs
    - Critical business paths override BFS results
    - BFS for shortest path (fallback for non-critical paths)
    - Path caching for repeated queries
    - Template generation with proper aliases
    - Handles tables with multiple FK paths
    """

    # CRITICAL BUSINESS PATHS - these override BFS results
    # Format: (source, target) -> [(intermediate_tables...), join_sql]
    # These are the CORRECT paths for common business queries
    CRITICAL_PATHS = {
        ("Client", "Order"): {
            "path": ["Client", "ClientAgreement", "Order"],
            "joins": [
                ("ClientAgreement", "ca", "ca.ClientID = c.ID"),
                ("Order", "o", "o.ClientAgreementID = ca.ID"),
            ]
        },
        ("Order", "Client"): {
            "path": ["Order", "ClientAgreement", "Client"],
            "joins": [
                ("ClientAgreement", "ca", "o.ClientAgreementID = ca.ID"),
                ("Client", "c", "ca.ClientID = c.ID"),
            ]
        },
        ("Client", "Sale"): {
            "path": ["Client", "ClientAgreement", "Sale"],
            "joins": [
                ("ClientAgreement", "ca", "ca.ClientID = c.ID"),
                ("Sale", "s", "s.ClientAgreementID = ca.ID"),
            ]
        },
        ("Product", "Client"): {
            "path": ["Product", "OrderItem", "Order", "ClientAgreement", "Client"],
            "joins": [
                ("OrderItem", "oi", "oi.ProductID = p.ID"),
                ("Order", "o", "oi.OrderID = o.ID"),
                ("ClientAgreement", "ca", "o.ClientAgreementID = ca.ID"),
                ("Client", "c", "ca.ClientID = c.ID"),
            ]
        },
        ("Client", "Product"): {
            "path": ["Client", "ClientAgreement", "Order", "OrderItem", "Product"],
            "joins": [
                ("ClientAgreement", "ca", "ca.ClientID = c.ID"),
                ("Order", "o", "o.ClientAgreementID = ca.ID"),
                ("OrderItem", "oi", "oi.OrderID = o.ID"),
                ("Product", "p", "oi.ProductID = p.ID"),
            ]
        },
        # Sale <-> Order paths (direct FK)
        ("Sale", "Order"): {
            "path": ["Sale", "Order"],
            "joins": [("Order", "o", "s.OrderID = o.ID")]
        },
        ("Order", "Sale"): {
            "path": ["Order", "Sale"],
            "joins": [("Sale", "s", "o.ID = s.OrderID")]
        },
        # Order <-> Product paths (via OrderItem)
        ("Order", "Product"): {
            "path": ["Order", "OrderItem", "Product"],
            "joins": [
                ("OrderItem", "oi", "o.ID = oi.OrderID"),
                ("Product", "p", "oi.ProductID = p.ID"),
            ]
        },
        ("Product", "Order"): {
            "path": ["Product", "OrderItem", "Order"],
            "joins": [
                ("OrderItem", "oi", "p.ID = oi.ProductID"),
                ("Order", "o", "oi.OrderID = o.ID"),
            ]
        },
        # Client <-> Debt paths (direct FK)
        ("Client", "Debt"): {
            "path": ["Client", "Debt"],
            "joins": [("Debt", "d", "c.ID = d.ClientID")]
        },
        ("Debt", "Client"): {
            "path": ["Debt", "Client"],
            "joins": [("Client", "c", "d.ClientID = c.ID")]
        },
        # Client <-> Payment paths (direct FK)
        ("Client", "Payment"): {
            "path": ["Client", "Payment"],
            "joins": [("Payment", "pay", "c.ID = pay.ClientID")]
        },
        ("Payment", "Client"): {
            "path": ["Payment", "Client"],
            "joins": [("Client", "c", "pay.ClientID = c.ID")]
        },
        # Product <-> ProductAvailability paths (direct FK)
        ("Product", "ProductAvailability"): {
            "path": ["Product", "ProductAvailability"],
            "joins": [("ProductAvailability", "pa", "p.ID = pa.ProductID")]
        },
        ("ProductAvailability", "Product"): {
            "path": ["ProductAvailability", "Product"],
            "joins": [("Product", "p", "pa.ProductID = p.ID")]
        },
        # ProductAvailability <-> Storage paths (direct FK)
        ("ProductAvailability", "Storage"): {
            "path": ["ProductAvailability", "Storage"],
            "joins": [("Storage", "st", "pa.StorageID = st.ID")]
        },
        ("Storage", "ProductAvailability"): {
            "path": ["Storage", "ProductAvailability"],
            "joins": [("ProductAvailability", "pa", "st.ID = pa.StorageID")]
        },
    }

    # Standard table aliases (consistent across queries)
    TABLE_ALIASES = {
        "Client": "c",
        "ClientAgreement": "ca",
        "Order": "o",
        "OrderItem": "oi",
        "Product": "p",
        "Sale": "s",
        "Region": "r",
        "ProductAvailability": "pa",
        "Warehouse": "w",
        "Storage": "st",
        "User": "u",
        "Payment": "pay",
        "Invoice": "inv",
        "Supplier": "sup",
        "Category": "cat",
        "Debt": "d",
        "ClientBalanceMovement": "cbm",
    }

    # Tables that need brackets (reserved words)
    RESERVED_TABLES = {"Order", "User", "Group", "Index", "Key", "Plan"}

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialize the join path graph.

        Args:
            schema: Database schema dict with tables and foreign_keys.
                   If None, call build_from_schema() later.
        """
        # Adjacency list: table -> [(neighbor_table, join_condition, fk_columns)]
        self.graph: Dict[str, List[Tuple[str, str, Tuple[str, str]]]] = defaultdict(list)

        # Reverse lookup: table -> tables that reference it
        self.referenced_by: Dict[str, List[str]] = defaultdict(list)

        # Path cache: (source, target) -> path
        self._path_cache: Dict[Tuple[str, str], Optional[List[Dict]]] = {}

        # Statistics
        self.table_count = 0
        self.edge_count = 0

        if schema:
            self.build_from_schema(schema)

    def build_from_schema(self, schema: Dict[str, Any]) -> None:
        """Build graph from database schema FK relationships.

        Args:
            schema: Schema dict with structure:
                {
                    "tables": {
                        "TableName": {
                            "foreign_keys": [
                                {
                                    "columns": ["ColumnName"],
                                    "referred_table": "OtherTable",
                                    "referred_columns": ["ID"]
                                }
                            ]
                        }
                    }
                }
        """
        self.graph.clear()
        self.referenced_by.clear()
        self._path_cache.clear()

        tables = schema.get("tables", {})
        self.table_count = len(tables)
        skipped_fks = 0

        for table_name, table_info in tables.items():
            fks = table_info.get("foreign_keys", [])

            for fk in fks:
                parent_cols = fk.get("columns", [])
                referred_table = fk.get("referred_table", "")
                referred_cols = fk.get("referred_columns", [])

                if not referred_table or not parent_cols or not referred_cols:
                    skipped_fks += 1
                    logger.debug(f"Skipped malformed FK in {table_name}: {fk}")
                    continue

                # Normalize table names (remove dbo. prefix if present)
                parent = self._normalize_table(table_name)
                child = self._normalize_table(referred_table)

                parent_col = parent_cols[0]
                child_col = referred_cols[0]

                # Create join condition string
                # Format: {parent_alias}.{col} = {child_alias}.{col}
                join_info = (parent_col, child_col)

                # Add edge: parent -> child (FK direction)
                self.graph[parent].append((child, "fk", join_info))

                # Add reverse edge: child -> parent (for bidirectional search)
                self.graph[child].append((parent, "ref", (child_col, parent_col)))

                # Track references
                self.referenced_by[child].append(parent)

                self.edge_count += 1

        logger.info(f"JoinPathGraph built: {self.table_count} tables, {self.edge_count} edges, {skipped_fks} skipped FKs")

    def _normalize_table(self, table_name: str) -> str:
        """Normalize table name (remove dbo., brackets)."""
        return normalize_table_name(table_name)

    def _get_alias(self, table_name: str, used_aliases: Set[str]) -> str:
        """Get unique alias for table."""
        table = self._normalize_table(table_name)

        # Check predefined aliases
        if table in self.TABLE_ALIASES:
            alias = self.TABLE_ALIASES[table]
            if alias not in used_aliases:
                return alias

        # Generate from first letter(s)
        base = table[0].lower()
        alias = base
        counter = 1

        while alias in used_aliases:
            alias = f"{base}{counter}"
            counter += 1

        return alias

    def _format_table_name(self, table: str) -> str:
        """Format table name for SQL (add brackets if reserved word)."""
        if table in self.RESERVED_TABLES:
            return f"dbo.[{table}]"
        return f"dbo.{table}"

    def _critical_path_to_joins(self, key: Tuple[str, str]) -> List[Dict]:
        """Convert a CRITICAL_PATHS entry to standard join format.

        Args:
            key: Tuple of (source, target) table names

        Returns:
            List of join dicts in standard format
        """
        critical = self.CRITICAL_PATHS[key]
        path = critical["path"]
        joins_def = critical["joins"]

        # Build table_to_alias mapping from path
        used_aliases = set()
        table_to_alias = {}

        # First table (source) gets its standard alias
        source = path[0]
        source_alias = self._get_alias(source, used_aliases)
        used_aliases.add(source_alias)
        table_to_alias[source] = source_alias

        joins = []
        for table, alias, condition in joins_def:
            used_aliases.add(alias)
            table_to_alias[table] = alias

            # Determine from_table (the previous table in path)
            table_idx = path.index(table)
            from_table = path[table_idx - 1] if table_idx > 0 else source
            from_alias = table_to_alias.get(from_table, source_alias)

            joins.append({
                "table": table,
                "table_formatted": self._format_table_name(table),
                "alias": alias,
                "join_type": "JOIN",
                "condition": condition,
                "from_table": from_table,
                "from_alias": from_alias,
            })

        return joins

    def find_join_path(self, source: str, target: str) -> Optional[List[Dict]]:
        """Find shortest join path between two tables using BFS.

        Args:
            source: Source table name
            target: Target table name

        Returns:
            List of join steps, each with:
            {
                "table": table name,
                "alias": table alias,
                "join_type": "JOIN" or "LEFT JOIN",
                "condition": "alias1.col = alias2.col"
            }
            Returns None if no path exists.
        """
        source = self._normalize_table(source)
        target = self._normalize_table(target)

        if source == target:
            return []

        # Check cache
        cache_key = (source, target)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Check CRITICAL_PATHS first (business-critical paths override BFS)
        if cache_key in self.CRITICAL_PATHS:
            result = self._critical_path_to_joins(cache_key)
            self._path_cache[cache_key] = result
            logger.debug(f"Using critical path for {source} -> {target}")
            return result

        # BFS for shortest path
        # Queue: (current_table, path_so_far, visited_set)
        queue = deque([(source, [], {source})])

        while queue:
            current, path, visited = queue.popleft()

            for neighbor, edge_type, join_info in self.graph.get(current, []):
                if neighbor in visited:
                    continue

                # Build path step
                step = {
                    "from_table": current,
                    "to_table": neighbor,
                    "edge_type": edge_type,
                    "join_info": join_info,  # (from_col, to_col)
                }
                new_path = path + [step]

                if neighbor == target:
                    # Found! Convert to join template
                    result = self._path_to_joins(source, new_path)
                    self._path_cache[cache_key] = result
                    return result

                new_visited = visited | {neighbor}
                queue.append((neighbor, new_path, new_visited))

        # No path found
        self._path_cache[cache_key] = None
        return None

    def _path_to_joins(self, source: str, path: List[Dict]) -> List[Dict]:
        """Convert BFS path to JOIN statements with aliases."""
        if not path:
            return []

        used_aliases = set()
        table_to_alias = {}

        # Assign alias to source
        source_alias = self._get_alias(source, used_aliases)
        used_aliases.add(source_alias)
        table_to_alias[source] = source_alias

        joins = []

        for step in path:
            from_table = step["from_table"]
            to_table = step["to_table"]
            from_col, to_col = step["join_info"]

            # Get or assign aliases
            from_alias = table_to_alias.get(from_table)
            if not from_alias:
                from_alias = self._get_alias(from_table, used_aliases)
                used_aliases.add(from_alias)
                table_to_alias[from_table] = from_alias

            to_alias = self._get_alias(to_table, used_aliases)
            used_aliases.add(to_alias)
            table_to_alias[to_table] = to_alias

            # Build join condition
            # For FK edge: child.FK = parent.PK → from_alias.from_col = to_alias.to_col
            # For ref edge: parent.PK = child.FK → to_alias.to_col = from_alias.from_col
            if step["edge_type"] == "fk":
                condition = f"{from_alias}.{from_col} = {to_alias}.{to_col}"
            else:  # ref - reverse
                condition = f"{to_alias}.{to_col} = {from_alias}.{from_col}"

            joins.append({
                "table": to_table,
                "table_formatted": self._format_table_name(to_table),
                "alias": to_alias,
                "join_type": "JOIN",
                "condition": condition,
                "from_table": from_table,
                "from_alias": from_alias,
            })

        return joins

    def get_join_template(
        self,
        tables: List[str],
        main_table: Optional[str] = None
    ) -> str:
        """Generate complete JOIN template for a set of tables.

        Args:
            tables: List of table names to join
            main_table: Primary table for FROM clause (auto-detected if None)

        Returns:
            SQL template string with FROM and JOINs
        """
        if not tables:
            return ""

        # Normalize all table names
        tables = [self._normalize_table(t) for t in tables]
        tables = list(dict.fromkeys(tables))  # Remove duplicates, preserve order

        if len(tables) == 1:
            alias = self._get_alias(tables[0], set())
            return f"FROM {self._format_table_name(tables[0])} {alias}"

        # Determine main table (first one, or specified)
        if main_table:
            main_table = self._normalize_table(main_table)
            if main_table in tables:
                tables.remove(main_table)
                tables.insert(0, main_table)

        main = tables[0]
        used_aliases = set()
        table_to_alias = {}

        # Assign main table alias
        main_alias = self._get_alias(main, used_aliases)
        used_aliases.add(main_alias)
        table_to_alias[main] = main_alias

        # Build joins for remaining tables
        join_lines = []
        included = {main}

        for table in tables[1:]:
            if table in included:
                continue

            # Find path from any included table to this table
            path_found = False
            for source in list(included):
                path = self.find_join_path(source, table)
                if path:
                    # Add all tables in path
                    for join in path:
                        join_table = join["table"]
                        if join_table in included:
                            continue

                        # Get consistent alias
                        if join_table in table_to_alias:
                            alias = table_to_alias[join_table]
                        else:
                            alias = self._get_alias(join_table, used_aliases)
                            used_aliases.add(alias)
                            table_to_alias[join_table] = alias

                        # Update condition with current aliases
                        from_alias = table_to_alias.get(join["from_table"], join["from_alias"])
                        condition = join["condition"]
                        # Replace aliases in condition
                        old_from = join["from_alias"]
                        old_to = join["alias"]
                        condition = condition.replace(f"{old_from}.", f"{from_alias}.")
                        condition = condition.replace(f"{old_to}.", f"{alias}.")

                        table_formatted = self._format_table_name(join_table)
                        join_lines.append(f"JOIN {table_formatted} {alias} ON {condition}")
                        included.add(join_table)

                    path_found = True
                    break

            if not path_found:
                logger.warning(f"No join path found to table: {table}")

        # Build complete template
        result = [f"FROM {self._format_table_name(main)} {main_alias}"]
        result.extend(join_lines)

        return "\n".join(result)

    def get_join_template_for_prompt(
        self,
        tables: List[str],
        include_comments: bool = True
    ) -> str:
        """Generate JOIN template formatted for LLM prompt.

        Args:
            tables: Tables that need to be joined
            include_comments: Add explanatory comments

        Returns:
            Formatted template string for prompt injection
        """
        if len(tables) < 2:
            return ""

        template = self.get_join_template(tables)

        if not template:
            return ""

        if include_comments:
            tables_str = ", ".join(tables[:5])
            if len(tables) > 5:
                tables_str += f"... (+{len(tables)-5} more)"

            header = f"-- JOIN TEMPLATE for [{tables_str}]:\n"
            header += "-- Use this EXACT pattern for joins!\n"
            return header + template

        return template

    def get_critical_paths(self) -> Dict[Tuple[str, str], str]:
        """Get pre-computed paths for most common table pairs.

        Returns:
            Dict mapping (source, target) to join template
        """
        # Common business query patterns
        critical_pairs = [
            ("Client", "Order"),
            ("Client", "Region"),
            ("Product", "Order"),
            ("Product", "OrderItem"),
            ("Order", "OrderItem"),
            ("Client", "Sale"),
            ("Product", "ProductAvailability"),
            ("Client", "Payment"),
        ]

        result = {}
        for source, target in critical_pairs:
            if source in self.graph and target in self.graph:
                template = self.get_join_template([source, target], main_table=source)
                if template:
                    result[(source, target)] = template

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "table_count": self.table_count,
            "edge_count": self.edge_count,
            "cached_paths": len(self._path_cache),
            "tables_with_fks": len([t for t in self.graph if self.graph[t]]),
        }

    def clear_cache(self) -> None:
        """Clear the path cache."""
        self._path_cache.clear()


# Singleton instance (lazily initialized)
_join_graph: Optional[JoinPathGraph] = None


def get_join_graph(schema: Optional[Dict[str, Any]] = None) -> JoinPathGraph:
    """Get or create the singleton JoinPathGraph instance.

    Args:
        schema: Schema dict (required on first call)

    Returns:
        JoinPathGraph instance
    """
    global _join_graph

    if _join_graph is None:
        if schema is None:
            raise ValueError("Schema required for first initialization")
        _join_graph = JoinPathGraph(schema)
    elif schema is not None:
        # Rebuild if schema provided
        _join_graph.build_from_schema(schema)

    return _join_graph
