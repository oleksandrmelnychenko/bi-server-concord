"""SQL generation agent using local LLM via Ollama."""
import re
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from loguru import logger
import ollama

from config import settings
from schema_extractor import SchemaExtractor
from table_selector import TableSelector
from training_data import QueryExampleRetriever
from join_path_graph import JoinPathGraph, get_join_graph, normalize_table_name


class SQLAgent:
    """Generate and execute SQL queries using local LLM."""

    # SQL result cache: {cache_key: (results, timestamp)}
    _result_cache: Dict[str, tuple] = {}
    RESULT_CACHE_TTL = 300  # 5 minutes

    # LLM SQL generation cache: {question_hash: (sql, timestamp)}
    # Saves 1-5 seconds per cached hit by avoiding LLM call
    _sql_generation_cache: Dict[str, tuple] = {}
    SQL_GENERATION_CACHE_TTL = 3600  # 1 hour (SQL templates are stable)

    # LLM generation parameters for faster, more deterministic SQL (defaults)
    LLM_OPTIONS_DEFAULT = {
        "temperature": 0.1,    # Low for deterministic SQL generation
        "num_predict": 2048,   # Increased for complex queries with JOINs, CTEs, GROUP BY
        "top_k": 10,           # Focused decoding
        "top_p": 0.9,          # Nucleus sampling
    }

    def __init__(
        self,
        engine: Optional[Engine] = None,
        schema_extractor: Optional[SchemaExtractor] = None,
        table_selector: Optional[TableSelector] = None,
        query_example_retriever: Optional[QueryExampleRetriever] = None,
        ollama_model: Optional[str] = None,
    ):
        self.engine = engine or create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=False,
            pool_recycle=1800,
        )
        self.schema_extractor = schema_extractor or SchemaExtractor(self.engine)
        self.table_selector = table_selector or TableSelector(self.schema_extractor)
        # Force refresh to ensure FK coverage and embeddings are current
        self.table_selector.index_schema(force_refresh=True)

        # Query example retriever for few-shot RAG
        self.example_retriever = query_example_retriever or QueryExampleRetriever(
            db_path=settings.query_examples_db)
        if self.example_retriever.is_available():
            stats = self.example_retriever.get_stats()
            logger.info(f"Query examples loaded: {stats['total_examples']} examples")
        else:
            logger.warning("Query examples not available")

        self.ollama_model = ollama_model or settings.ollama_model

        # Sync and async Ollama clients
        self.ollama_client = ollama.Client(host=settings.ollama_base_url)
        self.ollama_async_client = ollama.AsyncClient(host=settings.ollama_base_url)

        # Runtime-configurable LLM options (env-driven)
        self.llm_options = {
            "temperature": float(settings.llm_temperature or self.LLM_OPTIONS_DEFAULT["temperature"]),
            "num_predict": int(settings.llm_num_predict or self.LLM_OPTIONS_DEFAULT["num_predict"]),
            "top_k": int(settings.llm_top_k or self.LLM_OPTIONS_DEFAULT["top_k"]),
            "top_p": float(settings.llm_top_p or self.LLM_OPTIONS_DEFAULT["top_p"]),
        }

        # Initialize JoinPathGraph for dynamic join path computation
        schema: Optional[Dict[str, Any]] = None
        try:
            schema = self.schema_extractor.extract_full_schema(force_refresh=True)
            if not schema or not schema.get("tables"):
                raise ValueError("Schema extraction returned empty or malformed data")
            self.join_graph = JoinPathGraph(schema)
            stats = self.join_graph.get_stats()
            logger.info(f"JoinPathGraph initialized: {stats}")
            if stats["edge_count"] < 100:
                logger.warning(f"JoinPathGraph has few edges ({stats['edge_count']}), FK extraction may have failed")
            # Precompute join templates (priority-ordered, capped via settings)
            # Disable precompute for faster startup (compute on-demand instead)
            self.precomputed_join_templates = {}
            logger.info("Join templates will be computed on-demand (precompute disabled for fast startup)")
        except Exception as e:
            logger.error(f"Failed to initialize JoinPathGraph: {e}")
            self.join_graph = None  # Graceful degradation
            self.precomputed_join_templates = {}

        # Cache for dynamic relationships (generated from schema FKs)
        self._relationships_cache: Optional[str] = None
        self._join_rulebook_cache: Optional[str] = None

        # Load external prompt templates (with fallback to inline)
        self._sql_prompt_template = self._load_prompt_template(settings.sql_prompt_path)
        self._system_prompt_template = self._load_prompt_template(settings.system_prompt_path)
        if self._sql_prompt_template:
            logger.info(f"Loaded SQL prompt template from {settings.sql_prompt_path}")
        if self._system_prompt_template:
            logger.info(f"Loaded system prompt template from {settings.system_prompt_path}")

        # Schema index for validation
        self._table_columns: Dict[str, set] = {}
        self._fk_pairs: set = set()
        if schema:
            try:
                self._build_schema_index(schema)
            except Exception as e:
                logger.warning(f"Failed to build schema index for validation: {e}")

    def _load_prompt_template(self, path: str) -> Optional[str]:
        """Load prompt template from file. Returns None if file doesn't exist or is unreadable."""
        try:
            prompt_path = Path(path)
            if not prompt_path.is_absolute():
                # Resolve relative to this file's directory
                prompt_path = Path(__file__).parent / path
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8").strip()
            logger.debug(f"Prompt template not found: {prompt_path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load prompt template from {path}: {e}")
            return None

    def _canonical_pair_key(self, t1: str, t2: str) -> str:
        """Stable key for unordered table pair."""
        return "||".join(sorted([normalize_table_name(t1), normalize_table_name(t2)]))

    def _build_schema_index(self, schema: Dict[str, Any]) -> None:
        """Build lookup maps for columns and FK pairs for validation."""
        for table_name, info in schema.get("tables", {}).items():
            norm = normalize_table_name(table_name)
            cols = set(c.get("name", "").replace("[", "").replace("]", "") for c in info.get("columns", []))
            self._table_columns[norm] = cols
            for fk in info.get("foreign_keys", []):
                parent_col = fk.get("columns", [None])[0]
                ref_table = fk.get("referred_table", "")
                ref_col = fk.get("referred_columns", [None])[0]
                if parent_col and ref_table and ref_col:
                    parent_col = parent_col.replace("[", "").replace("]", "")
                    ref_table_norm = normalize_table_name(ref_table)
                    ref_col = ref_col.replace("[", "").replace("]", "")
                    self._fk_pairs.add((norm, parent_col, ref_table_norm, ref_col))
                    # Allow reverse direction for validation convenience
                    self._fk_pairs.add((ref_table_norm, ref_col, norm, parent_col))
        # Include views so validation does not reject them as missing tables
        for view_name, info in schema.get("views", {}).items():
            norm = normalize_table_name(view_name)
            cols = set(c.get("name", "").replace("[", "").replace("]", "") for c in info.get("columns", []))
            self._table_columns[norm] = cols
        logger.info(f"Schema index built: {len(self._table_columns)} tables, {len(self._fk_pairs)//2} FK edges")

    def _get_relationships_from_schema(self) -> str:
        """Generate TABLE RELATIONSHIPS dynamically from schema foreign keys."""
        if self._relationships_cache:
            return self._relationships_cache

        schema = self.schema_extractor.extract_full_schema()
        relationships = []

        for table_name, table_info in schema["tables"].items():
            fks = table_info.get("foreign_keys", [])
            for fk in fks:
                parent_cols = ", ".join(fk.get("columns", []))
                referred_table = fk.get("referred_table", "")
                referred_cols = ", ".join(fk.get("referred_columns", []))
                if referred_table:
                    # Format: TableA.ColA -> TableB.ColB
                    rel = f"- dbo.{table_name}.{parent_cols} -> {referred_table}.{referred_cols}"
                    relationships.append(rel)

        if relationships:
            result = "ALL TABLE RELATIONSHIPS (from schema FKs):\n" + "\n".join(sorted(relationships))
        else:
            result = ""

        self._relationships_cache = result
        logger.info(f"Generated {len(relationships)} relationships from schema FKs")
        return result

    def _alias_for(self, table: str) -> str:
        """Best-effort alias for rulebook display (does not enforce uniqueness)."""
        t = normalize_table_name(table)
        return JoinPathGraph.TABLE_ALIASES.get(t, t[:3].lower())

    def _get_join_rulebook(self, max_chars: Optional[int] = None) -> str:
        """Generate a compact JOIN RULEBOOK listing all FK edges with aliases.

        Set max_chars=None to include all edges without truncation.
        """
        if self._join_rulebook_cache:
            return self._join_rulebook_cache

        schema = self.schema_extractor.extract_full_schema()
        lines = []
        seen = set()

        for table_name, table_info in schema["tables"].items():
            parent_alias = self._alias_for(table_name)
            for fk in table_info.get("foreign_keys", []):
                child_cols = ", ".join(fk.get("columns", []))
                ref_table = fk.get("referred_table", "")
                ref_cols = ", ".join(fk.get("referred_columns", []))
                if not ref_table:
                    continue
                ref_alias = self._alias_for(ref_table)
                key = f"{table_name}:{child_cols}->{ref_table}:{ref_cols}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {parent_alias}.{child_cols} -> {ref_alias}.{ref_cols}  ({table_name} -> {ref_table})")

        if not lines:
            return ""

        # Priority sort: core/critical hubs first, then alphabetically
        def priority(line: str) -> tuple:
            # Extract table names from line tail "(TableA -> TableB)"
            if "(" in line and "->" in line:
                tail = line.split("(")[-1].rstrip(")")
                parts = tail.split("->")
                if len(parts) == 2:
                    a = parts[0].strip()
                    b = parts[1].strip()
                else:
                    a = b = ""
            else:
                a = b = ""
            core_hubs = {"Client", "ClientAgreement", "Order", "OrderItem", "Product", "Sale", "ProductAvailability", "Region"}
            score = 0
            if a.replace("dbo.", "") in core_hubs:
                score -= 2
            if b.replace("dbo.", "") in core_hubs:
                score -= 2
            return (score, a, b, line)

        lines = sorted(lines, key=priority)
        header = "######################## JOIN RULEBOOK (FK edges) ########################\n"
        clipped = []
        if max_chars is not None:
            budget = max_chars - len(header)
            used = 0
            for line in lines:
                if used + len(line) + 1 > budget:
                    clipped.append(f"... ({len(lines) - len(clipped)} more edges clipped)")
                    break
                clipped.append(line)
                used += len(line) + 1
        else:
            clipped = lines

        rulebook = header + "\n".join(clipped)
        self._join_rulebook_cache = rulebook
        logger.info(f"Join rulebook generated with {len(clipped)} edges" + (f" (budget {max_chars} chars)" if max_chars else ""))
        return rulebook

    def _precompute_join_templates(self, schema: Dict[str, Any], max_pairs: Optional[int] = None) -> Dict[str, str]:
        """Precompute join templates for many table pairs to reduce LLM work.

        Priority order:
        1) Direct FK-linked pairs
        2) Critical business paths
        3) Core table combinations
        4) High-row-count table combinations (top 30 by rows, cross-join pairwise)
        If max_pairs is None, attempt full coverage of all table pairs (may be heavy).
        """
        if not self.join_graph:
            return {}

        pairs = []
        seen = set()

        # FK-linked pairs from schema
        for table_name, table_info in schema.get("tables", {}).items():
            for fk in table_info.get("foreign_keys", []):
                parent = normalize_table_name(table_name)
                referred = normalize_table_name(fk.get("referred_table", ""))
                if not referred:
                    continue
                key = self._canonical_pair_key(parent, referred)
                if key not in seen:
                    seen.add(key)
                    pairs.append((parent, referred))

        # Critical business paths from JoinPathGraph
        for source, target in getattr(JoinPathGraph, "CRITICAL_PATHS", {}).keys():
            key = self._canonical_pair_key(source, target)
            if key not in seen:
                seen.add(key)
                pairs.append((normalize_table_name(source), normalize_table_name(target)))

        # Pairwise combinations of core tables (broaden coverage)
        core_tables = [normalize_table_name(t.replace("dbo.", "").replace("[", "").replace("]", ""))
                       for t in TableSelector.CORE_TABLES]
        for i in range(len(core_tables)):
            for j in range(i + 1, len(core_tables)):
                key = self._canonical_pair_key(core_tables[i], core_tables[j])
                if key not in seen:
                    seen.add(key)
                    pairs.append((core_tables[i], core_tables[j]))

        templates: Dict[str, str] = {}
        # High-row-count tables (top 30) pairwise to bias important joins
        row_sorted = sorted(
            [(info.get("row_count") or 0, name) for name, info in schema.get("tables", {}).items()],
            reverse=True
        )
        top_row_tables = [name for _, name in row_sorted[:30]]
        for i in range(len(top_row_tables)):
            for j in range(i + 1, len(top_row_tables)):
                key = self._canonical_pair_key(top_row_tables[i], top_row_tables[j])
                if key not in seen:
                    seen.add(key)
                    pairs.append((top_row_tables[i], top_row_tables[j]))

        # If not capped, add all table pairs for full coverage
        if max_pairs is None:
            table_names = list(schema.get("tables", {}).keys())
            for i in range(len(table_names)):
                for j in range(i + 1, len(table_names)):
                    key = self._canonical_pair_key(table_names[i], table_names[j])
                    if key not in seen:
                        seen.add(key)
                        pairs.append((table_names[i], table_names[j]))

        limited_pairs = pairs if max_pairs is None else pairs[:max_pairs]
        skipped = 0

        for a, b in limited_pairs:
            key = self._canonical_pair_key(a, b)
            try:
                template = self.join_graph.get_join_template([a, b], main_table=a)
                if template:
                    templates[key] = template
            except Exception as e:
                skipped += 1
                logger.debug(f"Join template precompute failed for {a}-{b}: {e}")

        logger.info(f"Precomputed {len(templates)} join templates (pairs capped at {max_pairs}, skipped={skipped})")
        return templates

    def _get_join_templates_for_tables(self, tables: List[str]) -> str:
        """Generate JOIN templates for the selected tables using JoinPathGraph.

        This replaces hardcoded join fixes with dynamic path computation.
        The graph algorithm finds optimal paths between any table pair.

        Args:
            tables: List of table names selected by RAG

        Returns:
            Formatted JOIN templates for prompt injection
        """
        if not tables or len(tables) < 2:
            logger.debug(f"Skipping join templates: {len(tables) if tables else 0} tables selected")
            return ""

        if not self.join_graph:
            logger.debug("JoinPathGraph unavailable; skipping join templates")
            return ""

        templates = []

        # Critical paths that are commonly needed (pre-compute for consistency)
        # These match the CRITICAL_PATHS defined in JoinPathGraph
        critical_pairs = [
            ("Client", "Order", "Get orders for a client"),
            ("Client", "Region", "Get client's region name"),
            ("Order", "OrderItem", "Get order line items"),
            ("Product", "OrderItem", "Get product order details"),
            ("Product", "ProductAvailability", "Get product stock levels"),
            ("ProductAvailability", "Storage", "Get warehouse stock"),
            ("Client", "Debt", "Get client debts"),
            ("Client", "Payment", "Get client payments"),
            ("Sale", "Order", "Get sale order details"),
            ("Order", "Product", "Get products in order"),
            ("Client", "Sale", "Get sales for a client"),
        ]

        # Check which critical paths are relevant based on selected tables
        table_set = set(normalize_table_name(t) for t in tables)

        for source, target, description in critical_pairs:
            if source in table_set and target in table_set:
                template = self.join_graph.get_join_template([source, target], main_table=source)
                if template:
                    templates.append(f"-- {description}:\n{template}")
                else:
                    logger.warning(f"Failed to generate template for {source}→{target}")

        # Also generate template for all selected tables together
        if len(tables) > 2:
            # Get just the core tables from selection for the main template
            core_in_selection = [t for t in tables[:5] if t in table_set]
            if core_in_selection:
                full_template = self.join_graph.get_join_template(core_in_selection)
                if full_template and full_template not in "\n".join(templates):
                    templates.append(f"-- Combined join for selected tables:\n{full_template}")

        if templates:
            header = """
################################################################################
#                                                                              #
#   STOP! READ THESE JOIN TEMPLATES FIRST!                                     #
#                                                                              #
#   Order.ClientID DOES NOT EXIST!                                             #
#                                                                              #
#   To join Order with Client, you MUST use:                                   #
#   Client -> ClientAgreement -> Order                                         #
#                                                                              #
#   Copy the JOIN patterns below EXACTLY. Do NOT invent your own joins!        #
#                                                                              #
################################################################################

MANDATORY JOIN TEMPLATES:
"""
            return header + "\n\n".join(templates)

        logger.debug(f"No critical path templates matched for: {list(table_set)[:5]}")
        return ""

    def _get_precomputed_join_templates_for_tables(self, tables: List[str], limit: Optional[int] = None) -> str:
        """Retrieve precomputed join templates for selected table pairs.

        If limit is None, include all available templates for the selected pairs.
        """
        if not tables or not self.precomputed_join_templates:
            return ""

        selected = [normalize_table_name(t) for t in tables]
        templates = []
        used_keys = set()

        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                key = self._canonical_pair_key(selected[i], selected[j])
                if key in self.precomputed_join_templates and key not in used_keys:
                    templates.append(self.precomputed_join_templates[key])
                    used_keys.add(key)
                    if limit and len(templates) >= limit:
                        break
            if limit and len(templates) >= limit:
                break

        if not templates:
            return ""

        header = """
######################## PRECOMPUTED JOIN TEMPLATES ########################
These are ready-made FROM/JOIN blocks for selected table pairs. Copy/paste
directly. Do NOT invent alternative paths if a template exists below.
"""
        return header + "\n\n".join(templates)

    def _contains_any(self, text: str, tokens: List[str]) -> bool:
        """Return True if any token is present in text (case-insensitive)."""
        text_lower = text.lower()
        return any(token.lower() in text_lower for token in tokens)

    def _has_cyrillic(self, text: str) -> bool:
        """Detect Cyrillic characters in text."""
        return any("\u0400" <= ch <= "\u04FF" for ch in text)

    def _extract_tables_from_sql(self, sql: str) -> set:
        """Extract normalized table names from FROM/JOIN clauses."""
        tables = set()
        for match in re.finditer(r'\b(?:FROM|JOIN)\s+(?:dbo\.)?\[?(\w+)\]?', sql, re.IGNORECASE):
            tables.add(normalize_table_name(match.group(1)).lower())
        return tables

    def _infer_domain_filter(self, question: str) -> Optional[str]:
        """Infer query domain for example retrieval filtering."""
        q = question.lower()
        if self._contains_any(q, [
            "client", "clients", "customer", "customers",
            "клієнт", "клієнти", "клиент", "клиенты",
            "замовник", "замовники", "покупець", "покупці",
            "покупатель", "покупатели",
            # Client type keywords (ФОП, ТОВ, ПП)
            "фоп", "тов", "пп",
            # Buy/purchase verbs (all tenses)
            "купляє", "купляють", "купує", "купують",
            "купляв", "купляли", "купив", "купили", "закупив", "закупили",
            # Region names (for region-based purchases)
            "київ", "львів", "одеса", "харків", "дніпро", "хмельницьк",
            "регіон", "регіони", "область", "області",
        ]):
            return "customers"
        # Storage/Warehouse domain - check BEFORE products to catch "товару на складі"
        if self._contains_any(q, [
            "storage", "warehouse", "storages", "warehouses",
            "склад", "склади", "складі", "складу", "складах",
            "відвантаж", "залишк", "залишок",
        ]):
            return "storage"
        if self._contains_any(q, [
            "product", "products",
            "\u0442\u043e\u0432\u0430\u0440", "\u0442\u043e\u0432\u0430\u0440\u0438",
            "\u043f\u0440\u043e\u0434\u0443\u043a\u0442", "\u043f\u0440\u043e\u0434\u0443\u043a\u0442\u0438",
            "\u043d\u043e\u043c\u0435\u043d\u043a\u043b\u0430\u0442\u0443\u0440",
        ]):
            return "products"
        if self._contains_any(q, [
            "sale", "sales", "revenue", "\u043e\u0431\u043e\u0440\u043e\u0442",
            "\u0432\u0438\u0440\u0443\u0447", "\u0432\u044b\u0440\u0443\u0447", "\u043f\u0440\u043e\u0434\u0430\u0436",
        ]):
            return "sales"
        # Debt/financial domain - борги, боржники, платежі
        if self._contains_any(q, [
            "debt", "debts", "debtor", "debtors", "payment", "payments",
            "receivable", "receivables", "owe", "owed", "owing",
            "\u0431\u043e\u0440\u0433", "\u0431\u043e\u0440\u0433\u0438",  # борг, борги
            "\u0431\u043e\u0440\u0436\u043d\u0438\u043a", "\u0431\u043e\u0440\u0436\u043d\u0438\u043a\u0438",  # боржник, боржники
            "\u0437\u0430\u0431\u043e\u0440\u0433\u043e\u0432\u0430\u043d", # заборгованість
            "\u043f\u043b\u0430\u0442\u0456\u0436", "\u043f\u043b\u0430\u0442\u0435\u0436",  # платіж, платеж
            "\u0432\u0438\u043d\u0435\u043d", "\u0432\u0438\u043d\u043d\u0438\u0439",  # винен, винний
        ]):
            return "financial"
        # Bank domain - банки
        if self._contains_any(q, [
            "bank", "banks",
            "\u0431\u0430\u043d\u043a", "\u0431\u0430\u043d\u043a\u0438", "\u0431\u0430\u043d\u043a\u0456\u0432",  # банк, банки, банків
        ]):
            return "customers"  # Bank is in customers domain
        return None

    def _build_intent_hints(self, question: str) -> str:
        """Build intent hints to steer SQL generation."""
        q = question.lower()
        hints = []
        client_tokens = [
            "client", "clients", "customer", "customers",
            "\u043a\u043b\u0456\u0454\u043d\u0442", "\u043a\u043b\u0456\u0454\u043d\u0442\u0438",
            "\u043a\u043b\u0438\u0435\u043d\u0442", "\u043a\u043b\u0438\u0435\u043d\u0442\u044b",
            "\u0437\u0430\u043c\u043e\u0432\u043d\u0438\043a", "\u0437\u0430\u043c\u043e\u0432\u043d\u0438\043a\u0438",
            "\u043f\u043e\u043a\u0443\u043f\u0435\u0446\u044c", "\u043f\u043e\u043a\u0443\u043f\u0446\u0456",
            "\u043f\u043e\u043a\u0443\u043f\u0430\u0442\u0435\u043b\u044c", "\u043f\u043e\u043a\u0443\u043f\u0430\u0442\u0435\u043b\u0438",
        ]
        region_tokens = [
            "region", "oblast", "\u043e\u0431\u043b", "\u043e\u0431\u043b\u0430\u0441\u0442\u044c", "\u043e\u0431\u043b\u0430\u0441\u0442\u0438",
            "\u0440\u0435\u0433\u0456\u043e\u043d", "\u0440\u0435\u0433\u0438\u043e\u043d",
            # Ukrainian region names (using stems for case-insensitive matching)
            # киє matches: київ, києві, києва, києво etc.
            "киє", "київ", "льв", "львів", "одес", "одеса",
            "харк", "харків", "дніпр", "дніпро", "хмельницьк",
            "kyiv", "lviv", "odesa", "kharkiv", "dnipro", "khmelnytskyi",
        ]
        list_tokens = [
            "show", "list", "\u043f\u043e\u043a\u0430\u0436\u0438", "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0438", "\u0432\u0438\u0432\u0435\u0434\u0438",
            "\u0441\u043f\u0438\u0441\u043e\u043a", "\u043f\u0435\u0440\u0435\u043b\u0456\u043a",
        ]

        # Debt-related tokens - борги, боржники
        debt_tokens = [
            "debt", "debts", "debtor", "debtors", "owe", "owed", "owing",
            "receivable", "receivables",
            "\u0431\u043e\u0440\u0433", "\u0431\u043e\u0440\u0433\u0438",  # борг, борги
            "\u0431\u043e\u0440\u0436\u043d\u0438\u043a", "\u0431\u043e\u0440\u0436\u043d\u0438\u043a\u0438",  # боржник, боржники
            "\u0437\u0430\u0431\u043e\u0440\u0433\u043e\u0432\u0430\u043d",  # заборгованість
            "\u0432\u0438\u043d\u0435\u043d", "\u0432\u0438\u043d\u043d\u0438\u0439",  # винен, винний
        ]

        if self._contains_any(q, client_tokens):
            hints.append("CLIENT QUERIES: Use dbo.Client table! For name search use: WHERE Name LIKE N'%name%' AND Deleted = 0. For counting: SELECT COUNT(*) FROM dbo.Client WHERE ... Use N prefix for Unicode strings!")
        if self._contains_any(q, region_tokens):
            hints.append("REGION QUERIES: Join dbo.Region ON Client.RegionID = Region.ID. CRITICAL: Region.Name contains 2-LETTER CODES, NOT full names! Map: Київ=KI, Львів=LV, Одеса=OD, Харків=XV, Хмельницький=XM, Дніпро=DP. Example: WHERE Region.Name = 'XM' for Хмельницький.")
        if self._contains_any(q, list_tokens) and self._contains_any(q, client_tokens):
            hints.append("For client lists, return Client.ID and Client.Name (and Region.Name when filtering by region).")
        # CRITICAL: Debt queries must use Debt and ClientInDebt tables, NOT Order/OrderItem!
        if self._contains_any(q, debt_tokens):
            hints.append("DEBT QUERIES: Use dbo.Debt and dbo.ClientInDebt tables! Join path: Client -> ClientInDebt (ClientID) -> Debt (DebtID). Use Debt.Total for amounts. DO NOT use Order/OrderItem for debt queries!")

        # Brand queries - бренди
        brand_tokens = [
            "brand", "brands",
            "бренд", "бренди", "брендів", "брендах",
        ]
        if self._contains_any(q, brand_tokens):
            # CRITICAL: No Brand table exists! Brand = Product.VendorCode
            if self._contains_any(q, ["популярн", "топ", "найкращ", "найбільш", "popular", "top", "best"]):
                hints.append("BRAND POPULARITY: NO Brand table! Brand = Product.VendorCode. Use: SELECT TOP 10 p.VendorCode as Brand, SUM(oi.Qty) as TotalSold FROM dbo.Product p JOIN dbo.OrderItem oi ON oi.ProductID = p.ID WHERE p.Deleted = 0 AND oi.Deleted = 0 AND p.VendorCode IS NOT NULL GROUP BY p.VendorCode ORDER BY TotalSold DESC")
            else:
                hints.append("BRAND QUERIES: NO Brand table exists! Brand = Product.VendorCode. For brand list: SELECT DISTINCT VendorCode as Brand FROM dbo.Product WHERE Deleted = 0 AND VendorCode IS NOT NULL ORDER BY VendorCode")

        # Bank queries - банки
        bank_tokens = [
            "bank", "banks",
            "банк", "банки", "банків",  # bank, banks
        ]
        if self._contains_any(q, bank_tokens):
            hints.append("BANK QUERIES: Use dbo.Bank table! Columns: ID, Name, MfoCode, EdrpouCode, City, Address, Phones. Simple query: SELECT ID, Name, City, MfoCode FROM dbo.Bank WHERE Deleted = 0 ORDER BY Name")

        # CRITICAL: Storage/Warehouse queries - especially "склад браку" (defect warehouse)
        storage_tokens = [
            "склад", "склади", "складі", "складу", "складах",
            "storage", "warehouse", "storages", "warehouses",
        ]
        # Check specifically for defect warehouse pattern
        defect_warehouse_patterns = [
            "склад брак", "складі брак", "складу брак",
            "склад браку", "складі браку", "складу браку",
            "defect warehouse", "brak warehouse",
        ]
        if any(p in q for p in defect_warehouse_patterns):
            hints.append("CRITICAL - DEFECT WAREHOUSE: 'склад браку' is a warehouse NAME (Storage.Name LIKE N'%БРАК%'), NOT 'shortage' or 'defect'! Use: JOIN dbo.Storage s ON pa.StorageID = s.ID WHERE s.Name LIKE N'%БРАК%'. Use ProductAvailability.Amount for current stock quantity.")
        elif self._contains_any(q, ["відвантаж", "shipment", "dispatch"]):
            hints.append("SHIPMENTS BY WAREHOUSE: Use dbo.Consignment table (NOT OrderItem!). Consignment has StorageID column. Query: SELECT s.Name, COUNT(c.ID) as ShipmentCount, SUM(c.TotalSum) as TotalSum FROM dbo.Consignment c JOIN dbo.Storage s ON c.StorageID = s.ID WHERE c.Deleted = 0 AND s.Deleted = 0 GROUP BY s.ID, s.Name ORDER BY ShipmentCount DESC")
        elif self._contains_any(q, storage_tokens):
            hints.append("STORAGE/WAREHOUSE QUERIES: Use dbo.Storage table. For inventory use dbo.ProductAvailability (Amount = current quantity). Join: ProductAvailability.StorageID = Storage.ID")

        # SUPPLIER QUERIES
        supplier_tokens = ["постачальник", "постачальники", "supplier", "suppliers", "закупівл", "поставк"]
        if self._contains_any(q, supplier_tokens):
            hints.append("SUPPLIER QUERIES: SupplyOrder has NO direct supplier FK. Use payments path: OutcomePaymentOrder -> SupplyOrganizationAgreement -> SupplyOrganization. Example: SELECT TOP 10 so.Name, SUM(opo.TotalPrice) AS Total FROM dbo.OutcomePaymentOrder opo JOIN dbo.SupplyOrganizationAgreement soa ON soa.ID = opo.SupplyOrganizationAgreementID JOIN dbo.SupplyOrganization so ON so.ID = soa.SupplyOrganizationID WHERE opo.Deleted = 0 AND soa.Deleted = 0 AND so.Deleted = 0 GROUP BY so.ID, so.Name ORDER BY Total DESC")

        # Delivery time
        if self._contains_any(q, ["доставк", "delivery"]) and self._contains_any(q, supplier_tokens):
            hints.append("DELIVERY TIME: SupplyOrder has OrderShippedDate and OrderArrivedDate. For average delivery time: AVG(DATEDIFF(day, s.OrderShippedDate, s.OrderArrivedDate)) WHERE s.OrderShippedDate IS NOT NULL AND s.OrderArrivedDate IS NOT NULL. Join: SupplyOrganization so -> SupplyOrganizationAgreement soa -> SupplyOrder s")

        # CLIENT TYPE - ФОП, ТОВ, ПП
        if self._contains_any(q, ["фоп", "тов", " пп ", "типах клієнтів"]):
            hints.append("CLIENT TYPE: Use CASE WHEN c.Name LIKE N'%ФОП%' THEN 'ФОП' WHEN c.Name LIKE N'%ТОВ%' THEN 'ТОВ' WHEN c.Name LIKE N'%ПП%' THEN 'ПП' ELSE 'Інші' END. GROUP BY same CASE!")

        # RETURNS
        if self._contains_any(q, ["повернен", "return"]):
            hints.append("RETURNS: OrderItem.ReturnedQty column for returned quantity")

        # STOCK WITH VALUE
        if self._contains_any(q, ["залишк", "вартіст"]) and self._contains_any(q, storage_tokens):
            hints.append("STOCK VALUE: IMPORTANT - Product has NO Price column! Show only quantity: SELECT s.Name, SUM(pa.Amount) AS Qty FROM Storage s JOIN ProductAvailability pa ON pa.StorageID = s.ID WHERE s.Deleted = 0 AND pa.Deleted = 0 GROUP BY s.ID, s.Name")

        # NEGATIVE STOCK
        if self._contains_any(q, ["від'ємн", "негатив"]) and self._contains_any(q, ["залишок", "залишки"]):
            hints.append("NEGATIVE STOCK: WHERE pa.Amount < 0. Subquery for last sold from OrderItem.")

        # PRODUCT MOVEMENT
        if self._contains_any(q, ["рух", "руху"]) and self._contains_any(q, ["товар", "продукт"]):
            hints.append("PRODUCT MOVEMENT: CRITICAL - GROUP BY must include ALL non-aggregated columns!")

        # PAYMENT QUERIES - платежі
        payment_tokens = ["платіж", "платежі", "платежів", "оплат", "payment", "payments"]
        if self._contains_any(q, payment_tokens):
            hints.append("PAYMENT QUERIES: Use dbo.IncomePaymentOrder table! Columns: ID, Amount, Created, ClientID, Deleted. Example: SELECT MONTH(ipo.Created) as Month, SUM(ipo.Amount) as Total FROM dbo.IncomePaymentOrder ipo WHERE ipo.Deleted = 0 AND YEAR(ipo.Created) = 2025 GROUP BY MONTH(ipo.Created). For client payments join: JOIN dbo.Client c ON c.ID = ipo.ClientID")

        # CATEGORY/MARGIN QUERIES - категорії, маржа
        category_tokens = ["категорі", "category", "categories", "маржа", "маржинальн", "margin"]
        if self._contains_any(q, category_tokens):
            hints.append("CATEGORY/MARGIN QUERIES: IMPORTANT - Product has NO Price column! For margin by category, use ProductGroup with ProductPricing.Price for cost: SELECT TOP 20 pg.Name AS CategoryName, SUM(oi.Qty * oi.PricePerItem) AS Revenue, SUM(oi.Qty * pp.Price) AS Cost, SUM(oi.Qty * oi.PricePerItem) - SUM(oi.Qty * pp.Price) AS Margin FROM ProductGroup pg JOIN ProductProductGroup ppg ON ppg.ProductGroupID = pg.ID JOIN Product p ON p.ID = ppg.ProductID JOIN ProductPricing pp ON pp.ProductID = p.ID JOIN OrderItem oi ON oi.ProductID = p.ID WHERE pg.Deleted = 0 AND pp.Deleted = 0 GROUP BY pg.ID, pg.Name ORDER BY Revenue DESC")

        # SUPPLY ORDER DETAILS - закупки від постачальників
        supply_detail_tokens = ["закупили", "закупівл"]
        if self._contains_any(q, supply_detail_tokens) and self._contains_any(q, ["товар", "продукт", "product"]):
            hints.append("SUPPLY/PURCHASE QUERIES: For products purchased from suppliers use: dbo.SupplyOrderItem (ProductID, Qty, PricePerItem) -> dbo.SupplyOrder (Created). Example purchased but not sold: SELECT p.Name FROM dbo.Product p JOIN dbo.SupplyOrderItem soi ON soi.ProductID = p.ID WHERE NOT EXISTS (SELECT 1 FROM dbo.OrderItem oi WHERE oi.ProductID = p.ID)")

        # OVERDUE/LATE DELIVERIES - прострочені
        if self._contains_any(q, ["простроч", "overdue", "late", "запізн"]):
            hints.append("OVERDUE DELIVERIES: SupplyOrder has IsCompleted, OrderShippedDate, OrderArrivedDate, IsOrderArrived. Overdue = IsCompleted = 0 AND DATEDIFF(day, Created, GETDATE()) > 30. Query: SELECT so.Name FROM dbo.SupplyOrganization so JOIN dbo.SupplyOrganizationAgreement soa ON soa.SupplyOrganizationID = so.ID JOIN dbo.SupplyOrder s ON s.ClientAgreementID IN (SELECT ca.ID FROM dbo.ClientAgreement ca WHERE ca.SupplyOrganizationAgreementID = soa.ID) WHERE s.IsCompleted = 0 AND s.Deleted = 0")

        # STOCK BY STORAGE - залишки по складах
        if self._contains_any(q, ["залишк", "залишок"]) and self._contains_any(q, ["склад", "складах", "складу"]):
            hints.append("STOCK BY STORAGE: IMPORTANT - Product has NO Price column! Show quantity only: SELECT s.Name AS StorageName, COUNT(DISTINCT pa.ProductID) AS ProductCount, SUM(pa.Amount) AS TotalQty FROM Storage s JOIN ProductAvailability pa ON pa.StorageID = s.ID WHERE s.Deleted = 0 AND pa.Deleted = 0 AND pa.Amount > 0 GROUP BY s.ID, s.Name ORDER BY TotalQty DESC")

        if not hints:
            return ""

        return "\n".join(f"- {hint}" for hint in hints)

    def _validate_intent_alignment(self, question: str, sql: str) -> None:
        """Validate SQL matches obvious intent cues in the question."""
        q = question.lower()
        tables = self._extract_tables_from_sql(sql)
        logger.debug(f"[Intent Validation] Question: {question[:50]}..., Tables: {tables}")

        client_tokens = [
            "client", "clients", "customer", "customers",
            "\u043a\u043b\u0456\u0454\u043d\u0442", "\u043a\u043b\u0456\u0454\u043d\u0442\u0438",
            "\u043a\u043b\u0438\u0435\u043d\u0442", "\u043a\u043b\u0438\u0435\u043d\u0442\u044b",
            "\u0437\u0430\u043c\u043e\u0432\u043d\u0438\u043a", "\u0437\u0430\u043c\u043e\u0432\u043d\u0438\u043a\u0438",
            "\u043f\u043e\u043a\u0443\u043f\u0435\u0446\u044c", "\u043f\u043e\u043a\u0443\u043f\u0446\u0456",
            "\u043f\u043e\u043a\u0443\u043f\u0430\u0442\u0435\u043b\u044c", "\u043f\u043e\u043a\u0443\u043f\u0430\u0442\u0435\u043b\u0438",
        ]
        region_tokens = [
            "region", "oblast", "\u043e\u0431\u043b", "\u043e\u0431\u043b\u0430\u0441\u0442\u044c", "\u043e\u0431\u043b\u0430\u0441\u0442\u0438",
            "\u0440\u0435\u0433\u0456\u043e\u043d", "\u0440\u0435\u0433\u0438\u043e\u043d",
            # Ukrainian region names (using stems for case-insensitive matching)
            # киє matches: київ, києві, києва, києво etc.
            "киє", "київ", "льв", "львів", "одес", "одеса",
            "харк", "харків", "дніпр", "дніпро", "хмельницьк",
            "kyiv", "lviv", "odesa", "kharkiv", "dnipro", "khmelnytskyi",
        ]

        # Check if any table contains the required keyword (e.g., "clientagreement" contains "client")
        has_client_table = any("client" in t for t in tables)
        has_region_table = any("region" in t for t in tables)

        # Check if SQL uses address-based filtering (valid alternative to Region table)
        # Use regex to handle table aliases (e.g., c.LegalAddress LIKE)
        sql_lower = sql.lower()
        has_address_filter = bool(re.search(r'address\s+(like|=)', sql_lower)) or \
                             bool(re.search(r'city\s+(like|=)', sql_lower))
        logger.debug(f"[Address Filter Check] sql_lower contains 'address': {'address' in sql_lower}, has_address_filter={has_address_filter}")

        if self._contains_any(q, client_tokens) and not has_client_table:
            raise ValueError("Intent mismatch: client query without Client table")
        # Allow region queries if they use Region table OR address-based filtering
        if self._contains_any(q, region_tokens) and not has_region_table and not has_address_filter:
            logger.warning(f"[Region Validation FAIL] region_tokens matched, has_region_table={has_region_table}, has_address_filter={has_address_filter}")
            raise ValueError("Intent mismatch: region query without Region table or address filter")

        # Bank validation - банк, банки
        bank_tokens = [
            "bank", "banks",
            "банк", "банки", "банків",
        ]
        has_bank_table = any("bank" in t for t in tables)
        has_bank_intent = self._contains_any(q, bank_tokens)
        logger.debug(f"[Bank Validation] has_bank_intent={has_bank_intent}, has_bank_table={has_bank_table}")
        if has_bank_intent and not has_bank_table:
            logger.warning(f"Bank intent detected but no Bank table! Raising validation error.")
            raise ValueError("Intent mismatch: bank query without Bank table - use dbo.Bank!")

    def generate_sql(self, question: str, max_retries: int = 0, include_explanation: bool = False) -> Dict[str, Any]:
        logger.info(f"Generating SQL for: {question}")

        # Check SQL generation cache first (saves 1-5 seconds per hit)
        cache_key = hashlib.md5(question.strip().lower().encode()).hexdigest()
        if cache_key in self._sql_generation_cache:
            cached_sql, cached_time = self._sql_generation_cache[cache_key]
            if time.time() - cached_time < self.SQL_GENERATION_CACHE_TTL:
                try:
                    self._validate_intent_alignment(question, cached_sql)
                    logger.info(f"SQL generation cache hit for: {question[:50]}...")
                    return {"question": question, "sql": cached_sql, "attempts": 0, "cached": True}
                except Exception as cache_err:
                    logger.warning(f"Cached SQL failed intent validation, regenerating: {cache_err}")
                    del self._sql_generation_cache[cache_key]

        # Use full schema context (all 300+ tables in compact form + detailed relevant tables)
        context = self.table_selector.get_full_schema_context(question)
        selected_tables = self.table_selector.get_selected_tables(question)
        sql_query = None
        attempts = []

        for attempt in range(max_retries + 1):
            try:
                prompt = self._build_prompt(question, context, attempts, selected_tables)
                response = self.ollama_client.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    options=self.llm_options
                )
                sql_query = self._extract_sql(response["response"])
                sql_query = self._fix_transliterated_names(sql_query, question)
                sql_query = self._fix_unwanted_brand_filters(sql_query, question)
                sql_query = self._enhance_top_products_sql(sql_query, question)
                self._validate_sql(sql_query)
                self._validate_intent_alignment(question, sql_query)
                logger.info(f"Generated SQL on attempt {attempt + 1}")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                attempts.append({"sql": sql_query, "error": str(e)})
                if attempt == max_retries:
                    raise ValueError(f"Failed after {max_retries + 1} attempts")

        # Cache successful SQL generation (LRU eviction at 200 entries)
        if sql_query:
            if len(self._sql_generation_cache) >= 200:
                oldest_key = next(iter(self._sql_generation_cache))
                del self._sql_generation_cache[oldest_key]
            self._sql_generation_cache[cache_key] = (sql_query, time.time())

        return {"question": question, "sql": sql_query, "attempts": len(attempts) + 1}

    def execute_sql(self, sql_query: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """Execute SQL with result caching for repeated queries."""
        if max_rows is None:
            max_rows = settings.max_rows_returned
        if settings.read_only_mode:
            self._check_read_only(sql_query)

        # Generate cache key from SQL + max_rows
        cache_key = hashlib.md5(f"{sql_query}:{max_rows}".encode()).hexdigest()

        # Check cache first
        if cache_key in self._result_cache:
            cached_result, cached_time = self._result_cache[cache_key]
            if time.time() - cached_time < self.RESULT_CACHE_TTL:
                logger.debug(f"Result cache hit: {cache_key[:8]}")
                return cached_result

        # Execute and cache
        result = self._execute_sql_impl(sql_query, max_rows)

        # Only cache successful results
        if result.get("success"):
            # Limit cache size (LRU-style eviction)
            if len(self._result_cache) >= 100:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            self._result_cache[cache_key] = (result, time.time())

        return result

    def _execute_sql_impl(self, sql_query: str, max_rows: int) -> Dict[str, Any]:
        """Internal SQL execution (called when cache misses)."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = []
                columns = list(result.keys()) if result.returns_rows else []
                if result.returns_rows:
                    for i, row in enumerate(result):
                        if i >= max_rows:
                            break
                        rows.append(dict(row._mapping))
                return {"success": True, "columns": columns, "rows": rows, "row_count": len(rows)}
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {"success": False, "error": str(e)}

    def query(self, question: str, execute: bool = True, max_rows: Optional[int] = None, include_explanation: bool = False) -> Dict[str, Any]:
        sql_result = self.generate_sql(question)
        response = {"question": question, "sql": sql_result["sql"]}
        if execute:
            response["execution"] = self.execute_sql(sql_result["sql"], max_rows)
        return response

    # ==================== ASYNC METHODS ====================
    # These allow concurrent request handling in FastAPI

    async def generate_sql_async(self, question: str, max_retries: int = 0, include_explanation: bool = False) -> Dict[str, Any]:
        """Async version of generate_sql - allows concurrent requests."""
        logger.info(f"Generating SQL (async) for: {question}")

        # Check SQL generation cache first (saves 1-5 seconds per hit)
        cache_key = hashlib.md5(question.strip().lower().encode()).hexdigest()
        if cache_key in self._sql_generation_cache:
            cached_sql, cached_time = self._sql_generation_cache[cache_key]
            if time.time() - cached_time < self.SQL_GENERATION_CACHE_TTL:
                try:
                    self._validate_intent_alignment(question, cached_sql)
                    logger.info(f"SQL generation cache hit (async) for: {question[:50]}...")
                    return {"question": question, "sql": cached_sql, "attempts": 0, "cached": True}
                except Exception as cache_err:
                    logger.warning(f"Cached SQL failed intent validation (async), regenerating: {cache_err}")
                    del self._sql_generation_cache[cache_key]

        # Use full schema context (all 300+ tables in compact form + detailed relevant tables)
        context = self.table_selector.get_full_schema_context(question)
        selected_tables = self.table_selector.get_selected_tables(question)
        sql_query = None
        attempts = []

        for attempt in range(max_retries + 1):
            try:
                prompt = self._build_prompt(question, context, attempts, selected_tables)
                # ASYNC Ollama call - doesn't block other requests
                response = await self.ollama_async_client.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    options=self.llm_options
                )
                raw_resp = response["response"]
                logger.warning(f"RAW LLM len={len(raw_resp)}: '{raw_resp[:800]}'")
                sql_query = self._extract_sql(raw_resp)
                sql_query = self._fix_transliterated_names(sql_query, question)
                sql_query = self._fix_unwanted_brand_filters(sql_query, question)
                sql_query = self._enhance_top_products_sql(sql_query, question)
                self._validate_sql(sql_query)
                self._validate_intent_alignment(question, sql_query)
                logger.info(f"Generated SQL on attempt {attempt + 1}")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                attempts.append({"sql": sql_query, "error": str(e)})
                if attempt == max_retries:
                    raise ValueError(f"Failed after {max_retries + 1} attempts")

        # Cache successful SQL generation (LRU eviction at 200 entries)
        if sql_query:
            if len(self._sql_generation_cache) >= 200:
                oldest_key = next(iter(self._sql_generation_cache))
                del self._sql_generation_cache[oldest_key]
            self._sql_generation_cache[cache_key] = (sql_query, time.time())

        return {"question": question, "sql": sql_query, "attempts": len(attempts) + 1}

    async def query_async(self, question: str, execute: bool = True, max_rows: Optional[int] = None, include_explanation: bool = False) -> Dict[str, Any]:
        """Async version of query - allows concurrent requests."""
        sql_result = await self.generate_sql_async(question)
        response = {"question": question, "sql": sql_result["sql"]}
        if execute:
            # Run blocking DB operation in thread pool
            loop = asyncio.get_event_loop()
            response["execution"] = await loop.run_in_executor(
                None, lambda: self.execute_sql(sql_result["sql"], max_rows)
            )
        return response

    # Direct joins that are known to be wrong (force a retry)
    FORBIDDEN_DIRECT_JOIN_PAIRS = {
        frozenset({"client", "order"}),
        frozenset({"client", "debt"}),
        frozenset({"order", "product"}),
        frozenset({"client", "sale"}),
        # SupplyOrder has NO direct FK to SupplyOrganization or SupplyOrganizationAgreement
        frozenset({"supplyorder", "supplyorganization"}),
        frozenset({"supplyorder", "supplyorganizationagreement"}),
    }

    # Critical business rules (NOT in schema - must be hardcoded)
    CRITICAL_RULES = """
================================================================================
                    CRITICAL JOIN PATTERNS - READ FIRST!
================================================================================

ORDER DOES NOT HAVE ClientID! This is the #1 most common error.

X WRONG (will FAIL):
  SELECT * FROM dbo.[Order] o
  JOIN dbo.Client c ON o.ClientID = c.ID   -- ClientID DOES NOT EXIST!

V CORRECT (always use this):
  SELECT * FROM dbo.[Order] o
  JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
  JOIN dbo.Client c ON ca.ClientID = c.ID

Path: Client -> ClientAgreement -> Order (through ClientAgreementID)

================================================================================
          COLUMNS/TABLES THAT DO NOT EXIST - NEVER USE THESE!
================================================================================

| WRONG (will fail)                        | CORRECT (use this instead)                    |
|------------------------------------------|-----------------------------------------------|
| Order.ClientID                           | Order.ClientAgreementID -> Client             |
| Client.ClientName                        | Client.Name                                   |
| Product.ProductName                      | Product.Name                                  |
| Client.Address                           | Client.LegalAddress or ActualAddress          |
| Client.City                              | Client.LegalAddress LIKE '%City%'             |
| Client.Region                            | Client.RegionID -> Region.Name                |
| Client.Phone                             | Client.MobileNumber                           |
| Client.Email                             | Client.EmailAddress                           |
| ProductBrand table                       | Product.VendorCode LIKE 'BRAND%'              |
| Order.ProductID                          | Order -> OrderItem -> Product                 |
| Sale.ClientID                            | Sale -> Order -> ClientAgreement              |
| SupplyOrder.SupplyOrganizationAgreementID| DOES NOT EXIST! See below                     |
| SupplyPaymentTask.SupplyOrderID          | DOES NOT EXIST!                               |

================================================================================
            CRITICAL: SupplyOrder -> SupplyOrganization PATH DOES NOT EXIST!
================================================================================

SupplyOrder has NO direct FK to SupplyOrganization or SupplyOrganizationAgreement!

SupplyOrder has these columns: OrganizationID (links to Organization, NOT SupplyOrganization!),
ClientID, ClientAgreementID, StorageID, TransportationServiceID.

FOR SUPPLIER ANALYTICS, USE OutcomePaymentOrder INSTEAD:
  SELECT so.Name, SUM(opo.TotalPrice) AS TotalPaid
  FROM dbo.OutcomePaymentOrder opo
  JOIN dbo.SupplyOrganizationAgreement soa ON opo.SupplyOrganizationAgreementID = soa.ID
  JOIN dbo.SupplyOrganization so ON soa.SupplyOrganizationID = so.ID
  WHERE opo.Deleted = 0 AND soa.Deleted = 0 AND so.Deleted = 0
  GROUP BY so.ID, so.Name

================================================================================

COLUMN NAME RULES:
1. ALWAYS use table alias for Deleted: WHERE c.Deleted = 0 AND o.Deleted = 0
2. Brand filtering: WHERE p.VendorCode LIKE N'SEM%' (not ProductBrand table!)
3. Stock/inventory: ProductAvailability.Amount column
4. Use "Name" column for Client/Product names (NOT ClientName/ProductName)
5. Use "LegalAddress" for addresses (NOT Address)
6. Use "MobileNumber" for phones (NOT Phone)
"""

    def _build_prompt(self, question: str, context: str, previous_attempts: List[Dict], selected_tables: Optional[List[str]] = None) -> str:
        examples_section = ""
        if self.example_retriever.is_available():
            try:
                # 1. Semantic search examples (based on question text)
                domain_filter = self._infer_domain_filter(question)
                min_score = 0.35 if self._has_cyrillic(question) else 0.45
                # Use find_similar instead of find_similar_with_correction
                # The correction pipeline has encoding issues with Cyrillic text
                similar_examples = self.example_retriever.find_similar(
                    question,
                    top_k=3,
                    domain_filter=domain_filter,
                    min_score=min_score,
                )

                # 2. Table-based examples (based on selected tables) - NEW
                table_examples = []
                if selected_tables:
                    try:
                        table_examples = self.example_retriever.get_examples_by_tables(selected_tables[:5], limit=2)
                    except Exception as te:
                        logger.debug(f"Table-based retrieval failed: {te}")

                # 3. Combine and deduplicate examples
                seen_ids = set()
                all_examples = []

                # Add semantic examples first (higher priority)
                for ex in similar_examples:
                    if ex.get("id") not in seen_ids:
                        all_examples.append(ex)
                        seen_ids.add(ex.get("id"))

                # Add table-based examples (if not already included)
                for ex in table_examples:
                    if ex.get("id") not in seen_ids and len(all_examples) < 5:
                        all_examples.append(ex)
                        seen_ids.add(ex.get("id"))

                filtered_examples = []
                for ex in all_examples:
                    score = ex.get("similarity_score", 0)
                    if score >= min_score:
                        filtered_examples.append(ex)

                if filtered_examples:
                    examples_section = self.example_retriever.format_examples_for_prompt(filtered_examples, include_ukrainian=True)
                    avg_score = sum(e.get("similarity_score", 0) for e in filtered_examples) / len(filtered_examples)
                    logger.info(f"Retrieved {len(filtered_examples)} examples ({len(similar_examples)} semantic + {len(table_examples)} table-based, avg score: {avg_score:.2f})")
                else:
                    logger.info("No example matches above similarity threshold; skipping examples")
            except Exception as e:
                logger.error(f"Failed to retrieve examples: {e}")

        # Add retry context if previous attempts failed
        retry_context = ""
        if previous_attempts:
            retry_context = "\nPREVIOUS FAILED ATTEMPTS (avoid these errors):\n"
            for attempt in previous_attempts[-2:]:  # Last 2 attempts
                retry_context += f"- SQL: {attempt.get('sql', 'N/A')}\n  Error: {attempt.get('error', 'N/A')}\n"

        # Schema context with limited join guidance for selected tables
        schema_section = context

        # Optional prompt budget guardrail
        def _truncate_to_budget(text: str, budget: int) -> str:
            if budget and len(text) > budget:
                return text[:budget] + "\n-- TRUNCATED TO FIT PROMPT BUDGET --"
            return text

        intent_hints = self._build_intent_hints(question)
        intent_section = f"\nINTENT HINTS:\n{intent_hints}\n" if intent_hints else ""

        join_templates = ""
        if selected_tables:
            join_templates = self._get_join_templates_for_tables(selected_tables[:4])
        join_budget = getattr(settings, "join_rulebook_max_chars", None)
        if join_templates and join_budget:
            join_templates = _truncate_to_budget(join_templates, join_budget)
        join_section = f"\n{join_templates}\n" if join_templates else ""

        # Build prompt body using external template or inline fallback
        if self._sql_prompt_template:
            # Use external template with placeholder substitution
            prompt_body = self._sql_prompt_template.format(
                question=question,
                intent_hints=intent_section.strip() if intent_section else "",
                join_templates=join_section.strip() if join_section else "",
                examples=examples_section.strip() if examples_section else "",
                schema=schema_section,
                retry_context=retry_context.strip() if retry_context else "",
            )
        else:
            # Fallback to inline prompt
            prompt_body = f"""Generate a T-SQL query for this question. Output ONLY the SQL, nothing else.

QUESTION: {question}

RULES:
- Use TOP N (not LIMIT)
- Use dbo.[Order] with brackets
- Add WHERE alias.Deleted = 0
- ALWAYS use short aliases for every table in FROM/JOIN (e.g., FROM dbo.[User] u). Never use SQL keywords (WHERE, AND, OR) as aliases.
- Order has NO ClientID! Use: Order.ClientAgreementID -> ClientAgreement.ClientID -> Client
{intent_section}{join_section}
{examples_section}
SCHEMA:
{schema_section}
{retry_context}
SQL:
"""
        # Apply prompt budget if configured
        prompt_budget = getattr(settings, "prompt_max_chars", None)
        prompt_body = _truncate_to_budget(prompt_body, prompt_budget) if prompt_budget else prompt_body
        return prompt_body

    def _extract_sql(self, llm_response: str) -> str:
        # Try to extract from code blocks first
        patterns = [r"```sql\s*(.*?)```", r"```\s*(.*?)```"]
        for pattern in patterns:
            matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
            if matches:
                return self._fix_sql_dialect(matches[0].strip().strip().rstrip(";").strip().rstrip("`").strip())
        # Try to find SELECT statement
        select_pattern = r"(SELECT\s+.*?)(?=\n\n|$)"
        matches = re.findall(select_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if matches:
            return self._fix_sql_dialect(matches[0].strip().strip().rstrip(";").strip().rstrip("`").strip())
        return self._fix_sql_dialect(llm_response.strip().strip().rstrip(";").strip().rstrip("`").strip())

    # Common column name hallucinations and their corrections
    COLUMN_FIXES = {
        # Bank table hallucinations (LegalAddress doesn't exist, use Address)
        r'\bb\.LegalAddress\b': 'b.Address',
        r'\bBank\.LegalAddress\b': 'Bank.Address',
        # Debt table hallucinations (Amount/IsPaid don't exist, use Total)
        r'\bd\.Amount\b': 'd.Total',
        r'\bDebt\.Amount\b': 'Debt.Total',
        r'\bSUM\(d\.Amount\)': 'SUM(d.Total)',
        r'\bSUM\(Debt\.Amount\)': 'SUM(Debt.Total)',
        # IsPaid doesn't exist - remove entire condition or replace
        r'\s+AND\s+d\.IsPaid\s*=\s*0\b': '',  # Remove "AND d.IsPaid = 0"
        r'\s+AND\s+d\.IsPaid\s*=\s*1\b': '',  # Remove "AND d.IsPaid = 1"
        r'\bd\.IsPaid\s*=\s*0\b': 'd.Total > 0',  # Replace standalone with Total > 0
        r'\bd\.IsPaid\s*=\s*1\b': 'd.Total = 0',  # Replace standalone with Total = 0
        # Client table hallucinations
        r'\bLegalassment\b': 'LegalAddress',
        r'\blegalassment\b': 'LegalAddress',
        r'\bActualizationment\b': 'ActualAddress',
        r'\bactualizationment\b': 'ActualAddress',
        r'\bLegalAdress\b': 'LegalAddress',
        r'\bActualAdress\b': 'ActualAddress',
        r'\bClientName\b': 'Name',
        r'\bclientname\b': 'Name',
        r'\bc\.Address\b(?!\s*LIKE)': 'c.LegalAddress',  # Client.Address -> LegalAddress (not Bank!)
        r'\bPhone\b': 'MobileNumber',  # Phone -> MobileNumber
        r'\bphone\b': 'MobileNumber',
        r'\bTelephone\b': 'MobileNumber',
        r'\bc\.City\b': 'c.LegalAddress',  # Client.City doesn't exist, use LegalAddress (not Bank!)
        r'\bEmail\b': 'EmailAddress',  # Email -> EmailAddress
        r'\bemail\b': 'EmailAddress',
        # Product table hallucinations
        r'\bProductName\b': 'Name',
        r'\bproductname\b': 'Name',
        r'\bArticle\b': 'VendorCode',
        r'\bCode\b(?!\s*=)': 'VendorCode',
        # Product.CostPrice does NOT exist! Replace with Price (best approximation)
        r'\bp\.CostPrice\b': 'p.Price',
        r'\bProduct\.CostPrice\b': 'Product.Price',
        r'\bpr\.CostPrice\b': 'pr.Price',
        r'\bprod\.CostPrice\b': 'prod.Price',
        # Common typos in column names
        r'\bClientAgrementID\b': 'ClientAgreementID',  # Missing 'e' typo
        r'\bClientAgreementId\b': 'ClientAgreementID',  # Case fix
        # Table name typos
        r'\bClientAgrement\b': 'ClientAgreement',  # Missing 'e' typo in table name
    }

    def _fix_sql_dialect(self, sql: str) -> str:
        """Fix PostgreSQL/MySQL syntax to T-SQL and correct hallucinated column names."""
        # First strip whitespace so backtick patterns match reliably
        sql = sql.strip()
        # Clean up markdown backticks that LLM sometimes leaves in SQL
        sql = re.sub(r'^```\s*sql\s*', '', sql, flags=re.IGNORECASE)  # ```sql at start
        sql = re.sub(r'^```\s*', '', sql)  # ``` at start
        sql = re.sub(r'```$', '', sql)  # ``` at end (after strip)
        sql = sql.replace('```', '').strip()  # Any remaining and final cleanup
        # First, convert any Cyrillic region codes to Latin (e.g., 'Хм' -> 'XM')
        sql = self._fix_cyrillic_region_codes(sql)
        # Fix duplicate WHERE keyword (robust to whitespace and dot variants)
        original = sql
        sql = re.sub(r'\bWHERE\s+WHERE\s*\.\s*', 'WHERE ', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bAND\s+WHERE\s*\.\s*', 'AND ', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bWHERE\s+WHERE\s+', 'WHERE ', sql, flags=re.IGNORECASE)
        # Fix "AND WHERE" -> "AND" (LLM sometimes uses WHERE as table alias)
        sql = re.sub(r'\bAND\s+WHERE\s+', 'AND ', sql, flags=re.IGNORECASE)
        if 'WHERE.' in original:
            logger.debug(f"WHERE alias fix: '{original[:100]}' -> '{sql[:100]}'")
        # Convert ILIKE to LIKE (PostgreSQL -> T-SQL)
        sql = re.sub(r"\bILIKE\b", "LIKE", sql, flags=re.IGNORECASE)
        # Convert LIMIT to TOP
        limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", sql, re.IGNORECASE)
        if limit_match:
            sql = re.sub(r"\bLIMIT\s+\d+\s*$", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"^SELECT\b", f"SELECT TOP {limit_match.group(1)}", sql, flags=re.IGNORECASE)
        # Fix trailing TOP N placed after ORDER BY (move to SELECT)
        trailing_top = re.search(r"\bTOP\s+(\d+)\s*$", sql, re.IGNORECASE)
        if trailing_top and not re.search(r"\bSELECT\s+TOP\s+\d+\b", sql, re.IGNORECASE):
            sql = re.sub(r"\bTOP\s+\d+\s*$", "", sql, flags=re.IGNORECASE).rstrip()
            sql = re.sub(r"^SELECT\b", f"SELECT TOP {trailing_top.group(1)}", sql, flags=re.IGNORECASE)
        # Fix Order table (reserved word) - need brackets
        sql = re.sub(r"\bdbo\.Order\b(?!\s*Item)", "dbo.[Order]", sql, flags=re.IGNORECASE)
        # Remove NULLS LAST/FIRST (PostgreSQL syntax, not valid in T-SQL)
        sql = re.sub(r"\s+NULLS\s+(LAST|FIRST)\s*", " ", sql, flags=re.IGNORECASE)
        # Fix SELECT TOP N DISTINCT -> SELECT DISTINCT TOP N (T-SQL requires DISTINCT before TOP)
        sql = re.sub(r"\bSELECT\s+TOP\s+(\d+)\s+DISTINCT\b", r"SELECT DISTINCT TOP \1", sql, flags=re.IGNORECASE)
        # Fix hallucinated column names
        for wrong_pattern, correct_name in self.COLUMN_FIXES.items():
            sql = re.sub(wrong_pattern, correct_name, sql, flags=re.IGNORECASE)
        # Fix ambiguous Deleted column (add table alias if missing)
        sql = self._fix_ambiguous_deleted(sql)
        # Post-fix: Remove any WHERE. prefix that slipped through
        sql = re.sub(r'\bWHERE\s+WHERE\.', 'WHERE ', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bAND\s+WHERE\.', 'AND ', sql, flags=re.IGNORECASE)
        # Fix incorrect ClientAgreement join patterns
        sql = self._fix_client_agreement_joins(sql)
        # Fix incorrect Order.ClientID join patterns (LLM hallucination)
        sql = self._fix_order_client_join(sql)
        # Fix Client.Region column hallucination
        sql = self._fix_region_column(sql)
        # Fix incorrect Debt.ClientID join patterns (LLM hallucination)
        sql = self._fix_debt_client_join(sql)

        # FINAL: Remove any WHERE. alias prefix that may remain
        import sys
        if 'WHERE.' in sql:
            print(f"DEBUG BEFORE FINAL: {sql[:200]}", file=sys.stderr)
        sql = re.sub(r'\bWHERE\.([A-Za-z])', r'\1', sql, flags=re.IGNORECASE)
        if 'WHERE.' in sql:
            print(f"DEBUG AFTER FINAL STILL HAS WHERE.: {sql[:200]}", file=sys.stderr)
        return sql.strip()

    # SQL keywords that should NOT be treated as table aliases
    SQL_KEYWORDS = {
        'where', 'and', 'or', 'on', 'join', 'left', 'right', 'inner', 'outer',
        'full', 'cross', 'order', 'group', 'by', 'having', 'select', 'from',
        'as', 'in', 'not', 'null', 'is', 'like', 'between', 'exists', 'case',
        'when', 'then', 'else', 'end', 'distinct', 'top', 'set', 'update',
        'delete', 'insert', 'into', 'values', 'union', 'all', 'asc', 'desc',
        'limit', 'offset', 'with', 'over', 'partition', 'row', 'rows'
    }

    def _fix_ambiguous_deleted(self, sql: str) -> str:
        """Fix unqualified 'Deleted' column references by inferring table alias.

        Enhanced to handle ALL unqualified Deleted = 0 patterns including:
        - WHERE Deleted = 0
        - AND Deleted = 0
        - ON Deleted = 0
        - Multiple unqualified Deleted in same query
        """
        # Find all table aliases used in the query
        # Handle both [dbo].[Table] and dbo.Table formats
        alias_pattern = r'(?:FROM|JOIN)\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?\s+(?:AS\s+)?([a-zA-Z]\w*)\b'
        all_matches = re.findall(alias_pattern, sql, re.IGNORECASE)

        # Filter out SQL keywords from matched aliases
        aliases = [(table, alias) for table, alias in all_matches
                   if alias.lower() not in self.SQL_KEYWORDS]

        if not aliases:
            # No aliases defined - just return as-is, can't fix without knowing the table
            return sql

        # Get first alias (usually main table)
        first_alias = aliases[0][1]

        # Fix ALL unqualified "Deleted = 0" patterns at once
        # Use word boundary approach - Deleted not preceded by alias.
        # Pattern: whitespace or start, then Deleted = 0
        sql = re.sub(
            r'(?<![.\w])Deleted\s*=\s*0',
            f'{first_alias}.Deleted = 0',
            sql,
            flags=re.IGNORECASE
        )
        return sql

    def _fix_client_agreement_joins(self, sql: str) -> str:
        """Fix incorrect ClientAgreement join patterns that LLM hallucinates.

        Common LLM mistakes:
        - c.ClientAgreementID = ca.ClientAgreementID (wrong!)
        - Should be: c.ID = ca.ClientID (Client to ClientAgreement)

        - ca.ClientAgreementID = o.ClientAgreementID (wrong!)
        - Should be: ca.ID = o.ClientAgreementID (ClientAgreement to Order)
        """
        original = sql

        # Fix: Client.ClientAgreementID = ClientAgreement.ClientAgreementID
        # → Client.ID = ClientAgreement.ClientID
        # Pattern: c.ClientAgreementID = ca.ClientAgreementID (any alias)
        sql = re.sub(
            r'(\w+)\.ClientAgreementID\s*=\s*(\w+)\.ClientAgreementID',
            r'\1.ID = \2.ClientID',
            sql,
            flags=re.IGNORECASE
        )

        # Fix: ClientAgreement ca ON ca.ClientAgreementID = c.ID
        # → ClientAgreement ca ON ca.ClientID = c.ID
        sql = re.sub(
            r'(\w+)\.ClientAgreementID\s*=\s*(\w+)\.ID\b',
            r'\1.ClientID = \2.ID',
            sql,
            flags=re.IGNORECASE
        )

        # Fix: ON c.ID = ca.ClientAgreementID (when joining Client to ClientAgreement)
        # → ON ca.ClientID = c.ID
        sql = re.sub(
            r'ON\s+(\w+)\.ID\s*=\s*(\w+)\.ClientAgreementID\b(?!\s*(?:JOIN|WHERE|AND|OR|GROUP|ORDER))',
            r'ON \2.ClientID = \1.ID',
            sql,
            flags=re.IGNORECASE
        )

        # Fix: JOIN Order o ON o.ID = c.ClientID (completely wrong - Client has ID, not ClientID)
        # This is a common LLM hallucination where it tries to join Order directly to Client
        # The fix inserts ClientAgreement as the intermediary
        sql = re.sub(
            r'JOIN\s+dbo\.\[?Order\]?\s+(\w+)\s+ON\s+\.ID\s*=\s*(\w+)\.ClientID',
            r'JOIN dbo.ClientAgreement ca ON ca.ClientID = .ID JOIN dbo.[Order]  ON .ClientAgreementID = ca.ID',
            sql,
            flags=re.IGNORECASE
        )

        if sql != original:
            logger.debug(f"ClientAgreement join fix applied: {sql[:150]}...")

        return sql

    # Cyrillic to Latin region code mapping
    CYRILLIC_TO_LATIN_REGIONS = {
        'ХМ': 'XM', 'Хм': 'XM', 'хм': 'XM',  # Khmelnytskyi
        'КИ': 'KI', 'Ки': 'KI', 'ки': 'KI', 'КІ': 'KI', 'Кі': 'KI',  # Kyiv
        'ЛВ': 'LV', 'Лв': 'LV', 'лв': 'LV',  # Lviv
        'ОД': 'OD', 'Од': 'OD', 'од': 'OD',  # Odesa
        'ДН': 'DN', 'Дн': 'DN', 'дн': 'DN',  # Dnipro
        'ХН': 'XN', 'Хн': 'XN', 'хн': 'XN',  # Kherson
        'ХВ': 'XV', 'Хв': 'XV', 'хв': 'XV',  # Kharkiv
        'ВІ': 'VI', 'Ві': 'VI', 'ві': 'VI', 'ВИ': 'VI',  # Vinnytsia
        'ВЛ': 'VL', 'Вл': 'VL', 'вл': 'VL',  # Volyn
        'ЗП': 'ZP', 'Зп': 'ZP', 'зп': 'ZP',  # Zaporizhzhia
        'ЗК': 'ZK', 'Зк': 'ZK', 'зк': 'ZK',  # Zakarpattia
        'ІФ': 'IF', 'Іф': 'IF', 'іф': 'IF',  # Ivano-Frankivsk
        'ТЕ': 'TE', 'Те': 'TE', 'те': 'TE',  # Ternopil
        'РІ': 'RI', 'Рі': 'RI', 'рі': 'RI', 'РИ': 'RI',  # Rivne
        'СМ': 'SM', 'См': 'SM', 'см': 'SM',  # Sumy
        'ЧЕ': 'CE', 'Че': 'CE', 'че': 'CE',  # Cherkasy
        'ЧК': 'CK', 'Чк': 'CK', 'чк': 'CK',  # Chernihiv (actually CN)
        'ЧН': 'CN', 'Чн': 'CN', 'чн': 'CN',  # Chernihiv
        'ПА': 'PA', 'Па': 'PA', 'па': 'PA',  # Poltava
        'МІ': 'MI', 'Мі': 'MI', 'мі': 'MI', 'МИ': 'MI',  # Mykolaiv
        'КД': 'KD', 'Кд': 'KD', 'кд': 'KD',  # Kirovohrad
        'ДП': 'DP', 'Дп': 'DP', 'дп': 'DP',  # Dnipropetrovsk
        'ЛК': 'LK', 'Лк': 'LK', 'лк': 'LK',  # Luhansk
        'ГТ': 'GT', 'Гт': 'GT', 'гт': 'GT',  # Zhytomyr (GT code)
    }

    def _fix_cyrillic_region_codes(self, sql: str) -> str:
        """Convert Cyrillic region codes to Latin in SQL."""
        for cyrillic, latin in self.CYRILLIC_TO_LATIN_REGIONS.items():
            # Replace in string literals: 'ХМ' -> 'XM', N'ХМ' -> N'XM'
            sql = sql.replace(f"'{cyrillic}'", f"'{latin}'")
            sql = sql.replace(f"'{cyrillic}%'", f"'{latin}%'")
            sql = sql.replace(f"'%{cyrillic}'", f"'%{latin}'")
            sql = sql.replace(f"'%{cyrillic}%'", f"'%{latin}%'")
        return sql

    def _fix_transliterated_names(self, sql: str, question: str) -> str:
        """Restore Cyrillic names from original question that LLM may have transliterated.

        Example: Question has 'Луньов Микола' but LLM generates LIKE '%Lunov Mikola%'
        This method extracts the Cyrillic name from question and replaces the Latin version.
        """
        # Extract proper names and company names from question
        # Pattern 1: Capitalized words like "Луньов Микола"
        name_pattern = r"[А-ЯІЇЄҐ][а-яіїєґ']+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ']+)+"
        potential_names = re.findall(name_pattern, question)
        
        # Pattern 2: All-caps company names like "УКР АГРО СПЕЦ ТРЕЙД ТОВ"
        company_pattern = r"[А-ЯІЇЄҐ]{2,}(?:\s+[А-ЯІЇЄҐ]{2,})+"
        for match in re.finditer(company_pattern, question):
            name = match.group(0)
            if name not in potential_names and len(name) >= 5:
                potential_names.append(name)
        
        # Pattern 3: Single capitalized words with 4+ chars
        single_name_pattern = r"([А-ЯІЇЄҐ][а-яіїєґ']{3,})"
        for match in re.finditer(single_name_pattern, question):
            name = match.group(1)
            if name not in potential_names:
                potential_names.append(name)

        if not potential_names:
            return sql

        # Map first Cyrillic letter to expected Latin transliteration
        cyrillic_to_latin_first = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
            'є': 'y', 'ж': 'z', 'з': 'z', 'и': 'y', 'і': 'i', 'ї': 'y',
            'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
            'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
            'х': 'k', 'ц': 't', 'ч': 'c', 'ш': 's', 'щ': 's', 'ь': '',
            'ю': 'y', 'я': 'y',
        }

        # Find Latin strings in LIKE clauses
        like_pattern = r"LIKE\s+N?'%?([^']+)%?'"
        for match in re.finditer(like_pattern, sql, re.IGNORECASE):
            latin_value = match.group(1)

            # Skip if it's already Cyrillic or too short
            if re.search(r'[А-Яа-яІЇЄҐіїєґ]', latin_value) or len(latin_value) < 3:
                continue

            # Try to match with potential names
            for cyrillic_name in potential_names:
                latin_lower = latin_value.lower().replace(' ', '')
                cyrillic_words = cyrillic_name.lower().split()

                # Check if first letters match (quick heuristic)
                if cyrillic_words and latin_lower:
                    first_cyrillic = cyrillic_words[0][0] if cyrillic_words[0] else ''
                    first_latin = latin_lower[0] if latin_lower else ''

                    expected_first = cyrillic_to_latin_first.get(first_cyrillic, first_cyrillic)
                    if expected_first == first_latin:
                        # Likely match - replace Latin with Cyrillic
                        old_like = match.group(0)
                        new_like = old_like.replace(latin_value, cyrillic_name)
                        sql = sql.replace(old_like, new_like)
                        logger.debug(f"Restored Cyrillic name: '{latin_value}' -> '{cyrillic_name}'")
                        break

        # Ensure Cyrillic strings have N prefix and proper wildcards
        sql = self._fix_cyrillic_string_format(sql)
        return sql

    def _fix_cyrillic_string_format(self, sql: str) -> str:
        """Ensure Cyrillic strings have N prefix and proper wildcards for name searches."""
        # Pattern to find strings with Cyrillic (with or without N prefix)
        # Match LIKE N'value' or LIKE 'value' patterns - catches any string with Cyrillic chars
        pattern = r"LIKE\s+N?'(%?)([^']*[А-Яа-яІЇЄҐіїєґ][^']*)(%?)'"
        
        def add_n_prefix_and_wildcards(match):
            prefix_wild = match.group(1)
            value = match.group(2)
            suffix_wild = match.group(3)
            # Ensure wildcards on both sides for name searches
            if not prefix_wild:
                prefix_wild = '%'
            if not suffix_wild:
                suffix_wild = '%'
            return f"LIKE N'{prefix_wild}{value}{suffix_wild}'"
        
        sql = re.sub(pattern, add_n_prefix_and_wildcards, sql, flags=re.IGNORECASE)
        
        # Also fix strings with high-byte characters (partially transliterated names)
        # Pattern for any non-ASCII in LIKE clause
        pattern2 = r"LIKE\s+'(%?)([^']*[-ÿ][^']*)(%?)'"
        sql = re.sub(pattern2, add_n_prefix_and_wildcards, sql, flags=re.IGNORECASE)
        
        return sql

    def _fix_region_column(self, sql: str) -> str:
        """Fix Client.Region column hallucination.

        Client table doesn't have a Region column, it has RegionID (FK to Region table).
        Pattern: c.Region LIKE N'XM%' (or = 'XM')
        Fix to: c.RegionID IN (SELECT ID FROM dbo.Region WHERE Name LIKE N'XM%')
        """
        # First convert any Cyrillic region codes to Latin
        sql = self._fix_cyrillic_region_codes(sql)
        original = sql

        # Pattern: alias.Region LIKE N'value%'
        sql = re.sub(
            r'(\w+)\.Region\s+LIKE\s+(N?\'[^\']+\')',
            r'\1.RegionID IN (SELECT ID FROM dbo.Region WHERE Name LIKE \2)',
            sql,
            flags=re.IGNORECASE
        )

        # Pattern: alias.Region = N'value' or = 'value'
        sql = re.sub(
            r'(\w+)\.Region\s*=\s*(N?\'[^\']+\')',
            r'\1.RegionID IN (SELECT ID FROM dbo.Region WHERE Name = \2)',
            sql,
            flags=re.IGNORECASE
        )

        # Pattern: unqualified Region = 'value' (no alias) - for simple queries like SELECT COUNT(*) FROM Client
        sql = re.sub(
            r'\bWHERE\s+Region\s*=\s*(N?\'[^\']+\')',
            r'WHERE RegionID IN (SELECT ID FROM dbo.Region WHERE Name = \1)',
            sql,
            flags=re.IGNORECASE
        )

        # Pattern: unqualified Region LIKE 'value%' (no alias)
        sql = re.sub(
            r'\bWHERE\s+Region\s+LIKE\s+(N?\'[^\']+\')',
            r'WHERE RegionID IN (SELECT ID FROM dbo.Region WHERE Name LIKE \1)',
            sql,
            flags=re.IGNORECASE
        )

        if sql != original:
            logger.info(f"Region column fix applied: {sql[:200]}...")

        return sql

    # Known brand prefixes in VendorCode - only filter when user explicitly asks for these
    KNOWN_BRAND_PREFIXES = ['SEM', 'MG', 'ALKO', 'BKT', 'STARCO', 'VREDESTEIN', 'TRELLEBORG', 'ATF']

    def _fix_unwanted_brand_filters(self, sql: str, question: str) -> str:
        """Remove unwanted VendorCode LIKE brand filters from general 'top products' queries.

        The LLM sometimes adds brand filters (e.g., VendorCode LIKE N'SEM%' OR VendorCode LIKE N'MG%')
        even when the user asks a general question like "топ 10 товарів по продажах" without
        mentioning any specific brand.

        This function removes such filters when:
        1. The question contains general product ranking keywords
        2. The question does NOT contain any specific brand names
        """
        q = question.lower()

        # Check if this is a general product ranking query
        general_product_keywords = [
            'топ товар', 'топ продаж', 'топ 10', 'топ 5', 'топ 20',
            'рейтинг товар', 'рейтинг продаж', 'найпопулярніш',
            'найкращ товар', 'найбільш продаван',
            'top product', 'top selling', 'top 10', 'best selling',
            'product ranking', 'most popular product',
        ]

        is_general_product_query = any(kw in q for kw in general_product_keywords)

        if not is_general_product_query:
            return sql

        # Check if user explicitly mentions a brand name
        brand_mentions = [prefix.lower() for prefix in self.KNOWN_BRAND_PREFIXES]
        brand_mentions.extend(['семперіт', 'семперит', 'semperit', 'алко', 'бкт'])

        user_mentions_brand = any(brand in q for brand in brand_mentions)

        if user_mentions_brand:
            # User explicitly asked for a brand - keep the filter
            return sql

        # Remove VendorCode LIKE filters from general queries
        original = sql

        # Pattern: AND (p.VendorCode LIKE N'SEM%' OR p.VendorCode LIKE N'MG%')
        # or: AND p.VendorCode LIKE N'SEM%'
        # Match the whole condition including surrounding AND
        patterns_to_remove = [
            # Pattern with parentheses and OR
            r"\s*AND\s*\(\s*\w+\.VendorCode\s+LIKE\s+N?'[^']+%?'(?:\s+OR\s+\w+\.VendorCode\s+LIKE\s+N?'[^']+%?')+\s*\)",
            # Single VendorCode LIKE condition after AND
            r"\s*AND\s+\w+\.VendorCode\s+LIKE\s+N?'[^']+%?'",
            # Pattern where VendorCode filter is in WHERE directly (not after AND)
            r"\s*WHERE\s+\w+\.VendorCode\s+LIKE\s+N?'[^']+%?'\s+AND\s+",
        ]

        for pattern in patterns_to_remove:
            # For WHERE ... AND pattern, replace with just WHERE
            if 'WHERE' in pattern:
                sql = re.sub(pattern, ' WHERE ', sql, flags=re.IGNORECASE)
            else:
                sql = re.sub(pattern, '', sql, flags=re.IGNORECASE)

        if sql != original:
            logger.info(f"Removed unwanted brand filter from general product query: {question[:50]}...")

        return sql

    def _enhance_top_products_sql(self, sql: str, question: str) -> str:
        """Add missing columns to top products queries.
        
        When user asks for 'top products by sales', ensure the SQL includes:
        - VendorCode
        - OriginalVendorCode  
        - TotalRevenue (SUM(Qty * PricePerItem))
        - OrderCount (COUNT(DISTINCT OrderID))
        """
        q = question.lower()
        
        # Check if this is a top products query
        top_product_keywords = [
            'топ товар', 'топ 10', 'топ 5', 'топ 20', 'топ продаж',
            'рейтинг товар', 'найкращ товар', 'найпопулярніш',
            'top product', 'top 10', 'top selling', 'best selling',
        ]
        
        is_top_product_query = any(kw in q for kw in top_product_keywords)
        
        if not is_top_product_query:
            return sql
            
        # Check if SQL is missing key columns
        sql_upper = sql.upper()
        
        has_vendorcode = 'VENDORCODE' in sql_upper and 'AS' in sql_upper  # Selected, not just in WHERE
        has_originalvendorcode = 'ORIGINALVENDORCODE' in sql_upper
        has_revenue = 'TOTALREVENUE' in sql_upper or ('QTY' in sql_upper and 'PRICEPERITEM' in sql_upper)
        has_ordercount = 'ORDERCOUNT' in sql_upper or 'COUNT(DISTINCT' in sql_upper
        
        if has_vendorcode and has_originalvendorcode and has_revenue and has_ordercount:
            # Already has all columns
            return sql
            
        # Replace simple TOP 10 product queries with enhanced version
        # Pattern: SELECT TOP N p.Name, SUM(oi.Qty) as TotalSold FROM Product p JOIN OrderItem...
        simple_pattern = r"SELECT\s+TOP\s+\d+\s+p\.Name\s*(?:AS\s+\w+)?\s*,\s*SUM\(oi\.Qty\)\s*(?:AS\s+\w+)?"
        
        if re.search(simple_pattern, sql, re.IGNORECASE):
            # Replace with enhanced SELECT
            enhanced_select = "SELECT TOP 10 p.Name as ProductName, p.VendorCode, SUM(oi.Qty) as TotalSold, COUNT(DISTINCT oi.OrderID) as OrderCount, SUM(oi.Qty * oi.PricePerItem) as TotalRevenue"
            sql = re.sub(simple_pattern, enhanced_select, sql, flags=re.IGNORECASE)
            
            # Also need to update GROUP BY to include new columns
            if 'GROUP BY' in sql.upper():
                # Replace GROUP BY p.ID, p.Name with GROUP BY p.ID, p.Name, p.VendorCode, p.OriginalVendorCode
                old_group = r"GROUP BY\s+p\.ID\s*,\s*p\.Name(?!\s*,\s*p\.VendorCode)"
                new_group = "GROUP BY p.ID, p.Name, p.VendorCode"
                sql = re.sub(old_group, new_group, sql, flags=re.IGNORECASE)
            
            # Also fix ORDER BY to use TotalSold
            sql = re.sub(r'ORDER BY\s+\w+\s+DESC', 'ORDER BY TotalSold DESC', sql, flags=re.IGNORECASE)
            logger.info(f"Enhanced top products SQL with additional columns")
        
        return sql


    def _fix_order_client_join(self, sql: str) -> str:
        """Fix incorrect Order-Client join patterns.

        LLM often hallucinates direct join between Client and Order.
        This fixes it to use the correct path: Client -> ClientAgreement -> Order

        Patterns fixed:
        1. JOIN [Order] o ON o.ClientID = c.ID  (ClientID doesn't exist on Order)
        2. JOIN [Order] o ON c.ID = o.ClientAgreementID (skips ClientAgreement table)
        3. JOIN [Order] o ON o.ClientID = ca.ID  (wrong column when ClientAgreement exists)
        """
        original = sql

        # Pattern 3: Order.ClientID = ClientAgreement.ID (wrong column - should be ClientAgreementID)
        # This happens when ClientAgreement is in query but Order uses wrong column
        # Fix: o.ClientID = ca.ID  ->  o.ClientAgreementID = ca.ID
        if re.search(r'\bClientAgreement\b', sql, re.IGNORECASE):
            sql = re.sub(
                r'\b(\w+)\.ClientID\s*=\s*(\w+)\.ID\b',
                lambda m: f'{m.group(1)}.ClientAgreementID = {m.group(2)}.ID'
                if m.group(1).lower() in ('o', 'ord', 'order') and m.group(2).lower() in ('ca', 'clientagreement')
                else m.group(0),
                sql,
                flags=re.IGNORECASE
            )

        # Pattern 1: JOIN Order o ON o.ClientID = c.ID (ClientID doesn't exist!)
        if re.search(r'\b(?:o|Order)\.ClientID\b', sql, re.IGNORECASE):
            sql = re.sub(
                r'(FROM\s+(?:dbo\.)?\[?Client\]?\s+(\w+))\s+'
                r'(?:LEFT\s+|RIGHT\s+|INNER\s+)?JOIN\s+(?:dbo\.)?\[?Order\]?\s+(\w+)\s+'
                r'ON\s+(?:\w+\.ClientID\s*=\s*\w+\.ID|\w+\.ID\s*=\s*\w+\.ClientID)',
                r'\1 '
                r'JOIN dbo.ClientAgreement ca ON ca.ClientID = \2.ID '
                r'JOIN dbo.[Order] \3 ON \3.ClientAgreementID = ca.ID',
                sql,
                flags=re.IGNORECASE
            )

        # Pattern 2: JOIN Order o ON c.ID = o.ClientAgreementID (skips ClientAgreement!)
        # This is wrong because Client.ID != ClientAgreement.ID
        # Should be: Client.ID = ClientAgreement.ClientID AND ClientAgreement.ID = Order.ClientAgreementID
        if re.search(r'FROM\s+(?:dbo\.)?\[?Client\]?\s+\w+\s+(?:LEFT\s+|RIGHT\s+|INNER\s+)?JOIN\s+(?:dbo\.)?\[?Order\]?\s+\w+\s+ON\s+(?:\w+\.ID\s*=\s*\w+\.ClientAgreementID|\w+\.ClientAgreementID\s*=\s*\w+\.ID)', sql, re.IGNORECASE):
            # Check if ClientAgreement is NOT already in the query
            if not re.search(r'\bClientAgreement\b', sql, re.IGNORECASE):
                sql = re.sub(
                    r'(FROM\s+(?:dbo\.)?\[?Client\]?\s+(\w+))\s+'
                    r'(?:LEFT\s+|RIGHT\s+|INNER\s+)?JOIN\s+(?:dbo\.)?\[?Order\]?\s+(\w+)\s+'
                    r'ON\s+(?:\w+\.ID\s*=\s*\w+\.ClientAgreementID|\w+\.ClientAgreementID\s*=\s*\w+\.ID)',
                    r'\1 '
                    r'JOIN dbo.ClientAgreement ca ON ca.ClientID = \2.ID '
                    r'JOIN dbo.[Order] \3 ON \3.ClientAgreementID = ca.ID',
                    sql,
                    flags=re.IGNORECASE
                )
                if sql != original:
                    logger.info(f"Order.ClientAgreementID direct join fix applied: {sql[:200]}...")
                    return sql

        if sql != original:
            logger.info(f"Order.ClientID fix applied: {sql[:200]}...")

        return sql

    def _fix_debt_client_join(self, sql: str) -> str:
        """Fix incorrect Debt-Client join patterns.

        LLM often hallucinates direct join between Client and Debt using d.ClientID.
        This fixes it to use the correct path: Client -> ClientInDebt -> Debt

        Patterns fixed:
        1. JOIN Debt d ON d.ClientID = c.ID  (ClientID doesn't exist on Debt)
        2. JOIN Debt d ON c.ID = d.ClientID  (reverse order of above)
        """
        original = sql

        # Check if Debt table is referenced and has incorrect ClientID join
        if re.search(r'\bDebt\b', sql, re.IGNORECASE) and re.search(r'\b(?:d|Debt)\.ClientID\b', sql, re.IGNORECASE):
            # Pattern: FROM Client c ... JOIN Debt d ON d.ClientID = c.ID
            # or: FROM Client c ... JOIN Debt d ON c.ID = d.ClientID
            sql = re.sub(
                r'(FROM\s+(?:dbo\.)?\[?Client\]?\s+(\w+))\s+'
                r'(?:LEFT\s+|RIGHT\s+|INNER\s+)?JOIN\s+(?:dbo\.)?\[?Debt\]?\s+(\w+)\s+'
                r'ON\s+(?:\w+\.ClientID\s*=\s*\w+\.ID|\w+\.ID\s*=\s*\w+\.ClientID)',
                r'\1 '
                r'JOIN dbo.ClientInDebt cid ON cid.ClientID = \2.ID '
                r'JOIN dbo.Debt \3 ON cid.DebtID = \3.ID',
                sql,
                flags=re.IGNORECASE
            )

            # Also add cid.Deleted = 0 condition if there's a WHERE clause with Deleted
            if sql != original and 'WHERE' in sql.upper():
                # Add cid.Deleted = 0 to WHERE conditions if not already present
                if not re.search(r'\bcid\.Deleted\b', sql, re.IGNORECASE):
                    sql = re.sub(
                        r'\bWHERE\s+',
                        r'WHERE cid.Deleted = 0 AND ',
                        sql,
                        count=1,
                        flags=re.IGNORECASE
                    )

        if sql != original:
            logger.info(f"Debt.ClientID fix applied: {sql[:200]}...")

        return sql

    # Common schema hallucinations to detect and reject (triggers retry)
    # Error messages include CONCRETE SQL examples to help LLM fix on retry
    SCHEMA_ERRORS = [
        # Removed Order.ClientID - now fixed in _fix_order_client_join
        (r'\bdbo\.Brand\b', "Brand table does not exist! Use Product.VendorCode for brands"),
        (r'\bFROM\s+Brand\b', "Brand table does not exist! Use Product.VendorCode for brands"),
        (r'\bJOIN\s+(?:dbo\.)?Brand\b', "Brand table does not exist! Use Product.VendorCode for brands"),
        (r'\bProductBrand\b', "ProductBrand table does not exist! Use Product.VendorCode"),
        (r'\bClientAddress\b', "ClientAddress does not exist! Use Client.LegalAddress or Client.ActualAddress"),
        (r'\bOrderClient\b', "OrderClient does not exist! Use Client->ClientAgreement->Order"),
        # Product.CostPrice does not exist!
        (r'\bProduct\b[^;]*\.CostPrice\b',
         "Product has NO CostPrice column! For margin analysis, use only revenue: SELECT c.Name, SUM(oi.Qty * oi.PricePerItem) AS Revenue FROM Category c JOIN Product p ON p.CategoryID = c.ID JOIN OrderItem oi ON oi.ProductID = p.ID GROUP BY c.Name"),
        (r'\bp\.CostPrice\b',
         "Product has NO CostPrice column! For margin analysis, use only revenue: SELECT c.Name, SUM(oi.Qty * oi.PricePerItem) AS Revenue FROM Category c JOIN Product p ON p.CategoryID = c.ID JOIN OrderItem oi ON oi.ProductID = p.ID GROUP BY c.Name"),
        # Product.Price does not exist! (LLM hallucinates this)
        (r'\bp\.Price\b',
         "Product has NO Price column! Use ProductPricing.Price for cost or OrderItem.PricePerItem for sales: SELECT p.Name, pp.Price AS CostPrice, oi.PricePerItem AS SalePrice FROM Product p JOIN ProductPricing pp ON pp.ProductID = p.ID JOIN OrderItem oi ON oi.ProductID = p.ID"),
        (r'\bProduct\b[^;]*\.Price\b(?!\s*(?:Per|perItem))',
         "Product has NO Price column! Use ProductPricing.Price for cost or OrderItem.PricePerItem for sales prices"),
        # Debt.Days does not exist! (LLM hallucinates aging field)
        (r'\bDebt\b[^;]*\.Days\b',
         "Debt has NO Days column! For aging, calculate: DATEDIFF(day, d.Created, GETDATE()) AS DaysOverdue. Example: SELECT c.Name, DATEDIFF(day, d.Created, GETDATE()) AS DaysOverdue FROM Debt d JOIN ClientInDebt cid ON cid.DebtID = d.ID JOIN Client c ON c.ID = cid.ClientID"),
        (r'\bd\.Days\b',
         "Debt has NO Days column! Calculate aging: DATEDIFF(day, d.Created, GETDATE()) AS DaysOverdue"),
        # SupplyOrder schema hallucinations - CRITICAL: these columns DO NOT EXIST
        # Handle all formats: SupplyOrder, [SupplyOrder], [dbo].[SupplyOrder], dbo.SupplyOrder
        (r'(?:\[?dbo\]?\.)?\[?SupplyOrder\]?\s+\w+\s+ON\s+\w+\.SupplyOrganizationAgreementID\b',
         "SupplyOrder has NO SupplyOrganizationAgreementID! For suppliers, use: SELECT TOP 10 so.Name, SUM(opo.TotalPrice) AS Total FROM OutcomePaymentOrder opo JOIN SupplyOrganizationAgreement soa ON opo.SupplyOrganizationAgreementID = soa.ID JOIN SupplyOrganization so ON soa.SupplyOrganizationID = so.ID GROUP BY so.Name ORDER BY Total DESC"),
        (r'\bSupplyPaymentTask\b[^;]*\bSupplyOrderID\b',
         "SupplyPaymentTask has NO SupplyOrderID column! These tables are not directly linked"),
        (r'\w+\.SupplyOrganizationAgreementID\s*=.*(?:\[?dbo\]?\.)?\[?SupplyOrder',
         "SupplyOrder has NO SupplyOrganizationAgreementID! For suppliers, use: SELECT TOP 10 so.Name, SUM(opo.TotalPrice) AS Total FROM OutcomePaymentOrder opo JOIN SupplyOrganizationAgreement soa ON opo.SupplyOrganizationAgreementID = soa.ID JOIN SupplyOrganization so ON soa.SupplyOrganizationID = so.ID GROUP BY so.Name ORDER BY Total DESC"),
        (r'\bSupplyOrder\b[^;]*SupplyOrganizationID\b',
         "SupplyOrder has NO SupplyOrganizationID column! Supplier spend must use OutcomePaymentOrder -> SupplyOrganizationAgreement -> SupplyOrganization"),
    ]

    def _validate_sql(self, sql_query: str) -> None:
        if not sql_query:
            raise ValueError("Empty SQL")
        for pattern in [r";\s*DROP", r";\s*DELETE"]:
            if re.search(pattern, sql_query, re.IGNORECASE):
                raise ValueError("Dangerous SQL pattern")
        # Check for common schema hallucinations
        for pattern, error_msg in self.SCHEMA_ERRORS:
            if re.search(pattern, sql_query, re.IGNORECASE):
                raise ValueError(f"Schema error: {error_msg}")
        # Validate columns and joins against schema/rulebook
        errors, warnings = self._validate_joins_against_schema(sql_query)
        if warnings:
            logger.info(f"Join validation warnings: {warnings[:3]}{' ...' if len(warnings)>3 else ''}")
        if errors:
            raise ValueError(" ; ".join(errors))
        text(sql_query)

    def _check_read_only(self, sql_query: str) -> None:
        for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]:
            if re.search(rf"\b{kw}\b", sql_query.upper()):
                raise ValueError(f"Write operation not allowed: {kw}")

    def _validate_joins_against_schema(self, sql_query: str) -> tuple[list, list]:
        """Lightweight validation of aliases/columns and join pairs against schema.

        Returns:
            (errors, warnings)
        """
        errors = []
        warnings = []
        if not self._table_columns:
            return errors, warnings

        # Detect common CTE names so we do not flag them as missing tables
        cte_names = set()
        for match in re.finditer(r'\bWITH\s+([A-Za-z_]\w*)\s+AS\b', sql_query, re.IGNORECASE):
            cte_names.add(match.group(1))
        for match in re.finditer(r'\b,\s*([A-Za-z_]\w*)\s+AS\b', sql_query, re.IGNORECASE):
            cte_names.add(match.group(1))

        # Build alias -> table map
        alias_pattern = re.compile(r'\b(?:FROM|JOIN)\s+(?:dbo\.)?\[?(\w+)\]?\s+(?:AS\s+)?(\w+)?', re.IGNORECASE)
        aliases = {}
        for match in alias_pattern.finditer(sql_query):
            table = normalize_table_name(match.group(1))
            alias = match.group(2) or table
            aliases[alias] = table

        if not aliases:
            return errors, warnings

        # Validate table existence (skip CTEs)
        for alias, table in aliases.items():
            if table not in self._table_columns and table not in cte_names:
                errors.append(f"Unknown table {table}")

        # Find column references alias.col
        col_ref_pattern = re.compile(r'\b([A-Za-z_]\w*)\.\[?([A-Za-z_]\w*)\]?\b')
        for a, col in col_ref_pattern.findall(sql_query):
            if a not in aliases:
                continue
            table = aliases[a]
            col_clean = col.replace("[", "").replace("]", "")
            if table in self._table_columns and col_clean not in self._table_columns[table]:
                errors.append(f"Column {a}.{col_clean} not found in table {table}")

        # Validate join pairs
        join_eq_pattern = re.compile(r'([A-Za-z_]\w*)\.\[?([A-Za-z_]\w*)\]?\s*=\s*([A-Za-z_]\w*)\.\[?([A-Za-z_]\w*)\]?', re.IGNORECASE)
        for a1, c1, a2, c2 in join_eq_pattern.findall(sql_query):
            if a1 not in aliases or a2 not in aliases:
                continue
            t1 = aliases[a1]
            t2 = aliases[a2]
            c1_clean = c1.replace("[", "").replace("]", "")
            c2_clean = c2.replace("[", "").replace("]", "")
            # Column existence (already checked above, but keep to attach context)
            if t1 in self._table_columns and c1_clean not in self._table_columns[t1]:
                errors.append(f"Join uses missing column {a1}.{c1_clean} (table {t1})")
                continue
            if t2 in self._table_columns and c2_clean not in self._table_columns[t2]:
                errors.append(f"Join uses missing column {a2}.{c2_clean} (table {t2})")
                continue
            # FK consistency check (escalate only for known bad direct joins)
            if (t1, c1_clean, t2, c2_clean) not in self._fk_pairs:
                message = f"Join {a1}.{c1_clean} = {a2}.{c2_clean} not in FK map ({t1}<->{t2})"
                pair_key = frozenset({t1.lower(), t2.lower()})
                if pair_key in self.FORBIDDEN_DIRECT_JOIN_PAIRS:
                    errors.append(f"{message} (direct join forbidden)")
                else:
                    warnings.append(message)

        return errors, warnings

