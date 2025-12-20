"""SQL generation agent using local LLM via Ollama."""
import re
import asyncio
import hashlib
import time
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

    # LLM generation parameters for faster, more deterministic SQL
    LLM_OPTIONS = {
        "temperature": 0.1,    # Low for deterministic SQL generation
        "num_predict": 500,    # Cap response tokens (SQL rarely >500)
        "top_k": 10,           # Focused decoding
        "top_p": 0.9,          # Nucleus sampling
    }

    def __init__(
        self,
        engine: Optional[Engine] = None,
        schema_extractor: Optional[SchemaExtractor] = None,
        table_selector: Optional[TableSelector] = None,
        query_example_retriever: Optional[QueryExampleRetriever] = None,
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

        # Sync and async Ollama clients
        self.ollama_client = ollama.Client(host=settings.ollama_base_url)
        self.ollama_async_client = ollama.AsyncClient(host=settings.ollama_base_url)

        # Initialize JoinPathGraph for dynamic join path computation
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
            self.precomputed_join_templates = self._precompute_join_templates(
                schema, max_pairs=settings.precompute_max_pairs
            )
        except Exception as e:
            logger.error(f"Failed to initialize JoinPathGraph: {e}")
            self.join_graph = None  # Graceful degradation
            self.precomputed_join_templates = {}

        # Cache for dynamic relationships (generated from schema FKs)
        self._relationships_cache: Optional[str] = None
        self._join_rulebook_cache: Optional[str] = None

    def _canonical_pair_key(self, t1: str, t2: str) -> str:
        """Stable key for unordered table pair."""
        return "||".join(sorted([normalize_table_name(t1), normalize_table_name(t2)]))

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

    def generate_sql(self, question: str, max_retries: int = 2, include_explanation: bool = False) -> Dict[str, Any]:
        logger.info(f"Generating SQL for: {question}")

        # Check SQL generation cache first (saves 1-5 seconds per hit)
        cache_key = hashlib.md5(question.strip().lower().encode()).hexdigest()
        if cache_key in self._sql_generation_cache:
            cached_sql, cached_time = self._sql_generation_cache[cache_key]
            if time.time() - cached_time < self.SQL_GENERATION_CACHE_TTL:
                logger.info(f"SQL generation cache hit for: {question[:50]}...")
                return {"question": question, "sql": cached_sql, "attempts": 0, "cached": True}

        # Use full schema context (all 300+ tables in compact form + detailed relevant tables)
        context = self.table_selector.get_full_schema_context(question)
        selected_tables = self.table_selector.get_selected_tables(question)
        sql_query = None
        attempts = []

        for attempt in range(max_retries + 1):
            try:
                prompt = self._build_prompt(question, context, attempts, selected_tables)
                response = self.ollama_client.generate(
                    model=settings.ollama_model,
                    prompt=prompt,
                    options=self.LLM_OPTIONS
                )
                sql_query = self._extract_sql(response["response"])
                self._validate_sql(sql_query)
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

    async def generate_sql_async(self, question: str, max_retries: int = 2, include_explanation: bool = False) -> Dict[str, Any]:
        """Async version of generate_sql - allows concurrent requests."""
        logger.info(f"Generating SQL (async) for: {question}")

        # Check SQL generation cache first (saves 1-5 seconds per hit)
        cache_key = hashlib.md5(question.strip().lower().encode()).hexdigest()
        if cache_key in self._sql_generation_cache:
            cached_sql, cached_time = self._sql_generation_cache[cache_key]
            if time.time() - cached_time < self.SQL_GENERATION_CACHE_TTL:
                logger.info(f"SQL generation cache hit (async) for: {question[:50]}...")
                return {"question": question, "sql": cached_sql, "attempts": 0, "cached": True}

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
                    model=settings.ollama_model,
                    prompt=prompt,
                    options=self.LLM_OPTIONS
                )
                sql_query = self._extract_sql(response["response"])
                self._validate_sql(sql_query)
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

| WRONG (will fail)        | CORRECT (use this instead)           |
|--------------------------|--------------------------------------|
| Order.ClientID           | Order.ClientAgreementID -> Client    |
| Client.ClientName        | Client.Name                          |
| Product.ProductName      | Product.Name                         |
| Client.Address           | Client.LegalAddress or ActualAddress |
| Client.City              | Client.LegalAddress LIKE '%City%'    |
| Client.Region            | Client.RegionID -> Region.Name       |
| Client.Phone             | Client.MobileNumber                  |
| Client.Email             | Client.EmailAddress                  |
| ProductBrand table       | Product.VendorCode LIKE 'BRAND%'     |
| Order.ProductID          | Order -> OrderItem -> Product        |
| Sale.ClientID            | Sale -> Order -> ClientAgreement     |

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
                similar_examples = self.example_retriever.find_similar_with_correction(question, top_k=3)

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

                if all_examples:
                    examples_section = self.example_retriever.format_examples_for_prompt(all_examples, include_ukrainian=True)
                    avg_score = sum(e.get("similarity_score", 0) for e in all_examples) / len(all_examples)
                    logger.info(f"Retrieved {len(all_examples)} examples ({len(similar_examples)} semantic + {len(table_examples)} table-based, avg score: {avg_score:.2f})")
            except Exception as e:
                logger.error(f"Failed to retrieve examples: {e}")

        # Add retry context if previous attempts failed
        retry_context = ""
        if previous_attempts:
            retry_context = "\nPREVIOUS FAILED ATTEMPTS (avoid these errors):\n"
            for attempt in previous_attempts[-2:]:  # Last 2 attempts
                retry_context += f"- SQL: {attempt.get('sql', 'N/A')}\n  Error: {attempt.get('error', 'N/A')}\n"

        # Get dynamic JOIN templates based on selected tables (NEW - replaces flat FK list)
        join_templates = ""
        if selected_tables and self.join_graph:
            try:
                join_templates = self._get_join_templates_for_tables(selected_tables)
                if not join_templates:
                    logger.debug(f"No join templates generated for tables: {selected_tables[:5]}")
            except Exception as e:
                logger.error(f"Failed to generate join templates: {e}")
                join_templates = ""

        # Precomputed join templates for selected pairs (fast path, capped)
        precomputed_templates = ""
        if selected_tables and self.precomputed_join_templates:
            # use per-query cap from settings to preserve prompt space
            precomputed_templates = self._get_precomputed_join_templates_for_tables(
                selected_tables,
                limit=settings.precomputed_per_query_limit
            )

        # Full join rulebook (FK edges with aliases) for redundancy
        join_rulebook = self._get_join_rulebook(max_chars=settings.join_rulebook_max_chars)

        # Keep flat relationships as fallback for less common tables
        relationships = self._get_relationships_from_schema()

        # Combine detailed schema with explicit FK relationship map so the LLM
        # always sees how tables connect (helps it pick correct join keys)
        schema_section = context
        if relationships:
            schema_section = f"""{context}

{relationships}"""

        # Optional prompt budget guardrail
        def _truncate_to_budget(text: str, budget: int) -> str:
            if budget and len(text) > budget:
                return text[:budget] + "\n-- TRUNCATED TO FIT PROMPT BUDGET --"
            return text

        # Build prompt body
        prompt_body = f"""You are a Microsoft SQL Server T-SQL expert for ConcordDb database.

Standard aliases: Client=c, ClientAgreement=ca, Order=o, OrderItem=oi, Product=p, Sale=s, ProductAvailability=pa, Storage=st, Region=r, User=u, Payment=pay, Invoice=inv, Debt=d

{join_templates}
{precomputed_templates}
{join_rulebook}
{self.CRITICAL_RULES}

DATABASE SCHEMA (3 tiers):
- TIER 1: Core tables with FULL column details (use these first)
- TIER 2: Relevant tables with KEY columns only (ID, FKs, important fields)
- TIER 3: Other tables as compact list - TableName(col1, col2...) format

{schema_section}

SQL RULES:
1. Use ONLY columns listed above - NEVER invent or guess column names
2. Use TOP N instead of LIMIT
3. Use dbo.[Order] with brackets (Order is reserved word)
4. Add WHERE alias.Deleted = 0 for active records
5. Use N'text' prefix for Ukrainian text in LIKE patterns

{examples_section}
{retry_context}
QUESTION: {question}

Generate ONLY the SQL query, no explanation:
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
                return self._fix_sql_dialect(matches[0].strip().rstrip(";"))
        # Try to find SELECT statement
        select_pattern = r"(SELECT\s+.*?)(?=\n\n|$)"
        matches = re.findall(select_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if matches:
            return self._fix_sql_dialect(matches[0].strip().rstrip(";"))
        return self._fix_sql_dialect(llm_response.strip().rstrip(";"))

    # Common column name hallucinations and their corrections
    COLUMN_FIXES = {
        # Client table hallucinations
        r'\bLegalassment\b': 'LegalAddress',
        r'\blegalassment\b': 'LegalAddress',
        r'\bActualizationment\b': 'ActualAddress',
        r'\bactualizationment\b': 'ActualAddress',
        r'\bLegalAdress\b': 'LegalAddress',
        r'\bActualAdress\b': 'ActualAddress',
        r'\bClientName\b': 'Name',
        r'\bclientname\b': 'Name',
        r'\bAddress\b(?!\s*LIKE)': 'LegalAddress',  # Generic Address -> LegalAddress
        r'\bPhone\b': 'MobileNumber',  # Phone -> MobileNumber
        r'\bphone\b': 'MobileNumber',
        r'\bTelephone\b': 'MobileNumber',
        r'\bCity\b': 'LegalAddress',  # City doesn't exist, use LegalAddress
        r'\bEmail\b': 'EmailAddress',  # Email -> EmailAddress
        r'\bemail\b': 'EmailAddress',
        # Product table hallucinations
        r'\bProductName\b': 'Name',
        r'\bproductname\b': 'Name',
        r'\bArticle\b': 'VendorCode',
        r'\bCode\b(?!\s*=)': 'VendorCode',
        # Common typos in column names
        r'\bClientAgrementID\b': 'ClientAgreementID',  # Missing 'e' typo
        r'\bClientAgreementId\b': 'ClientAgreementID',  # Case fix
        # Table name typos
        r'\bClientAgrement\b': 'ClientAgreement',  # Missing 'e' typo in table name
    }

    def _fix_sql_dialect(self, sql: str) -> str:
        """Fix PostgreSQL/MySQL syntax to T-SQL and correct hallucinated column names."""
        # Fix duplicate WHERE keyword: "WHERE WHERE.Deleted" -> "WHERE Deleted"
        original = sql
        sql = re.sub(r'\bWHERE\s+WHERE\.', 'WHERE ', sql, flags=re.IGNORECASE)
        if 'WHERE WHERE' in original:
            logger.debug(f"WHERE WHERE fix: '{original[:100]}' -> '{sql[:100]}'")
        # Convert LIMIT to TOP
        limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", sql, re.IGNORECASE)
        if limit_match:
            sql = re.sub(r"\bLIMIT\s+\d+\s*$", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"^SELECT\b", f"SELECT TOP {limit_match.group(1)}", sql, flags=re.IGNORECASE)
        # Fix Order table (reserved word) - need brackets
        sql = re.sub(r"\bdbo\.Order\b(?!\s*Item)", "dbo.[Order]", sql, flags=re.IGNORECASE)
        # Remove NULLS LAST/FIRST (PostgreSQL syntax, not valid in T-SQL)
        sql = re.sub(r"\s+NULLS\s+(LAST|FIRST)\s*", " ", sql, flags=re.IGNORECASE)
        # Fix hallucinated column names
        for wrong_pattern, correct_name in self.COLUMN_FIXES.items():
            sql = re.sub(wrong_pattern, correct_name, sql, flags=re.IGNORECASE)
        # Fix ambiguous Deleted column (add table alias if missing)
        sql = self._fix_ambiguous_deleted(sql)
        # Fix incorrect ClientAgreement join patterns
        sql = self._fix_client_agreement_joins(sql)
        # Fix incorrect Order.ClientID join patterns (LLM hallucination)
        sql = self._fix_order_client_join(sql)
        # Fix Client.Region column hallucination
        sql = self._fix_region_column(sql)
        return sql.strip()

    def _fix_ambiguous_deleted(self, sql: str) -> str:
        """Fix unqualified 'Deleted' column references by inferring table alias."""
        # Find all table aliases used in the query
        alias_pattern = r'\b(?:FROM|JOIN)\s+(?:dbo\.)?(\[?\w+\]?)\s+(?:AS\s+)?(\w+)\b'
        aliases = re.findall(alias_pattern, sql, re.IGNORECASE)

        if not aliases:
            return sql

        # Get first alias (usually main table)
        first_alias = aliases[0][1] if aliases[0][1] else aliases[0][0][0].lower()

        # Fix "WHERE Deleted = 0" without alias
        sql = re.sub(
            r'\bWHERE\s+Deleted\s*=\s*0\b',
            f'WHERE {first_alias}.Deleted = 0',
            sql,
            flags=re.IGNORECASE
        )
        # Fix "AND Deleted = 0" without alias
        sql = re.sub(
            r'\bAND\s+Deleted\s*=\s*0\b',
            f'AND {first_alias}.Deleted = 0',
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

        if sql != original:
            logger.debug(f"ClientAgreement join fix applied: {sql[:150]}...")

        return sql

    def _fix_region_column(self, sql: str) -> str:
        """Fix Client.Region column hallucination.

        Client table doesn't have a Region column, it has RegionID (FK to Region table).
        Pattern: c.Region LIKE N'XM%' (or = 'XM')
        Fix to: c.RegionID IN (SELECT ID FROM dbo.Region WHERE Name LIKE N'XM%')
        """
        original = sql

        # Pattern: alias.Region LIKE N'value%'
        sql = re.sub(
            r'(\w+)\.Region\s+LIKE\s+(N\'[^\']+\')',
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

        if sql != original:
            logger.info(f"Region column fix applied: {sql[:200]}...")

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

    # Common schema hallucinations to detect and reject (triggers retry)
    SCHEMA_ERRORS = [
        # Removed Order.ClientID - now fixed in _fix_order_client_join
        (r'\bProductBrand\b', "ProductBrand table does not exist! Use Product.VendorCode LIKE N'BRAND%'"),
        (r'\bClientAddress\b', "ClientAddress does not exist! Use Client.LegalAddress or Client.ActualAddress"),
        (r'\bOrderClient\b', "OrderClient does not exist! Use Client->ClientAgreement->Order"),
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
        text(sql_query)

    def _check_read_only(self, sql_query: str) -> None:
        for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]:
            if re.search(rf"\b{kw}\b", sql_query.upper()):
                raise ValueError(f"Write operation not allowed: {kw}")
