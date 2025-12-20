# Text-to-SQL Enhancement Strategies - Deep Dive Analysis

## Executive Summary

After analyzing the current codebase, I've identified that **we're treating symptoms (regex post-processing) rather than root causes (poor LLM context)**. The proposed strategies address the root causes.

**Current Pain Points:**
- Join hallucinations → `_fix_order_client_join()`, `_fix_client_agreement_joins()`
- Column hallucinations → 15+ regex patterns in `COLUMN_FIXES`
- Region column confusion → `_fix_region_column()`
- 782 FK relationships dumped as flat list → LLM can't parse effectively

---

## Current System Architecture Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CURRENT FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Query: "Show clients from XM with order count"               │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ TABLE SELECTOR (table_selector.py)                          │   │
│  │                                                              │   │
│  │ TIER 1: 8 Core Tables (hardcoded)                           │   │
│  │   Product, OrderItem, Order, Client, ClientAgreement...     │   │
│  │   → Full CREATE TABLE statements                             │   │
│  │                                                              │   │
│  │ TIER 2: RAG-selected tables (top_k=10)                      │   │
│  │   → Key columns only                                         │   │
│  │                                                              │   │
│  │ TIER 3: All other non-empty tables                          │   │
│  │   → One-liner format: TableName(col1, col2...)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ RELATIONSHIP EXTRACTION (sql_agent.py:76-102)               │   │
│  │                                                              │   │
│  │ ALL TABLE RELATIONSHIPS (from schema FKs):                   │   │
│  │ - dbo.ClientAgreement.ClientID -> Client.ID                 │   │
│  │ - dbo.Order.ClientAgreementID -> ClientAgreement.ID         │   │
│  │ - ... 782 more relationships (FLAT LIST!)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ PROMPT ASSEMBLY (sql_agent.py:283-334)                      │   │
│  │                                                              │   │
│  │ + CRITICAL_RULES (hardcoded business rules)                 │   │
│  │ + Few-shot examples (1 example, semantically matched)       │   │
│  │ + Previous attempt errors (if retry)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│                      [codellama:34b]                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ POST-PROCESSING FIXES (sql_agent.py:375-540)                │   │
│  │                                                              │   │
│  │ - _fix_sql_dialect() → LIMIT→TOP, Order brackets            │   │
│  │ - COLUMN_FIXES → 15 regex patterns                          │   │
│  │ - _fix_ambiguous_deleted() → Add table aliases              │   │
│  │ - _fix_client_agreement_joins() → Fix join patterns         │   │
│  │ - _fix_order_client_join() → Order.ClientID fix             │   │
│  │ - _fix_region_column() → Region→RegionID subquery           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Observation:** We have 6+ post-processing functions fixing LLM output, but we should be preventing these errors with better context.

---

## Strategy Deep Dive

### Strategy 1: Enhanced Context Generation

#### 1.1 Row Counts in All Tiers

**Current State:** Row counts stored in metadata but not in prompt context.

```python
# table_selector.py:168 - we HAVE row counts
metadata = {
    "row_count": table_info.get("row_count", 0),  # ← Not in prompt!
}
```

**Problem:** LLM doesn't know:
- Which tables are huge (need TOP/pagination)
- Which are lookup tables (safe to JOIN freely)
- Which are empty (skip)

**Implementation:**
```python
# In _format_as_create_table():
def _format_as_create_table(self, table_name: str, table_info: Dict[str, Any]) -> str:
    row_count = table_info.get("row_count", 0)
    size_hint = "lookup" if row_count < 100 else "medium" if row_count < 100000 else "large"

    header = f"CREATE TABLE {table_name}  -- {row_count:,} rows ({size_hint})\n("
    # ... rest of columns
```

**Result in prompt:**
```sql
CREATE TABLE dbo.Client  -- 50,234 rows (medium)
CREATE TABLE dbo.Region  -- 25 rows (lookup)
CREATE TABLE dbo.[Order]  -- 2,345,678 rows (large)
```

**Impact:** LLM knows to use TOP for large tables, can freely join lookup tables.

---

#### 1.2 Sample Data in Tier 1

**Current State:** Sample data extracted but not consistently used in prompt.

```python
# table_selector.py:154-160 - sample data exists but limited
if table_info.get("sample_data"):
    sample = table_info["sample_data"][0]
    sample_str = ", ".join(f"{k}={v}" for k, v in list(sample.items())[:5])
```

**Problem:** LLM doesn't see:
- Real Ukrainian text patterns
- Actual data formats
- Valid enum values

**Implementation:**
```python
# Enhanced sample data for Tier 1 tables
def _format_sample_data(self, table_info: Dict[str, Any]) -> str:
    """Format 2-3 representative sample rows."""
    samples = table_info.get("sample_data", [])[:3]
    if not samples:
        return ""

    lines = ["/* Sample data:"]
    for sample in samples:
        # Format key columns only
        row_str = " | ".join(f"{k}={v}" for k, v in list(sample.items())[:6])
        lines.append(f"   {row_str}")
    lines.append("*/")
    return "\n".join(lines)
```

**Result in prompt:**
```sql
CREATE TABLE dbo.Product (
    ID bigint PRIMARY KEY,
    Name nvarchar(255),
    VendorCode nvarchar(50),  -- Brand prefix: SEM, MG, IVECO
    ...
);
/* Sample data:
   ID=1001 | Name=Гальмівні колодки | VendorCode=SEM001
   ID=1002 | Name=Фільтр масляний | VendorCode=MG002
*/
```

**Impact:** LLM sees real Ukrainian text, understands VendorCode = brand prefix.

---

#### 1.3 Enum Column Auto-Detection ⭐ HIGH VALUE

**Current State:** No enum detection. LLM hallucinates status values.

**Problem:** For columns like `Status`, `Type`, `Category`, LLM invents values:
- Hallucinates: `WHERE Status = 'Active'`
- Reality: `WHERE Status = 1` (it's a numeric enum)

**Implementation:**
```python
# In schema_extractor.py - add enum detection
def _detect_enum_columns(self, table_name: str, columns: List[Dict]) -> Dict[str, List]:
    """Detect columns with few distinct values (likely enums)."""
    enum_columns = {}

    for col in columns:
        col_name = col["name"]
        col_type = col["type"].lower()

        # Skip PKs, FKs, timestamps
        if col_name in ["ID", "Created", "Updated"] or col_name.endswith("ID"):
            continue

        # Check for enum-like patterns
        if col_type in ["int", "smallint", "tinyint", "bit", "nvarchar"] and col.get("max_length", 100) < 50:
            try:
                query = f"SELECT DISTINCT [{col_name}] FROM [{table_name}] WHERE [{col_name}] IS NOT NULL"
                result = self.engine.execute(text(query)).fetchall()
                if len(result) <= 20:  # Enum threshold
                    values = [str(r[0]) for r in result]
                    enum_columns[col_name] = values
            except:
                pass

    return enum_columns
```

**Result in prompt:**
```sql
CREATE TABLE dbo.[Order] (
    ID bigint PRIMARY KEY,
    Status int,  -- ENUM: [0, 1, 2, 3, 4] (0=New, 1=Processing, 2=Shipped, 3=Delivered, 4=Cancelled)
    ...
);
```

**Impact:** No more hallucinated status values. LLM sees exact valid values.

---

#### 1.4 Column Purpose Tagging ⭐ CRITICAL

**Current State:** Column names only, no semantic meaning.

**Problem:** LLM doesn't understand:
- `RegionID` is FK to Region.ID, not a region name
- `VendorCode` is brand prefix, not vendor reference
- `ClientAgreementID` in Order links to ClientAgreement, NOT Client

**Implementation - Two Approaches:**

**A) Automatic from FK analysis:**
```python
def _get_column_purpose(self, col_name: str, table_name: str, fks: List[Dict]) -> str:
    """Infer column purpose from FKs and naming patterns."""

    # Check if it's a FK
    for fk in fks:
        if col_name in fk.get("columns", []):
            return f"FK→{fk['referred_table']}.{fk['referred_columns'][0]}"

    # Pattern-based inference
    if col_name.endswith("ID") and col_name != "ID":
        # Likely FK even if not declared
        potential_table = col_name[:-2]  # Remove "ID"
        return f"FK→{potential_table}.ID (inferred)"

    return None
```

**B) Manual annotations for business-critical columns:**
```python
# New file: column_annotations.py
COLUMN_ANNOTATIONS = {
    "Client": {
        "RegionID": "FK→Region.ID. Join Region to get region name!",
        "LegalAddress": "Company registration address (contains city/region text)",
        "ActualAddress": "Physical location address",
    },
    "Order": {
        "ClientAgreementID": "FK→ClientAgreement.ID. NOT directly to Client! Path: Order→ClientAgreement→Client",
    },
    "Product": {
        "VendorCode": "Brand prefix (SEM=Semperit, MG=MG). Use LIKE N'SEM%' for brand filtering",
    },
}
```

**Result in prompt:**
```sql
CREATE TABLE dbo.Client (
    ID bigint PRIMARY KEY,
    Name nvarchar(255),
    RegionID bigint,  -- FK→Region.ID (join Region for name!)
    LegalAddress nvarchar(500),  -- Registration address (contains city)
    ...
);
```

**Impact:** LLM understands `c.RegionID` needs JOIN, not LIKE. Prevents `_fix_region_column()` need.

---

### Strategy 2: Relationship Graph Enhancement ⭐⭐⭐ HIGHEST IMPACT

#### 2.1 Pre-computed Join Path Templates

**Current State:** 782 FK relationships as flat list (sql_agent.py:76-102)

```
ALL TABLE RELATIONSHIPS (from schema FKs):
- dbo.ClientAgreement.ClientID -> Client.ID
- dbo.Order.ClientAgreementID -> ClientAgreement.ID
- dbo.OrderItem.OrderID -> Order.ID
- ... 779 more
```

**Problem:** LLM can't infer multi-hop paths from flat list. Result:
- Hallucinates `Order.ClientID` (doesn't exist!)
- Needs 5+ post-processing fixes

**Implementation - Join Path Graph:**

```python
# New file: join_path_graph.py
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class JoinPathGraph:
    """Pre-compute and cache optimal join paths between tables."""

    def __init__(self, schema: Dict[str, Any]):
        self.graph = defaultdict(list)  # table -> [(neighbor, join_condition)]
        self._build_graph(schema)
        self._path_cache = {}

    def _build_graph(self, schema: Dict[str, Any]):
        """Build bidirectional graph from FKs."""
        for table_name, table_info in schema["tables"].items():
            for fk in table_info.get("foreign_keys", []):
                parent = table_name
                child = fk["referred_table"]
                cols = fk["columns"]
                ref_cols = fk["referred_columns"]

                # Add both directions
                self.graph[parent].append((child, f"{parent}.{cols[0]} = {child}.{ref_cols[0]}"))
                self.graph[child].append((parent, f"{parent}.{cols[0]} = {child}.{ref_cols[0]}"))

    def find_join_path(self, source: str, target: str) -> Optional[List[Tuple[str, str, str]]]:
        """Find shortest path using BFS. Returns [(table, alias, join_condition), ...]"""
        if source == target:
            return []

        cache_key = f"{source}:{target}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # BFS for shortest path
        from collections import deque
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            for neighbor, condition in self.graph[current]:
                if neighbor == target:
                    # Found! Build join sequence
                    full_path = path + [neighbor]
                    result = self._path_to_joins(full_path)
                    self._path_cache[cache_key] = result
                    return result

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    def _path_to_joins(self, path: List[str]) -> List[Tuple[str, str, str]]:
        """Convert path to join statements with aliases."""
        result = []
        aliases = {}

        for i, table in enumerate(path):
            alias = table[0].lower() if table not in aliases else f"{table[0].lower()}{i}"
            aliases[table] = alias

            if i > 0:
                prev_table = path[i-1]
                # Find the join condition
                for neighbor, condition in self.graph[prev_table]:
                    if neighbor == table:
                        # Replace table names with aliases
                        cond = condition.replace(prev_table, aliases[prev_table])
                        cond = cond.replace(table, alias)
                        result.append((table, alias, cond))
                        break

        return result

    def get_join_template(self, tables: List[str]) -> str:
        """Generate complete JOIN template for a set of tables."""
        if len(tables) < 2:
            return ""

        # Find paths between all pairs, build optimal join order
        main_table = tables[0]
        joins = []
        included = {main_table}

        for table in tables[1:]:
            if table in included:
                continue

            # Find path from any included table
            for source in included:
                path = self.find_join_path(source, table)
                if path:
                    for tbl, alias, condition in path:
                        if tbl not in included:
                            joins.append(f"JOIN dbo.{tbl} {alias} ON {condition}")
                            included.add(tbl)
                    break

        return "\n".join(joins)
```

**Usage in sql_agent.py:**
```python
# Instead of flat FK list, generate targeted join templates
def _get_join_templates_for_query(self, selected_tables: List[str]) -> str:
    """Generate JOIN templates for tables selected by RAG."""

    # Common high-value paths (pre-computed)
    COMMON_PATHS = {
        ("Client", "Order"): """
            -- Client → Order path (CRITICAL - no direct FK!):
            FROM dbo.Client c
            JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID
            JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID
        """,
        ("Product", "Order"): """
            -- Product → Order path:
            FROM dbo.Product p
            JOIN dbo.OrderItem oi ON oi.ProductID = p.ID
            JOIN dbo.[Order] o ON oi.OrderID = o.ID
        """,
        ("Client", "Region"): """
            -- Client → Region path:
            FROM dbo.Client c
            JOIN dbo.Region r ON c.RegionID = r.ID
        """,
    }

    # Check which common paths are relevant
    relevant_paths = []
    table_set = set(t.replace("dbo.", "").replace("[", "").replace("]", "")
                    for t in selected_tables)

    for (t1, t2), template in COMMON_PATHS.items():
        if t1 in table_set and t2 in table_set:
            relevant_paths.append(template)

    if relevant_paths:
        return "JOIN PATH TEMPLATES (use these exact patterns!):\n" + "\n".join(relevant_paths)

    return ""
```

**Result in prompt:**
```
JOIN PATH TEMPLATES (use these exact patterns!):

-- Client → Order path (CRITICAL - no direct FK!):
FROM dbo.Client c
JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID
JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID

-- Client → Region path:
FROM dbo.Client c
JOIN dbo.Region r ON c.RegionID = r.ID
```

**Impact:**
- Eliminates `Order.ClientID` hallucination
- Removes need for `_fix_order_client_join()` and `_fix_client_agreement_joins()`
- LLM copies exact pattern instead of guessing

---

#### 2.2 Dynamic Relationship Discovery

**Current State:** Static FK dump.

**Enhancement:** When RAG selects tables, dynamically compute:
1. Which tables need joining?
2. Are there ambiguous paths?
3. What's the optimal join order?

```python
def _analyze_table_relationships(self, selected_tables: List[str]) -> Dict:
    """Analyze relationships between RAG-selected tables."""

    result = {
        "direct_joins": [],      # Tables with direct FK
        "indirect_joins": [],    # Tables needing intermediate table
        "no_relationship": [],   # Tables that can't be joined
        "join_order": [],        # Optimal FROM/JOIN sequence
    }

    # Build mini-graph for selected tables
    for i, t1 in enumerate(selected_tables):
        for t2 in selected_tables[i+1:]:
            path = self.join_graph.find_join_path(t1, t2)
            if path:
                if len(path) == 1:
                    result["direct_joins"].append((t1, t2, path[0][2]))
                else:
                    result["indirect_joins"].append((t1, t2, [p[0] for p in path]))
            else:
                result["no_relationship"].append((t1, t2))

    return result
```

---

### Strategy 3: Business Rules Automation

#### 3.1 Auto-detect Soft Delete Tables

**Current State:** Hardcoded in CRITICAL_RULES + post-processing fix.

```python
# sql_agent.py:265-267
"2. ALWAYS use table alias for Deleted column (it exists in ALL tables)"
```

**Enhancement:** Detect at schema extraction time:

```python
# In schema_extractor.py
def _analyze_table_patterns(self, table_info: Dict) -> Dict:
    """Detect common patterns like soft delete, audit fields."""
    patterns = {
        "has_soft_delete": False,
        "has_audit_fields": False,
        "audit_created": None,
        "audit_updated": None,
    }

    columns = {c["name"].lower(): c for c in table_info.get("columns", [])}

    # Soft delete detection
    if "deleted" in columns and columns["deleted"]["type"].lower() == "bit":
        patterns["has_soft_delete"] = True

    # Audit field detection
    for name in ["created", "createdat", "createddate", "datecreated"]:
        if name in columns:
            patterns["has_audit_fields"] = True
            patterns["audit_created"] = columns[name]["name"]
            break

    for name in ["updated", "updatedat", "modifieddate", "lastmodified"]:
        if name in columns:
            patterns["audit_updated"] = columns[name]["name"]
            break

    return patterns
```

**Result in context:**
```sql
CREATE TABLE dbo.Client (
    ...
) -- Soft-delete: ADD "WHERE c.Deleted = 0" | Audit: Created, Updated
```

---

### Strategy 4: Schema-Aware Few-Shot Examples

#### 4.1 Table-Aware Example Matching

**Current State:** Semantic similarity only (retriever.py).

```python
# Current: Match by text similarity
similar_examples = self.example_retriever.find_similar_with_correction(question, top_k=1)
```

**Problem:**
- Query about "clients with orders" might match example about "client addresses"
- Both contain "client" but use different tables/joins

**Enhancement:**

```python
# In retriever.py
def find_similar_with_table_awareness(
    self,
    query: str,
    selected_tables: List[str],  # NEW: tables from RAG
    top_k: int = 3
) -> List[Dict]:
    """Find examples matching both semantics AND table usage."""

    # Get more candidates for re-ranking
    semantic_results = self.find_similar_with_correction(query, top_k=top_k * 3)

    # Re-rank by table overlap
    for result in semantic_results:
        tables_used = set(result.get("metadata", {}).get("tables_used", []))
        selected_set = set(t.replace("dbo.", "") for t in selected_tables)

        # Calculate table overlap score
        overlap = len(tables_used & selected_set)
        max_possible = min(len(tables_used), len(selected_set))
        table_score = overlap / max_possible if max_possible > 0 else 0

        # Combined score: 60% semantic, 40% table match
        result["final_score"] = (
            0.6 * result["similarity_score"] +
            0.4 * table_score
        )

    # Sort by final score
    semantic_results.sort(key=lambda x: x["final_score"], reverse=True)

    return semantic_results[:top_k]
```

**Impact:** Examples now match by both meaning AND structure.

---

#### 4.2 Dynamic Example Count

**Current State:** Always 1 example (sql_agent.py:287).

**Enhancement:**
```python
def _determine_example_count(self, query: str, context: str) -> int:
    """Determine optimal example count based on query complexity."""

    # Complexity indicators
    complexity_score = 0

    # Multiple tables mentioned
    table_keywords = ["join", "клієнт", "замовлен", "товар", "продаж"]
    complexity_score += sum(1 for kw in table_keywords if kw in query.lower())

    # Aggregation keywords
    agg_keywords = ["топ", "top", "sum", "count", "group", "average", "total"]
    complexity_score += sum(2 for kw in agg_keywords if kw in query.lower())

    # Time-based keywords
    time_keywords = ["рік", "місяць", "year", "month", "period", "за"]
    complexity_score += sum(1 for kw in time_keywords if kw in query.lower())

    # Map score to example count
    if complexity_score <= 2:
        return 1  # Simple query
    elif complexity_score <= 5:
        return 2  # Medium complexity
    else:
        return 3  # Complex query
```

---

### Strategy 5: Schema Documentation Generation

This could generate:

1. **Business Glossary:**
```python
BUSINESS_GLOSSARY = {
    "uk": {
        "клієнт": "Client table",
        "замовлення": "Order table (use [Order] with brackets!)",
        "товар": "Product table",
        "продаж": "Sale table",
        "область": "Region table (Client.RegionID → Region.ID)",
    },
    "en": {
        "customer": "Client table",
        "order": "Order table",
        # ...
    }
}
```

2. **Domain Knowledge:**
```
DOMAIN KNOWLEDGE:
- Orders flow: Client → ClientAgreement → Order → OrderItem → Product
- Sales are linked to both Order (OrderID) and ClientAgreement (ClientAgreementID)
- Stock is in ProductAvailability.Amount, not Product table
- Brand filtering: Product.VendorCode LIKE 'BRAND%' (no ProductBrand table!)
```

---

## Implementation Priority Matrix

| Strategy | Files to Modify | Effort | Impact | ROI |
|----------|-----------------|--------|--------|-----|
| 2.1 Join Path Templates | `sql_agent.py`, new `join_path_graph.py` | Medium | CRITICAL | ⭐⭐⭐⭐⭐ |
| 1.4 Column Purpose Tags | `table_selector.py`, new `column_annotations.py` | Medium | HIGH | ⭐⭐⭐⭐ |
| 1.3 Enum Detection | `schema_extractor.py`, `table_selector.py` | Medium | HIGH | ⭐⭐⭐⭐ |
| 4.1 Table-Aware Examples | `retriever.py`, `sql_agent.py` | Low | HIGH | ⭐⭐⭐⭐ |
| 3.1 Soft Delete Auto | `schema_extractor.py` | Low | MEDIUM | ⭐⭐⭐ |
| 1.1 Row Counts | `table_selector.py` | Low | LOW | ⭐⭐ |
| 1.2 Sample Data | `table_selector.py` | Low | MEDIUM | ⭐⭐⭐ |
| 4.2 Dynamic Examples | `sql_agent.py` | Low | LOW | ⭐⭐ |
| 5.1 Schema Docs | New file | Medium | LOW | ⭐ |

---

## Recommended Implementation Order

### Phase 1: Eliminate Post-Processing Needs (Week 1)

1. **Join Path Templates (Strategy 2.1)**
   - Create `join_path_graph.py`
   - Pre-compute common paths (Client→Order, Product→Order, etc.)
   - Inject templates into prompt based on RAG-selected tables
   - **Success metric:** Remove `_fix_order_client_join()` and `_fix_client_agreement_joins()`

2. **Column Purpose Tagging (Strategy 1.4)**
   - Create `column_annotations.py` with manual annotations
   - Auto-detect FK purposes from schema
   - Add to CREATE TABLE output
   - **Success metric:** Remove `_fix_region_column()` and reduce COLUMN_FIXES

### Phase 2: Improve Context Quality (Week 2)

3. **Enum Column Detection (Strategy 1.3)**
   - Add to `schema_extractor.py`
   - Include enum values in column definitions
   - **Success metric:** No more hallucinated status/type values

4. **Table-Aware Example Matching (Strategy 4.1)**
   - Update `retriever.py` with table overlap scoring
   - **Success metric:** Examples match query structure, not just keywords

### Phase 3: Polish (Week 3)

5. **Row Count Context (Strategy 1.1)**
6. **Soft Delete Auto-detection (Strategy 3.1)**
7. **Sample Data Enhancement (Strategy 1.2)**

---

## Expected Results After Implementation

| Metric | Before | After |
|--------|--------|-------|
| Post-processing fixes needed | 6 functions | 1-2 functions |
| Join hallucinations | Common | Rare |
| Column name hallucinations | Common | Rare |
| Retry attempts needed | 2-3 | 0-1 |
| SQL generation latency | 10-30s | 5-15s |
| Query success rate | ~70% | ~90%+ |

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `join_path_graph.py` | CREATE | Join path computation |
| `column_annotations.py` | CREATE | Manual column annotations |
| `schema_extractor.py` | MODIFY | Add enum detection, pattern analysis |
| `table_selector.py` | MODIFY | Add row counts, column tags to output |
| `sql_agent.py` | MODIFY | Use join templates, reduce post-processing |
| `retriever.py` | MODIFY | Add table-aware matching |

---

## Conclusion

The proposed strategies address root causes rather than symptoms. By giving the LLM:

1. **Explicit join templates** → Eliminates join hallucinations
2. **Column purpose understanding** → Eliminates column confusion
3. **Valid enum values** → Eliminates value hallucinations
4. **Structure-aware examples** → Better few-shot learning

We can dramatically reduce the need for post-processing fixes and improve overall SQL generation quality.

The highest-ROI item is **Strategy 2.1 (Join Path Templates)** - it directly addresses the most common and most harmful hallucination type.
