"""Auto-generate SQL examples from schema FK relationships.

This script analyzes the database schema and generates training examples
for different query patterns, focusing on:
1. Uncovered tables (0% current coverage)
2. Multi-table JOIN patterns (3-5 tables)
3. Supply chain, financial, and complex business logic queries

Usage:
    python example_generator.py --output templates/supply_chain.json
    python example_generator.py --analyze  # Show coverage gaps
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict

from loguru import logger

# Add parent dir to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from schema_extractor import SchemaExtractor


class ExampleGenerator:
    """Auto-generate SQL examples from schema FK relationships."""

    # Query patterns for different complexity levels
    PATTERNS = {
        # 2-table patterns
        "2table_list": {
            "en": "List {table1} with {table2} info",
            "uk": "Список {table1} з інформацією про {table2}",
            "sql_template": "SELECT t1.ID, t1.Name, t2.Name as {table2}Name FROM dbo.{table1} t1 JOIN dbo.{table2} t2 ON t1.{fk_col} = t2.ID WHERE t1.Deleted = 0",
            "category": "list",
            "complexity": "simple"
        },
        "2table_count": {
            "en": "Count {table1} by {table2}",
            "uk": "Кількість {table1} по {table2}",
            "sql_template": "SELECT t2.Name, COUNT(t1.ID) as Count FROM dbo.{table1} t1 JOIN dbo.{table2} t2 ON t1.{fk_col} = t2.ID WHERE t1.Deleted = 0 GROUP BY t2.ID, t2.Name ORDER BY Count DESC",
            "category": "aggregation",
            "complexity": "medium"
        },
        "2table_sum": {
            "en": "Total amount by {table2}",
            "uk": "Сума по {table2}",
            "sql_template": "SELECT t2.Name, SUM(t1.{amount_col}) as Total FROM dbo.{table1} t1 JOIN dbo.{table2} t2 ON t1.{fk_col} = t2.ID WHERE t1.Deleted = 0 GROUP BY t2.ID, t2.Name ORDER BY Total DESC",
            "category": "aggregation",
            "complexity": "medium"
        },
    }

    # 3-table chain patterns
    CHAIN_PATTERNS_3 = {
        "3table_list": {
            "en": "{table1} with {table2} and {table3} details",
            "uk": "{table1} з деталями {table2} та {table3}",
            "category": "list",
            "complexity": "medium"
        },
        "3table_count": {
            "en": "Count {table1} grouped by {table3}",
            "uk": "Кількість {table1} по {table3}",
            "category": "aggregation",
            "complexity": "medium"
        },
    }

    # Ukrainian table name translations (common ones)
    TABLE_TRANSLATIONS = {
        "Client": "клієнтів",
        "ClientAgreement": "договорів",
        "Order": "замовлень",
        "OrderItem": "позицій",
        "Product": "товарів",
        "Sale": "продажів",
        "Organization": "організацій",
        "Storage": "складів",
        "Currency": "валют",
        "Region": "регіонів",
        "SupplyOrder": "замовлень постачальникам",
        "SupplyOrganization": "постачальників",
        "SupplyPaymentTask": "платежів постачальникам",
        "SupplyOrganizationAgreement": "договорів з постачальниками",
        "IncomePaymentOrder": "вхідних платежів",
        "OutcomePaymentOrder": "вихідних платежів",
        "TaxFree": "безподаткових операцій",
        "Agreement": "угод",
        "ProductAvailability": "залишків",
        "Debt": "боргів",
        "Price": "цін",
    }

    # Amount columns for aggregation patterns
    AMOUNT_COLUMNS = {
        "OrderItem": "Qty",
        "Sale": "TotalSum",
        "Order": "TotalSum",
        "SupplyOrder": "TotalSum",
        "ProductAvailability": "Amount",
        "IncomePaymentOrder": "Sum",
        "OutcomePaymentOrder": "Sum",
        "Debt": "Sum",
    }

    def __init__(self):
        """Initialize generator with schema extractor."""
        self.schema_extractor = SchemaExtractor()
        self.schema = self.schema_extractor.extract_full_schema()
        self._build_fk_graph()

    def _build_fk_graph(self):
        """Build graph of FK relationships between tables."""
        self.fk_graph = defaultdict(list)  # table -> [(fk_col, ref_table)]
        self.reverse_fk_graph = defaultdict(list)  # ref_table -> [(table, fk_col)]

        for table_name, table_info in self.schema['tables'].items():
            for fk in table_info.get('foreign_keys', []):
                if fk.get('columns'):
                    fk_col = fk['columns'][0]
                    ref_table = fk['referred_table'].replace('dbo.', '')

                    self.fk_graph[table_name].append((fk_col, ref_table))
                    self.reverse_fk_graph[ref_table].append((table_name, fk_col))

        logger.info(f"Built FK graph: {len(self.fk_graph)} tables with outgoing FKs")

    def analyze_coverage(self, existing_templates_dir: Path) -> Dict[str, Any]:
        """Analyze current example coverage vs schema.

        Args:
            existing_templates_dir: Path to templates directory

        Returns:
            Coverage analysis with gaps identified
        """
        # Load existing examples
        existing_tables = set()
        existing_examples = []
        table_example_count = defaultdict(int)

        for template_file in existing_templates_dir.glob("*.json"):
            with open(template_file, 'r', encoding='utf-8') as f:
                template = json.load(f)

            for example in template.get('examples', []):
                existing_examples.append(example)
                for table in example.get('tables_used', []):
                    existing_tables.add(table)
                    table_example_count[table] += 1

        # Count tables in schema
        all_tables = set()
        tables_with_data = set()
        tables_with_fks = set()

        for table_name, table_info in self.schema['tables'].items():
            all_tables.add(table_name)
            if table_info.get('row_count', 0) > 0:
                tables_with_data.add(table_name)
            if table_info.get('foreign_keys'):
                tables_with_fks.add(table_name)

        # Find gaps
        uncovered_tables = tables_with_data - existing_tables
        uncovered_with_fks = uncovered_tables & tables_with_fks

        # Analyze complexity distribution
        complexity_dist = defaultdict(int)
        tables_per_example = []
        for ex in existing_examples:
            tables_used = ex.get('tables_used', [])
            complexity_dist[len(tables_used)] += 1
            tables_per_example.append(len(tables_used))

        return {
            'total_tables': len(all_tables),
            'tables_with_data': len(tables_with_data),
            'tables_with_fks': len(tables_with_fks),
            'covered_tables': len(existing_tables),
            'uncovered_tables': sorted(uncovered_tables),
            'uncovered_with_fks': sorted(uncovered_with_fks),
            'total_examples': len(existing_examples),
            'complexity_distribution': dict(complexity_dist),
            'table_coverage': {
                table: count for table, count in sorted(
                    table_example_count.items(),
                    key=lambda x: -x[1]
                )
            },
            'avg_tables_per_example': sum(tables_per_example) / len(tables_per_example) if tables_per_example else 0,
        }

    def find_join_chains(self, max_depth: int = 5) -> List[List[str]]:
        """Find all valid FK chains up to max_depth.

        Args:
            max_depth: Maximum chain length

        Returns:
            List of table chains (paths through FK relationships)
        """
        chains = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) >= 2:
                chains.append(path.copy())

            if len(path) >= max_depth:
                return

            for fk_col, ref_table in self.fk_graph.get(current, []):
                if ref_table not in visited:
                    visited.add(ref_table)
                    path.append(ref_table)
                    dfs(ref_table, path, visited)
                    path.pop()
                    visited.remove(ref_table)

        # Start from each table
        for table_name in self.schema['tables'].keys():
            if self.schema['tables'][table_name].get('row_count', 0) > 0:
                dfs(table_name, [table_name], {table_name})

        # Remove duplicates (same set of tables)
        unique_chains = []
        seen = set()
        for chain in chains:
            key = tuple(sorted(chain))
            if key not in seen:
                seen.add(key)
                unique_chains.append(chain)

        return unique_chains

    def generate_2table_examples(self, tables: List[str]) -> List[Dict[str, Any]]:
        """Generate 2-table JOIN examples for specified tables.

        Args:
            tables: List of table names to generate examples for

        Returns:
            List of example dictionaries
        """
        examples = []
        example_id = 1

        for table1 in tables:
            # Get FKs for this table
            for fk_col, ref_table in self.fk_graph.get(table1, []):
                if ref_table not in self.schema['tables']:
                    continue

                # Skip if ref table is empty
                if self.schema['tables'][ref_table].get('row_count', 0) == 0:
                    continue

                # Generate list pattern
                table1_uk = self.TABLE_TRANSLATIONS.get(table1, table1)
                table2_uk = self.TABLE_TRANSLATIONS.get(ref_table, ref_table)

                examples.append({
                    "id": f"gen_2table_{table1.lower()}_{example_id:03d}",
                    "category": "list",
                    "complexity": "simple",
                    "question_en": f"List {table1} with {ref_table} information",
                    "question_uk": f"Список {table1_uk} з інформацією про {table2_uk}",
                    "variations_en": [
                        f"show {table1.lower()} and {ref_table.lower()}",
                        f"{table1.lower()} with their {ref_table.lower()}",
                    ],
                    "variations_uk": [
                        f"покажи {table1_uk} та {table2_uk}",
                    ],
                    "sql": f"SELECT t1.ID, t1.Name, t2.Name as {ref_table}Name FROM dbo.[{table1}] t1 JOIN dbo.[{ref_table}] t2 ON t1.{fk_col} = t2.ID WHERE t1.Deleted = 0 ORDER BY t1.Name",
                    "tables_used": [table1, ref_table]
                })
                example_id += 1

                # Generate count pattern
                examples.append({
                    "id": f"gen_2table_{table1.lower()}_{example_id:03d}",
                    "category": "aggregation",
                    "complexity": "medium",
                    "question_en": f"Count {table1} by {ref_table}",
                    "question_uk": f"Кількість {table1_uk} по {table2_uk}",
                    "variations_en": [
                        f"how many {table1.lower()} per {ref_table.lower()}",
                        f"{table1.lower()} count grouped by {ref_table.lower()}",
                    ],
                    "variations_uk": [
                        f"скільки {table1_uk} на кожен {table2_uk}",
                    ],
                    "sql": f"SELECT t2.Name as {ref_table}Name, COUNT(t1.ID) as Count FROM dbo.[{table1}] t1 JOIN dbo.[{ref_table}] t2 ON t1.{fk_col} = t2.ID WHERE t1.Deleted = 0 GROUP BY t2.ID, t2.Name ORDER BY Count DESC",
                    "tables_used": [table1, ref_table]
                })
                example_id += 1

                # Generate sum pattern if table has amount column
                if table1 in self.AMOUNT_COLUMNS:
                    amount_col = self.AMOUNT_COLUMNS[table1]
                    examples.append({
                        "id": f"gen_2table_{table1.lower()}_{example_id:03d}",
                        "category": "aggregation",
                        "complexity": "medium",
                        "question_en": f"Total {amount_col} by {ref_table}",
                        "question_uk": f"Сума {amount_col} по {table2_uk}",
                        "variations_en": [
                            f"sum of {table1.lower()} {amount_col.lower()} grouped by {ref_table.lower()}",
                        ],
                        "variations_uk": [
                            f"загальна сума {amount_col} по {table2_uk}",
                        ],
                        "sql": f"SELECT t2.Name as {ref_table}Name, SUM(t1.{amount_col}) as Total FROM dbo.[{table1}] t1 JOIN dbo.[{ref_table}] t2 ON t1.{fk_col} = t2.ID WHERE t1.Deleted = 0 GROUP BY t2.ID, t2.Name ORDER BY Total DESC",
                        "tables_used": [table1, ref_table]
                    })
                    example_id += 1

        return examples

    def generate_3table_examples(self, chains: List[List[str]]) -> List[Dict[str, Any]]:
        """Generate 3-table JOIN examples from FK chains.

        Args:
            chains: List of 3-table chains

        Returns:
            List of example dictionaries
        """
        examples = []
        example_id = 1

        for chain in chains:
            if len(chain) != 3:
                continue

            table1, table2, table3 = chain

            # Find FK columns
            fk1_col = None
            for fk_col, ref in self.fk_graph.get(table1, []):
                if ref == table2:
                    fk1_col = fk_col
                    break

            fk2_col = None
            for fk_col, ref in self.fk_graph.get(table2, []):
                if ref == table3:
                    fk2_col = fk_col
                    break

            if not fk1_col or not fk2_col:
                continue

            # Generate list pattern
            table1_uk = self.TABLE_TRANSLATIONS.get(table1, table1)
            table3_uk = self.TABLE_TRANSLATIONS.get(table3, table3)

            examples.append({
                "id": f"gen_3table_{example_id:03d}",
                "category": "list",
                "complexity": "medium",
                "question_en": f"{table1} with {table2} and {table3} details",
                "question_uk": f"{table1} з деталями {table2} та {table3}",
                "variations_en": [
                    f"show {table1.lower()} joined with {table2.lower()} and {table3.lower()}",
                ],
                "variations_uk": [
                    f"{table1_uk} з {table3_uk}",
                ],
                "sql": f"SELECT t1.ID, t1.Name, t2.Name as {table2}Name, t3.Name as {table3}Name FROM dbo.[{table1}] t1 JOIN dbo.[{table2}] t2 ON t1.{fk1_col} = t2.ID JOIN dbo.[{table3}] t3 ON t2.{fk2_col} = t3.ID WHERE t1.Deleted = 0 ORDER BY t1.Name",
                "tables_used": [table1, table2, table3]
            })
            example_id += 1

            # Generate count by end table
            examples.append({
                "id": f"gen_3table_{example_id:03d}",
                "category": "aggregation",
                "complexity": "complex",
                "question_en": f"Count {table1} grouped by {table3}",
                "question_uk": f"Кількість {table1_uk} по {table3_uk}",
                "variations_en": [
                    f"how many {table1.lower()} per {table3.lower()}",
                ],
                "variations_uk": [
                    f"скільки {table1_uk} на {table3_uk}",
                ],
                "sql": f"SELECT t3.Name as {table3}Name, COUNT(t1.ID) as Count FROM dbo.[{table1}] t1 JOIN dbo.[{table2}] t2 ON t1.{fk1_col} = t2.ID JOIN dbo.[{table3}] t3 ON t2.{fk2_col} = t3.ID WHERE t1.Deleted = 0 GROUP BY t3.ID, t3.Name ORDER BY Count DESC",
                "tables_used": [table1, table2, table3]
            })
            example_id += 1

        return examples

    def generate_supply_chain_examples(self) -> List[Dict[str, Any]]:
        """Generate examples for supply chain domain.

        Returns:
            List of supply chain example dictionaries
        """
        examples = []

        # Supply order examples
        examples.extend([
            {
                "id": "supply_order_001",
                "category": "list",
                "complexity": "simple",
                "question_en": "List all supply orders",
                "question_uk": "Список всіх замовлень постачальникам",
                "variations_en": ["show supply orders", "vendor orders", "purchase orders"],
                "variations_uk": ["покажи замовлення постачальникам", "закупівлі"],
                "sql": "SELECT TOP 100 ID, Created, TotalSum, Status FROM dbo.SupplyOrder WHERE Deleted = 0 ORDER BY Created DESC",
                "tables_used": ["SupplyOrder"]
            },
            {
                "id": "supply_order_by_org_002",
                "category": "list",
                "complexity": "medium",
                "question_en": "Supply orders by organization",
                "question_uk": "Замовлення постачальникам по організаціях",
                "variations_en": ["orders per organization", "show supply orders with org"],
                "variations_uk": ["замовлення по організаціям", "закупівлі організацій"],
                "sql": "SELECT o.ID, o.Created, o.TotalSum, org.Name as OrganizationName FROM dbo.SupplyOrder o JOIN dbo.Organization org ON o.OrganizationID = org.ID WHERE o.Deleted = 0 ORDER BY o.Created DESC",
                "tables_used": ["SupplyOrder", "Organization"]
            },
            {
                "id": "supply_order_total_by_org_003",
                "category": "aggregation",
                "complexity": "medium",
                "question_en": "Total supply order amount by organization",
                "question_uk": "Сума замовлень постачальникам по організаціях",
                "variations_en": ["purchase totals by org", "supply spending per organization"],
                "variations_uk": ["закупівлі по організаціям", "сума закупівель"],
                "sql": "SELECT org.Name as OrganizationName, SUM(o.TotalSum) as TotalAmount, COUNT(o.ID) as OrderCount FROM dbo.SupplyOrder o JOIN dbo.Organization org ON o.OrganizationID = org.ID WHERE o.Deleted = 0 GROUP BY org.ID, org.Name ORDER BY TotalAmount DESC",
                "tables_used": ["SupplyOrder", "Organization"]
            },
            {
                "id": "supply_vendor_list_004",
                "category": "list",
                "complexity": "simple",
                "question_en": "List all vendors/suppliers",
                "question_uk": "Список всіх постачальників",
                "variations_en": ["show vendors", "suppliers list", "supply organizations"],
                "variations_uk": ["покажи постачальників", "список постачальників"],
                "sql": "SELECT ID, Name, Phone, Address FROM dbo.SupplyOrganization WHERE Deleted = 0 ORDER BY Name",
                "tables_used": ["SupplyOrganization"]
            },
            {
                "id": "supply_orders_by_vendor_005",
                "category": "list",
                "complexity": "medium",
                "question_en": "Supply orders with vendor information",
                "question_uk": "Замовлення постачальникам з інформацією про постачальника",
                "variations_en": ["orders by vendor", "purchases from suppliers"],
                "variations_uk": ["замовлення по постачальникам"],
                "sql": "SELECT o.ID, o.Created, o.TotalSum, so.Name as VendorName FROM dbo.SupplyOrder o JOIN dbo.SupplyOrganizationAgreement soa ON o.SupplyOrganizationAgreementID = soa.ID JOIN dbo.SupplyOrganization so ON soa.SupplyOrganizationID = so.ID WHERE o.Deleted = 0 ORDER BY o.Created DESC",
                "tables_used": ["SupplyOrder", "SupplyOrganizationAgreement", "SupplyOrganization"]
            },
            {
                "id": "supply_payment_tasks_006",
                "category": "list",
                "complexity": "simple",
                "question_en": "List supply payment tasks",
                "question_uk": "Список платежів постачальникам",
                "variations_en": ["vendor payments", "supply payments", "payment tasks"],
                "variations_uk": ["платежі постачальникам", "заплановані платежі"],
                "sql": "SELECT TOP 100 ID, Created, Sum, Status FROM dbo.SupplyPaymentTask WHERE Deleted = 0 ORDER BY Created DESC",
                "tables_used": ["SupplyPaymentTask"]
            },
            {
                "id": "supply_payment_by_vendor_007",
                "category": "aggregation",
                "complexity": "complex",
                "question_en": "Total payments by vendor",
                "question_uk": "Сума платежів по постачальникам",
                "variations_en": ["payment amounts per supplier", "vendor payment totals"],
                "variations_uk": ["платежі постачальникам сума", "скільки заплатили постачальникам"],
                "sql": "SELECT so.Name as VendorName, SUM(spt.Sum) as TotalPaid, COUNT(spt.ID) as PaymentCount FROM dbo.SupplyPaymentTask spt JOIN dbo.SupplyOrder o ON spt.SupplyOrderID = o.ID JOIN dbo.SupplyOrganizationAgreement soa ON o.SupplyOrganizationAgreementID = soa.ID JOIN dbo.SupplyOrganization so ON soa.SupplyOrganizationID = so.ID WHERE spt.Deleted = 0 GROUP BY so.ID, so.Name ORDER BY TotalPaid DESC",
                "tables_used": ["SupplyPaymentTask", "SupplyOrder", "SupplyOrganizationAgreement", "SupplyOrganization"]
            },
            {
                "id": "supply_order_by_storage_008",
                "category": "list",
                "complexity": "medium",
                "question_en": "Supply orders by warehouse/storage",
                "question_uk": "Замовлення постачальникам по складах",
                "variations_en": ["orders per warehouse", "supply by storage"],
                "variations_uk": ["закупівлі по складах", "замовлення на склад"],
                "sql": "SELECT s.Name as StorageName, COUNT(o.ID) as OrderCount, SUM(o.TotalSum) as TotalAmount FROM dbo.SupplyOrder o JOIN dbo.Storage s ON o.StorageID = s.ID WHERE o.Deleted = 0 GROUP BY s.ID, s.Name ORDER BY TotalAmount DESC",
                "tables_used": ["SupplyOrder", "Storage"]
            },
        ])

        return examples

    def generate_financial_examples(self) -> List[Dict[str, Any]]:
        """Generate examples for financial/payment domain.

        Returns:
            List of financial example dictionaries
        """
        examples = []

        examples.extend([
            {
                "id": "income_payments_001",
                "category": "list",
                "complexity": "simple",
                "question_en": "List incoming payments",
                "question_uk": "Список вхідних платежів",
                "variations_en": ["show income payments", "customer payments", "received payments"],
                "variations_uk": ["вхідні платежі", "надходження", "платежі від клієнтів"],
                "sql": "SELECT TOP 100 ID, Created, Sum, Status FROM dbo.IncomePaymentOrder WHERE Deleted = 0 ORDER BY Created DESC",
                "tables_used": ["IncomePaymentOrder"]
            },
            {
                "id": "outcome_payments_002",
                "category": "list",
                "complexity": "simple",
                "question_en": "List outgoing payments",
                "question_uk": "Список вихідних платежів",
                "variations_en": ["show outcome payments", "vendor payments", "sent payments"],
                "variations_uk": ["вихідні платежі", "витрати", "платежі постачальникам"],
                "sql": "SELECT TOP 100 ID, Created, Sum, Status FROM dbo.OutcomePaymentOrder WHERE Deleted = 0 ORDER BY Created DESC",
                "tables_used": ["OutcomePaymentOrder"]
            },
            {
                "id": "income_by_client_003",
                "category": "aggregation",
                "complexity": "medium",
                "question_en": "Total payments received by client",
                "question_uk": "Сума платежів по клієнтах",
                "variations_en": ["client payment totals", "how much each client paid"],
                "variations_uk": ["скільки заплатив клієнт", "платежі клієнтів"],
                "sql": "SELECT c.Name as ClientName, SUM(ip.Sum) as TotalPaid, COUNT(ip.ID) as PaymentCount FROM dbo.IncomePaymentOrder ip JOIN dbo.ClientAgreement ca ON ip.ClientAgreementID = ca.ID JOIN dbo.Client c ON ca.ClientID = c.ID WHERE ip.Deleted = 0 GROUP BY c.ID, c.Name ORDER BY TotalPaid DESC",
                "tables_used": ["IncomePaymentOrder", "ClientAgreement", "Client"]
            },
            {
                "id": "currency_list_004",
                "category": "list",
                "complexity": "simple",
                "question_en": "List all currencies",
                "question_uk": "Список валют",
                "variations_en": ["show currencies", "available currencies", "currency codes"],
                "variations_uk": ["валюти", "список валют", "доступні валюти"],
                "sql": "SELECT ID, Name, Code FROM dbo.Currency WHERE Deleted = 0 ORDER BY Name",
                "tables_used": ["Currency"]
            },
            {
                "id": "payments_by_currency_005",
                "category": "aggregation",
                "complexity": "medium",
                "question_en": "Total payments by currency",
                "question_uk": "Сума платежів по валютах",
                "variations_en": ["payment totals per currency", "amounts by currency"],
                "variations_uk": ["платежі в різних валютах", "суми по валютах"],
                "sql": "SELECT cur.Name as CurrencyName, SUM(ip.Sum) as TotalAmount FROM dbo.IncomePaymentOrder ip JOIN dbo.Currency cur ON ip.CurrencyID = cur.ID WHERE ip.Deleted = 0 GROUP BY cur.ID, cur.Name ORDER BY TotalAmount DESC",
                "tables_used": ["IncomePaymentOrder", "Currency"]
            },
            {
                "id": "taxfree_list_006",
                "category": "list",
                "complexity": "simple",
                "question_en": "List tax-free operations",
                "question_uk": "Список безподаткових операцій",
                "variations_en": ["show tax free", "tax exempt transactions"],
                "variations_uk": ["такс фрі", "безподаткові", "операції без податку"],
                "sql": "SELECT TOP 100 ID, Created, Sum FROM dbo.TaxFree WHERE Deleted = 0 ORDER BY Created DESC",
                "tables_used": ["TaxFree"]
            },
            {
                "id": "agreement_list_007",
                "category": "list",
                "complexity": "simple",
                "question_en": "List all agreements",
                "question_uk": "Список угод",
                "variations_en": ["show agreements", "contracts list"],
                "variations_uk": ["угоди", "контракти", "договори"],
                "sql": "SELECT ID, Name, Created FROM dbo.Agreement WHERE Deleted = 0 ORDER BY Created DESC",
                "tables_used": ["Agreement"]
            },
            {
                "id": "debt_by_client_008",
                "category": "aggregation",
                "complexity": "medium",
                "question_en": "Client debts total",
                "question_uk": "Заборгованість клієнтів",
                "variations_en": ["outstanding debts", "client balances", "money owed by clients"],
                "variations_uk": ["борги клієнтів", "заборгованості", "скільки винні клієнти"],
                "sql": "SELECT c.Name as ClientName, SUM(d.Sum) as TotalDebt FROM dbo.Debt d JOIN dbo.ClientAgreement ca ON d.ClientAgreementID = ca.ID JOIN dbo.Client c ON ca.ClientID = c.ID WHERE d.Deleted = 0 GROUP BY c.ID, c.Name HAVING SUM(d.Sum) > 0 ORDER BY TotalDebt DESC",
                "tables_used": ["Debt", "ClientAgreement", "Client"]
            },
        ])

        return examples

    def generate_multi_join_examples(self) -> List[Dict[str, Any]]:
        """Generate complex multi-table JOIN examples.

        Returns:
            List of complex join example dictionaries
        """
        examples = []

        # Client -> Order chain (CRITICAL - the famous incorrect join issue)
        examples.extend([
            {
                "id": "client_orders_chain_001",
                "category": "join_pattern",
                "complexity": "critical",
                "question_en": "Get all orders for a client",
                "question_uk": "Отримати всі замовлення клієнта",
                "variations_en": ["client orders", "orders by client", "customer orders"],
                "variations_uk": ["замовлення клієнта", "замовлення по клієнту"],
                "sql": "SELECT o.ID, o.Created, o.TotalSum, c.Name as ClientName FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID WHERE c.Deleted = 0 AND ca.Deleted = 0 AND o.Deleted = 0 ORDER BY o.Created DESC",
                "tables_used": ["Client", "ClientAgreement", "Order"],
                "critical_note": "NEVER use Order.ClientID - it does not exist! Always use ClientAgreement as bridge."
            },
            {
                "id": "client_order_items_chain_002",
                "category": "join_pattern",
                "complexity": "complex",
                "question_en": "Products ordered by each client",
                "question_uk": "Товари замовлені кожним клієнтом",
                "variations_en": ["client product orders", "what did clients buy"],
                "variations_uk": ["що замовляли клієнти", "товари клієнтів"],
                "sql": "SELECT c.Name as ClientName, p.Name as ProductName, SUM(oi.Qty) as TotalQty FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID JOIN dbo.OrderItem oi ON oi.OrderID = o.ID JOIN dbo.Product p ON oi.ProductID = p.ID WHERE c.Deleted = 0 AND o.Deleted = 0 GROUP BY c.ID, c.Name, p.ID, p.Name ORDER BY c.Name, TotalQty DESC",
                "tables_used": ["Client", "ClientAgreement", "Order", "OrderItem", "Product"],
                "critical_note": "5-table chain: Client -> ClientAgreement -> Order -> OrderItem -> Product"
            },
            {
                "id": "client_sales_chain_003",
                "category": "join_pattern",
                "complexity": "complex",
                "question_en": "Client sales with order details",
                "question_uk": "Продажі клієнтів з деталями замовлень",
                "variations_en": ["sales by client", "client purchase history"],
                "variations_uk": ["продажі клієнтів", "історія покупок"],
                "sql": "SELECT c.Name as ClientName, s.SaleDate, s.TotalSum, o.Created as OrderDate FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.Sale s ON s.ClientAgreementID = ca.ID JOIN dbo.[Order] o ON s.OrderID = o.ID WHERE c.Deleted = 0 AND s.Deleted = 0 ORDER BY s.SaleDate DESC",
                "tables_used": ["Client", "ClientAgreement", "Sale", "Order"]
            },
            {
                "id": "product_stock_by_storage_004",
                "category": "list",
                "complexity": "medium",
                "question_en": "Product availability by warehouse",
                "question_uk": "Залишки товарів по складах",
                "variations_en": ["stock by warehouse", "inventory per storage"],
                "variations_uk": ["залишки по складах", "товари на складах"],
                "sql": "SELECT p.Name as ProductName, s.Name as StorageName, pa.Amount as Stock FROM dbo.Product p JOIN dbo.ProductAvailability pa ON pa.ProductID = p.ID JOIN dbo.Storage s ON pa.StorageID = s.ID WHERE p.Deleted = 0 AND pa.Amount > 0 ORDER BY p.Name, s.Name",
                "tables_used": ["Product", "ProductAvailability", "Storage"]
            },
            {
                "id": "order_region_chain_005",
                "category": "aggregation",
                "complexity": "complex",
                "question_en": "Orders by client region",
                "question_uk": "Замовлення по регіонах клієнтів",
                "variations_en": ["orders per region", "regional order distribution"],
                "variations_uk": ["замовлення по регіонах", "розподіл по регіонах"],
                "sql": "SELECT r.Name as RegionName, COUNT(o.ID) as OrderCount, SUM(o.TotalSum) as TotalAmount FROM dbo.Region r JOIN dbo.Client c ON c.RegionID = r.ID JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID WHERE r.Deleted = 0 AND o.Deleted = 0 GROUP BY r.ID, r.Name ORDER BY TotalAmount DESC",
                "tables_used": ["Region", "Client", "ClientAgreement", "Order"]
            },
            {
                "id": "sale_product_category_006",
                "category": "aggregation",
                "complexity": "complex",
                "question_en": "Sales by product category",
                "question_uk": "Продажі по категоріях товарів",
                "variations_en": ["category sales", "revenue by product type"],
                "variations_uk": ["продажі по категоріях", "виручка по типах"],
                "sql": "SELECT cat.Name as CategoryName, SUM(oi.Qty * oi.PricePerItem) as TotalSales FROM dbo.Category cat JOIN dbo.Product p ON p.CategoryID = cat.ID JOIN dbo.OrderItem oi ON oi.ProductID = p.ID JOIN dbo.[Order] o ON oi.OrderID = o.ID WHERE cat.Deleted = 0 AND o.Deleted = 0 GROUP BY cat.ID, cat.Name ORDER BY TotalSales DESC",
                "tables_used": ["Category", "Product", "OrderItem", "Order"]
            },
        ])

        return examples

    def generate_template_file(
        self,
        domain: str,
        description: str,
        tables: List[str],
        examples: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """Generate a template JSON file.

        Args:
            domain: Domain name
            description: Domain description
            tables: List of main tables
            examples: List of examples
            output_path: Output file path
        """
        template = {
            "domain": domain,
            "description": description,
            "tables": tables,
            "examples": examples
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

        logger.info(f"Generated template: {output_path} ({len(examples)} examples)")


def main():
    parser = argparse.ArgumentParser(description="Generate SQL training examples")
    parser.add_argument("--analyze", action="store_true", help="Analyze current coverage")
    parser.add_argument("--generate-supply", action="store_true", help="Generate supply chain examples")
    parser.add_argument("--generate-financial", action="store_true", help="Generate financial examples")
    parser.add_argument("--generate-multi-join", action="store_true", help="Generate multi-table join examples")
    parser.add_argument("--generate-all", action="store_true", help="Generate all example types")
    parser.add_argument("--output-dir", type=str, default="training_data/templates", help="Output directory")
    args = parser.parse_args()

    generator = ExampleGenerator()
    templates_dir = Path(__file__).parent / args.output_dir

    if args.analyze:
        print("\n" + "=" * 60)
        print("COVERAGE ANALYSIS")
        print("=" * 60)
        analysis = generator.analyze_coverage(templates_dir)

        print(f"\nTotal tables: {analysis['total_tables']}")
        print(f"Tables with data: {analysis['tables_with_data']}")
        print(f"Tables with FKs: {analysis['tables_with_fks']}")
        print(f"Covered tables: {analysis['covered_tables']}")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Avg tables per example: {analysis['avg_tables_per_example']:.2f}")

        print(f"\nComplexity distribution:")
        for tables_count, count in sorted(analysis['complexity_distribution'].items()):
            pct = count / analysis['total_examples'] * 100
            print(f"  {tables_count}-table queries: {count} ({pct:.1f}%)")

        print(f"\nUncovered tables with FKs ({len(analysis['uncovered_with_fks'])}):")
        for table in analysis['uncovered_with_fks'][:20]:
            print(f"  - {table}")

        return

    if args.generate_supply or args.generate_all:
        examples = generator.generate_supply_chain_examples()
        output_path = templates_dir / "supply_chain.json"
        generator.generate_template_file(
            domain="supply_chain",
            description="Supply orders, vendors, and payment examples",
            tables=["SupplyOrder", "SupplyOrganization", "SupplyPaymentTask", "SupplyOrganizationAgreement"],
            examples=examples,
            output_path=output_path
        )
        print(f"Generated {len(examples)} supply chain examples")

    if args.generate_financial or args.generate_all:
        examples = generator.generate_financial_examples()
        output_path = templates_dir / "financial_extended.json"
        generator.generate_template_file(
            domain="financial_extended",
            description="Payments, currency, tax-free, and debt examples",
            tables=["IncomePaymentOrder", "OutcomePaymentOrder", "Currency", "TaxFree", "Agreement", "Debt"],
            examples=examples,
            output_path=output_path
        )
        print(f"Generated {len(examples)} financial examples")

    if args.generate_multi_join or args.generate_all:
        examples = generator.generate_multi_join_examples()
        output_path = templates_dir / "multi_table_joins.json"
        generator.generate_template_file(
            domain="multi_table",
            description="Complex 3-5 table JOIN pattern examples",
            tables=["Client", "ClientAgreement", "Order", "OrderItem", "Product", "Sale", "Region"],
            examples=examples,
            output_path=output_path
        )
        print(f"Generated {len(examples)} multi-table join examples")

    if not any([args.analyze, args.generate_supply, args.generate_financial, args.generate_multi_join, args.generate_all]):
        parser.print_help()


if __name__ == "__main__":
    main()
