#!/usr/bin/env python3
"""
Build ANN item-item neighbors from MSSQL interactions and save to JSON.

Requirements:
- pymssql
- hnswlib

Outputs:
- JSON file: {product_id: [{"product_id": int, "score": float}, ...]}
  controlled by ANN_OUTPUT_PATH (default: ann_neighbors.json)

Environment:
  MSSQL_HOST, MSSQL_PORT, MSSQL_DATABASE, MSSQL_USER, MSSQL_PASSWORD
  ANN_OUTPUT_PATH (optional)
  ANN_METRIC (cosine | l2 | ip) default cosine
  ANN_M (graph degree, default 32)
  ANN_EF_CONSTRUCT (default 200)
  ANN_K (neighbors per item, default 200)
  ANN_MIN_INTERACTIONS (default 2)
  ANN_LOOKBACK_START (default: 3 years back from today)
"""
import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import pyodbc
import hnswlib
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DB_CONFIG = {
    "server": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": int(os.getenv("MSSQL_PORT", "1433")),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD"),  # Required - no default for security
}

ANN_OUTPUT_PATH = os.getenv("ANN_OUTPUT_PATH", "ann_neighbors.json")
ANN_METRIC = os.getenv("ANN_METRIC", "cosine")
ANN_M = int(os.getenv("ANN_M", "32"))
ANN_EF_CONSTRUCT = int(os.getenv("ANN_EF_CONSTRUCT", "200"))
ANN_K = int(os.getenv("ANN_K", "200"))
ANN_MIN_INTERACTIONS = int(os.getenv("ANN_MIN_INTERACTIONS", "2"))
# Default: 3 years back from today (safe for leap years)
# Using timedelta avoids Feb 29 crash with datetime.replace(year=...)
_default_lookback = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
ANN_LOOKBACK_START = os.getenv("ANN_LOOKBACK_START", _default_lookback)


def fetch_interactions(as_of_date: str):
    query = """
    SELECT
        ca.ClientID,
        oi.ProductID,
        SUM(oi.Qty) as total_qty,
        MAX(o.Created) as last_ts
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID AND o.Deleted = 0
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID AND oi.Deleted = 0
    INNER JOIN dbo.Product p ON p.ID = oi.ProductID AND p.Deleted = 0 AND p.IsForSale = 1
    WHERE o.Created >= ? AND o.Created < ?
    GROUP BY ca.ClientID, oi.ProductID;
    """
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={DB_CONFIG['server']},{DB_CONFIG['port']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['user']};"
        f"PWD={DB_CONFIG['password']};"
        f"TrustServerCertificate=yes;"
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(query, (ANN_LOOKBACK_START, as_of_date))

    # Convert rows to dict format
    columns = [column[0] for column in cursor.description]
    rows = []
    for row in cursor.fetchall():
        rows.append(dict(zip(columns, row)))

    cursor.close()
    conn.close()
    return rows


def build_item_vectors(rows, decay_days=180.0):
    # Build customer -> {product: weight}
    cust_products = defaultdict(dict)
    now = datetime.utcnow()
    for row in rows:
        pid = int(row["ProductID"])
        cid = int(row["ClientID"])
        qty = float(row["total_qty"] or 0.0)
        last_ts = row["last_ts"]
        days = max(0.0, (now - last_ts).days)
        weight = np.log1p(qty) * np.exp(-days / decay_days)
        cust_products[cid][pid] = cust_products[cid].get(pid, 0.0) + weight

    # Build item -> sparse vector (as dict)
    item_dict = defaultdict(dict)
    for cid, items in cust_products.items():
        norm = np.linalg.norm(list(items.values())) or 1.0
        for pid, w in items.items():
            item_dict[pid][cid] = w / norm

    # Convert to dense matrix with index mapping
    item_ids = list(item_dict.keys())
    id_to_idx = {pid: idx for idx, pid in enumerate(item_ids)}
    vectors = []
    for pid in item_ids:
        vec = item_dict[pid]
        vectors.append((pid, vec))
    return item_ids, vectors


def to_dense(vectors):
    # vectors: list of (pid, {cid: weight})
    # Build a customer index
    cust_set = set()
    for _, vec in vectors:
        cust_set.update(vec.keys())
    cust_ids = list(cust_set)
    cust_to_idx = {cid: idx for idx, cid in enumerate(cust_ids)}
    dim = len(cust_ids)
    data = np.zeros((len(vectors), dim), dtype=np.float32)
    for i, (_, vec) in enumerate(vectors):
        for cid, w in vec.items():
            data[i, cust_to_idx[cid]] = w
    return data, cust_ids


def build_ann(item_ids, data):
    dim = data.shape[1]
    p = hnswlib.Index(space=ANN_METRIC, dim=dim)
    p.init_index(max_elements=len(item_ids), ef_construction=ANN_EF_CONSTRUCT, M=ANN_M)
    p.add_items(data, item_ids)
    p.set_ef(max(ANN_K * 2, 100))
    return p


def main():
    as_of = datetime.utcnow().strftime("%Y-%m-%d")
    logger.info(f"Building ANN neighbors as of {as_of} (lookback from {ANN_LOOKBACK_START})")
    rows = fetch_interactions(as_of)
    logger.info(f"Fetched {len(rows):,} interactions")

    item_ids, vectors = build_item_vectors(rows)
    logger.info(f"Built normalized vectors for {len(item_ids):,} items")
    data, _ = to_dense(vectors)
    logger.info(f"Dense matrix shape: {data.shape}")

    index = build_ann(item_ids, data)
    neighbors = {}
    labels, dists = index.knn_query(data, k=min(ANN_K + 1, len(item_ids)))
    for i, pid in enumerate(item_ids):
        neigh = []
        for lbl, dist in zip(labels[i], dists[i]):
            if lbl == pid:
                continue
            score = 1.0 - dist if ANN_METRIC == "cosine" else -dist
            neigh.append({"product_id": int(lbl), "score": float(score)})
            if len(neigh) >= ANN_K:
                break
        neighbors[pid] = neigh

    with open(ANN_OUTPUT_PATH, "w") as f:
        json.dump(neighbors, f)
    logger.info(f"Saved neighbors for {len(neighbors)} items to {ANN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
