#!/usr/bin/env python3
"""
Real-Time Business Dashboard WebSocket Server

Provides simulated real-time updates for:
- Inventory levels (product stock)
- Revenue/Income (sales)
- Debt changes (payments vs new debt)

Uses REAL database baseline, with REALISTIC random mutations for visualization.

Run: uvicorn api.dashboard_server:app --host 0.0.0.0 --port 8200
"""

import os
import sys
import json
import random
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


class DashboardSimulator:
    """Simulates realistic business metrics changes."""

    def __init__(self):
        self.inventory_total: float = 0
        self.revenue_today: float = 0
        self.debt_total: float = 0
        self.orders_today: int = 0

        # History for charts (last 24 data points)
        self.inventory_history: List[Dict] = []
        self.revenue_history: List[Dict] = []
        self.debt_history: List[Dict] = []

        self.start_of_day_inventory: float = 0
        self.start_of_day_debt: float = 0
        self.initialized = False

        # NEW: Manager sales tracking
        self.managers: List[Dict] = []  # [{id, name, orders_today, revenue_today}]
        self.managers_start: Dict[int, Dict] = {}  # Start of day values per manager
        self.managers_history: Dict[int, List[Dict]] = {}  # History per manager

        # NEW: Storage inventory tracking
        self.storages: List[Dict] = []  # [{id, name, is_defective, is_ecommerce, total_stock}]
        self.storages_start: Dict[int, float] = {}  # Start of day stock per storage
        self.storages_history: Dict[int, List[Dict]] = {}  # History per storage

        # TOP manager of the month
        self.top_manager_month_id: int = None

    def get_activity_multiplier(self) -> float:
        """Return activity multiplier based on time of day."""
        hour = datetime.now().hour

        # Weekend - minimal activity
        if datetime.now().weekday() >= 5:
            return 0.2

        # Business hours patterns
        if 8 <= hour < 10:  # Morning ramp-up
            return 0.7
        elif 10 <= hour < 12:  # Peak morning
            return 1.2
        elif 12 <= hour < 14:  # Lunch rush
            return 1.5
        elif 14 <= hour < 17:  # Afternoon
            return 1.0
        elif 17 <= hour < 19:  # End of day
            return 0.6
        else:  # Off hours
            return 0.1

    def load_baseline_from_db(self):
        """Load real baseline values from database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get total inventory
            cursor.execute("""
                SELECT ISNULL(SUM(pa.Amount), 0) as total_stock
                FROM dbo.ProductAvailability pa
                JOIN dbo.Storage s ON pa.StorageID = s.ID
                WHERE pa.Deleted = 0 AND s.Deleted = 0 AND s.ForDefective = 0
            """)
            row = cursor.fetchone()
            self.inventory_total = float(row[0]) if row else 100000
            self.start_of_day_inventory = self.inventory_total

            # Get today's revenue
            cursor.execute("""
                SELECT ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as revenue,
                       COUNT(DISTINCT o.ID) as orders
                FROM dbo.[Order] o
                JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                WHERE CAST(o.Created AS DATE) = CAST(GETDATE() AS DATE)
                  AND o.Deleted = 0
            """)
            row = cursor.fetchone()
            if row:
                self.revenue_today = float(row[0] or 0)
                self.orders_today = int(row[1] or 0)

            # Get total unpaid debt
            cursor.execute("""
                SELECT ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as total_debt
                FROM dbo.[Order] o
                JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                LEFT JOIN dbo.Sale s ON s.OrderID = o.ID AND s.Deleted = 0
                WHERE o.Deleted = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM dbo.IncomePaymentOrderSale ipos
                      WHERE ipos.SaleID = s.ID
                  )
            """)
            row = cursor.fetchone()
            self.debt_total = float(row[0]) if row else 500000
            self.start_of_day_debt = self.debt_total

            # NEW: Load managers with sales (users who created orders)
            try:
                cursor.execute("""
                    SELECT TOP 10
                        o.UserID,
                        ISNULL(u.LastName + ' ' + u.FirstName, 'Manager ' + CAST(o.UserID AS VARCHAR)) as ManagerName,
                        COUNT(DISTINCT CASE WHEN CAST(o.Created AS DATE) = CAST(GETDATE() AS DATE) THEN o.ID END) as OrdersToday,
                        ISNULL(SUM(CASE WHEN CAST(o.Created AS DATE) = CAST(GETDATE() AS DATE) THEN oi.Qty * oi.PricePerItem ELSE 0 END), 0) as RevenueToday
                    FROM dbo.[Order] o
                    LEFT JOIN dbo.[User] u ON u.ID = o.UserID AND u.Deleted = 0
                    LEFT JOIN dbo.OrderItem oi ON o.ID = oi.OrderID AND oi.Deleted = 0
                    WHERE o.Deleted = 0 AND o.UserID IS NOT NULL
                    GROUP BY o.UserID, u.LastName, u.FirstName
                    ORDER BY COUNT(DISTINCT o.ID) DESC
                """)
                rows = cursor.fetchall()
                self.managers = []
                for row in rows:
                    manager_id = int(row[0])
                    manager = {
                        "id": manager_id,
                        "name": row[1] or f"Manager {manager_id}",
                        "orders_today": int(row[2] or 0),
                        "revenue_today": float(row[3] or 0)
                    }
                    self.managers.append(manager)
                    self.managers_start[manager_id] = {
                        "orders": manager["orders_today"],
                        "revenue": manager["revenue_today"]
                    }
                    self.managers_history[manager_id] = []
                logger.info(f"Loaded {len(self.managers)} managers: {[m['name'] for m in self.managers]}")

                # Get TOP manager of the month (most orders this month)
                cursor.execute("""
                    SELECT TOP 1 o.UserID
                    FROM dbo.[Order] o
                    WHERE o.Deleted = 0
                      AND o.UserID IS NOT NULL
                      AND MONTH(o.Created) = MONTH(GETDATE())
                      AND YEAR(o.Created) = YEAR(GETDATE())
                    GROUP BY o.UserID
                    ORDER BY COUNT(DISTINCT o.ID) DESC
                """)
                top_row = cursor.fetchone()
                if top_row:
                    self.top_manager_month_id = int(top_row[0])
                    logger.info(f"TOP manager of month: {self.top_manager_month_id}")

            except Exception as e:
                logger.error(f"Failed to load managers: {e}")
                self.managers = []

            # NEW: Load storages with inventory
            cursor.execute("""
                SELECT
                    s.ID,
                    s.Name as StorageName,
                    ISNULL(s.ForDefective, 0) as ForDefective,
                    ISNULL(s.ForEcommerce, 0) as ForEcommerce,
                    ISNULL(SUM(pa.Amount), 0) as TotalStock
                FROM dbo.Storage s
                LEFT JOIN dbo.ProductAvailability pa ON pa.StorageID = s.ID
                    AND pa.Deleted = 0
                WHERE s.Deleted = 0
                GROUP BY s.ID, s.Name, s.ForDefective, s.ForEcommerce
                HAVING ISNULL(SUM(pa.Amount), 0) > 0
                ORDER BY TotalStock DESC
            """)
            rows = cursor.fetchall()
            self.storages = []
            for row in rows[:15]:  # Top 15 storages
                storage_id = int(row[0])
                storage = {
                    "id": storage_id,
                    "name": row[1] or f"Storage {storage_id}",
                    "is_defective": bool(row[2]),
                    "is_ecommerce": bool(row[3]),
                    "total_stock": float(row[4] or 0)
                }
                self.storages.append(storage)
                self.storages_start[storage_id] = storage["total_stock"]
                self.storages_history[storage_id] = []

            logger.info(f"Loaded {len(self.storages)} storages")

            conn.close()
            self.initialized = True

            logger.info(f"Loaded baseline: Inventory={self.inventory_total:.0f}, "
                       f"Revenue={self.revenue_today:.0f}, Debt={self.debt_total:.0f}, "
                       f"Managers={len(self.managers)}, Storages={len(self.storages)}")

        except Exception as e:
            logger.error(f"Failed to load baseline from DB: {e}")
            # No dummy data - keep empty arrays, will retry on next connection
            self.inventory_total = 0
            self.revenue_today = 0
            self.debt_total = 0
            self.start_of_day_inventory = 0
            self.start_of_day_debt = 0
            self.managers = []
            self.storages = []
            self.initialized = True

    def generate_mutation(self) -> Dict[str, Any]:
        """Generate realistic random mutations."""
        multiplier = self.get_activity_multiplier()
        now = datetime.now()
        timestamp = now.isoformat()
        time_label = now.strftime("%H:%M:%S")

        # Inventory changes: more sales than restocks
        # Sales (negative): -1 to -100 units
        # Restocks (positive): occasional +50 to +200 units (10% chance)
        if random.random() < 0.1:  # 10% chance of restock
            inventory_change = random.randint(50, 200) * multiplier
        else:
            inventory_change = -random.randint(1, 50) * multiplier

        self.inventory_total = max(0, self.inventory_total + inventory_change)

        # Revenue: always positive (new sales)
        revenue_change = random.randint(500, 15000) * multiplier
        self.revenue_today += revenue_change

        # Occasionally add new order
        if random.random() < 0.3 * multiplier:
            self.orders_today += 1

        # Debt: payments (negative) vs new unpaid orders (positive)
        # 60% chance of payment, 40% chance of new debt
        if random.random() < 0.6:
            debt_change = -random.randint(1000, 10000) * multiplier  # Payment received
        else:
            debt_change = random.randint(500, 5000) * multiplier  # New unpaid order

        self.debt_total = max(0, self.debt_total + debt_change)

        # Update history (keep last 30 points)
        history_point_inv = {"time": time_label, "value": round(self.inventory_total)}
        history_point_rev = {"time": time_label, "value": round(self.revenue_today)}
        history_point_debt = {"time": time_label, "value": round(self.debt_total)}

        self.inventory_history.append(history_point_inv)
        self.revenue_history.append(history_point_rev)
        self.debt_history.append(history_point_debt)

        # Keep only last 30 points
        if len(self.inventory_history) > 30:
            self.inventory_history = self.inventory_history[-30:]
        if len(self.revenue_history) > 30:
            self.revenue_history = self.revenue_history[-30:]
        if len(self.debt_history) > 30:
            self.debt_history = self.debt_history[-30:]

        # Simulate manager sales mutations
        managers_data = []
        for manager in self.managers:
            manager_id = manager["id"]

            # Random chance of new order for this manager (weighted by multiplier)
            order_change = 0
            revenue_change_mgr = 0
            if random.random() < 0.25 * multiplier:
                order_change = random.randint(1, 3)
                revenue_change_mgr = order_change * random.randint(2000, 8000)
                manager["orders_today"] += order_change
                manager["revenue_today"] += revenue_change_mgr

            # Update manager history
            history_point = {"time": time_label, "value": manager["orders_today"]}
            if manager_id not in self.managers_history:
                self.managers_history[manager_id] = []
            self.managers_history[manager_id].append(history_point)
            if len(self.managers_history[manager_id]) > 30:
                self.managers_history[manager_id] = self.managers_history[manager_id][-30:]

            start_orders = self.managers_start.get(manager_id, {}).get("orders", 0)
            managers_data.append({
                "id": manager_id,
                "name": manager["name"],
                "orders_today": manager["orders_today"],
                "revenue_today": round(manager["revenue_today"]),
                "change": order_change,
                "change_today": manager["orders_today"] - start_orders,
                "history": self.managers_history[manager_id][-30:]
            })

        # Simulate storage inventory mutations
        storages_data = []
        for storage in self.storages:
            storage_id = storage["id"]

            # Different behavior for defective vs normal storage
            if storage["is_defective"]:
                # Defective storage only increases (items moved there)
                stock_change = random.randint(0, 5) * multiplier if random.random() < 0.1 else 0
            elif storage["is_ecommerce"]:
                # E-commerce has higher activity
                if random.random() < 0.15:  # 15% restock chance
                    stock_change = random.randint(20, 100) * multiplier
                else:
                    stock_change = -random.randint(5, 30) * multiplier
            else:
                # Regular storage
                if random.random() < 0.1:  # 10% restock
                    stock_change = random.randint(10, 50) * multiplier
                else:
                    stock_change = -random.randint(1, 20) * multiplier

            storage["total_stock"] = max(0, storage["total_stock"] + stock_change)

            # Update storage history
            history_point = {"time": time_label, "value": round(storage["total_stock"])}
            if storage_id not in self.storages_history:
                self.storages_history[storage_id] = []
            self.storages_history[storage_id].append(history_point)
            if len(self.storages_history[storage_id]) > 30:
                self.storages_history[storage_id] = self.storages_history[storage_id][-30:]

            start_stock = self.storages_start.get(storage_id, 0)
            storages_data.append({
                "id": storage_id,
                "name": storage["name"],
                "is_defective": storage["is_defective"],
                "is_ecommerce": storage["is_ecommerce"],
                "total_stock": round(storage["total_stock"]),
                "change": round(stock_change),
                "change_today": round(storage["total_stock"] - start_stock),
                "history": self.storages_history[storage_id][-30:]
            })

        return {
            "type": "dashboard_update",
            "timestamp": timestamp,
            "data": {
                "inventory": {
                    "total": round(self.inventory_total),
                    "change": round(inventory_change),
                    "change_today": round(self.inventory_total - self.start_of_day_inventory),
                    "history": self.inventory_history[-30:]
                },
                "revenue": {
                    "total": round(self.revenue_today),
                    "change": round(revenue_change),
                    "orders_today": self.orders_today,
                    "history": self.revenue_history[-30:]
                },
                "debt": {
                    "total": round(self.debt_total),
                    "change": round(debt_change),
                    "change_today": round(self.debt_total - self.start_of_day_debt),
                    "history": self.debt_history[-30:]
                },
                "managers": managers_data,
                "storages": storages_data,
                "top_manager_month_id": self.top_manager_month_id
            }
        }


# Global instances
manager = ConnectionManager()
simulator = DashboardSimulator()


async def broadcast_loop():
    """Background task that broadcasts updates every 3 seconds."""
    while True:
        await asyncio.sleep(3)

        if manager.active_connections:
            update = simulator.generate_mutation()
            await manager.broadcast(update)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("Starting Real-Time Dashboard WebSocket Server")
    logger.info("=" * 60)

    # Load baseline from database
    simulator.load_baseline_from_db()

    # Start broadcast loop
    broadcast_task = asyncio.create_task(broadcast_loop())

    logger.info("Dashboard server ready on ws://0.0.0.0:8200/ws")
    logger.info("=" * 60)

    yield

    # Cleanup
    broadcast_task.cancel()
    logger.info("Dashboard server shutting down")


app = FastAPI(
    title="Real-Time Dashboard WebSocket Server",
    description="Streams simulated business metrics (inventory, revenue, debt)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Real-Time Dashboard WebSocket Server",
        "status": "running",
        "connections": len(manager.active_connections),
        "websocket_url": "ws://localhost:8200/ws",
        "metrics": {
            "inventory": round(simulator.inventory_total),
            "revenue_today": round(simulator.revenue_today),
            "debt_total": round(simulator.debt_total),
            "orders_today": simulator.orders_today
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await manager.connect(websocket)

    # Send initial state
    initial_update = simulator.generate_mutation()
    initial_update["type"] = "initial_state"
    await websocket.send_json(initial_update)

    try:
        while True:
            # Keep connection alive, handle incoming messages if any
            data = await websocket.receive_text()
            # Could handle commands here (pause, resume, etc.)
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8200,
        reload=True,
        log_level="info"
    )
