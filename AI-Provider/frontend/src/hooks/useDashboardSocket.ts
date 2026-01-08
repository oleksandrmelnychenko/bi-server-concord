import { useState, useEffect, useCallback, useRef } from 'react';

const DASHBOARD_WS_URL = import.meta.env.VITE_DASHBOARD_WS_URL || 'ws://localhost:8200/ws';

export interface HistoryPoint {
  time: string;
  value: number;
}

export interface InventoryData {
  total: number;
  change: number;
  change_today: number;
  history: HistoryPoint[];
}

export interface RevenueData {
  total: number;
  change: number;
  orders_today: number;
  history: HistoryPoint[];
}

export interface DebtData {
  total: number;
  change: number;
  change_today: number;
  history: HistoryPoint[];
}

export interface ManagerData {
  id: number;
  name: string;
  orders_today: number;
  revenue_today: number;
  change: number;
  change_today: number;
  history: HistoryPoint[];
}

export interface StorageData {
  id: number;
  name: string;
  is_defective: boolean;
  is_ecommerce: boolean;
  total_stock: number;
  change: number;
  change_today: number;
  history: HistoryPoint[];
}

export interface DashboardData {
  inventory: InventoryData | null;
  revenue: RevenueData | null;
  debt: DebtData | null;
  managers: ManagerData[];
  storages: StorageData[];
  top_manager_month_id: number | null;
}

export interface UseDashboardSocketReturn {
  data: DashboardData;
  connected: boolean;
  error: string | null;
  reconnect: () => void;
}

export const useDashboardSocket = (): UseDashboardSocketReturn => {
  const [data, setData] = useState<DashboardData>({
    inventory: null,
    revenue: null,
    debt: null,
    managers: [],
    storages: [],
    top_manager_month_id: null,
  });
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isUnmountingRef = useRef(false);
  const isConnectingRef = useRef(false);
  const maxReconnectAttempts = 10;

  const connect = useCallback(() => {
    // Don't connect if unmounting or already connecting/connected
    if (isUnmountingRef.current) return;
    if (isConnectingRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    isConnectingRef.current = true;

    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.onclose = null; // Prevent onclose from firing
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      const ws = new WebSocket(DASHBOARD_WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        isConnectingRef.current = false;
        if (isUnmountingRef.current) return;
        setConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        if (isUnmountingRef.current) return;
        try {
          const message = JSON.parse(event.data);

          if (message.type === 'dashboard_update' || message.type === 'initial_state') {
            const { inventory, revenue, debt, managers, storages, top_manager_month_id } = message.data;
            setData({
              inventory: inventory || null,
              revenue: revenue || null,
              debt: debt || null,
              managers: managers || [],
              storages: storages || [],
              top_manager_month_id: top_manager_month_id || null,
            });
          } else if (message.type === 'pong') {
            // Ping-pong for keep-alive
          }
        } catch (e) {
          console.error('[Dashboard] Failed to parse message:', e);
        }
      };

      ws.onerror = () => {
        isConnectingRef.current = false;
        if (isUnmountingRef.current) return;
        setError('Connection error');
      };

      ws.onclose = () => {
        isConnectingRef.current = false;
        if (isUnmountingRef.current) return;

        setConnected(false);

        // Auto-reconnect with exponential backoff
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          reconnectAttemptsRef.current += 1;

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          setError('Connection lost');
        }
      };
    } catch (e) {
      isConnectingRef.current = false;
      setError('Failed to connect');
    }
  }, []);

  const reconnect = useCallback(() => {
    isConnectingRef.current = false;
    reconnectAttemptsRef.current = 0;
    setError(null);
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    connect();
  }, [connect]);

  useEffect(() => {
    isUnmountingRef.current = false;
    connect();

    // Ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 30000);

    return () => {
      isUnmountingRef.current = true;
      isConnectingRef.current = false;
      clearInterval(pingInterval);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - only run once on mount

  return { data, connected, error, reconnect };
};

export default useDashboardSocket;
