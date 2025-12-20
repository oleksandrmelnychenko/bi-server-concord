import { useState, useEffect, useCallback } from 'react';
import type { ApiStatus } from '../types';
import { checkApiStatus } from '../services/api';

export function useApiStatus(intervalMs = 30000) {
  const [status, setStatus] = useState<ApiStatus>({
    online: false,
    documents: null,
  });

  const refresh = useCallback(async () => {
    const newStatus = await checkApiStatus();
    setStatus(newStatus);
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, intervalMs);
    return () => clearInterval(interval);
  }, [refresh, intervalMs]);

  return { status, refresh };
}
