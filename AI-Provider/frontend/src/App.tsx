import { useState, useCallback, useRef, useEffect } from 'react';
import { WelcomeMessage, Language } from './components/WelcomeMessage';
import { Message, LoadingMessage } from './components/Message';
import { ChatInput } from './components/ChatInput';
import { ProductForecastScreen } from './components/ProductForecastScreen';
import { useApiStatus } from './hooks/useApiStatus';
import { useQueryHandler } from './hooks/useQueryHandler';
import type { Message as MessageType } from './types';
import logoSvg from './assets/logo.svg';
import {
  fetchFullRecommendations,
  fetchForecastForProduct,
  fetchProductsByIds,
  fetchClientById,
  fetchClientScore,
  type FullRecommendationResponse,
  type RecommendationCharts,
  type RecommendationProof,
  type ProductForecastResponse,
  type ProductCharts,
  type ProductProof,
  type ClientScoreData,
} from './services/api';
import { ClientScorePanel } from './components/ClientScorePanel';
import { LiveDashboard } from './components/LiveDashboard';
import { StorageListPanel } from './components/StorageListPanel';
import { ManagerListPanel } from './components/ManagerListPanel';
import { OrderRecommendationsPanel } from './components/OrderRecommendationsPanel';

const isAbortError = (error: unknown): boolean => {
  if (error instanceof DOMException) {
    return error.name === 'AbortError';
  }
  return (error as Error | undefined)?.name === 'AbortError';
};

function App() {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState<Language>('uk');
  const [sidebarOpen, setSidebarOpen] = useState(false);
const [insight, setInsight] = useState<{
    open: boolean;
    loading: boolean;
    clientId?: number | string;
    clientName?: string | null;
    segment?: string | null;
    productId?: number | string;
    productName?: string | null;
    vendorCode?: string | null;
    category?: string | null;
    recommendations: { product_id: number; score: number; rank: number; source: string }[];
    products: Record<string, unknown>[];
    forecast: ProductForecastResponse | null;
    charts?: RecommendationCharts | null;
    proof?: RecommendationProof | null;
    productCharts?: ProductCharts | null;
    productProof?: ProductProof | null;
    recError?: string | null;
    productError?: string | null;
    forecastError?: string | null;
    error?: string | null;
  }>({
    open: false,
    loading: false,
    clientName: null,
    segment: null,
    productName: null,
    vendorCode: null,
    category: null,
    recommendations: [],
    products: [],
    forecast: null,
    charts: null,
    proof: null,
    productCharts: null,
    productProof: null,
    recError: null,
    productError: null,
    forecastError: null,
    error: null,
  });
  const [chartTab, setChartTab] = useState<'6m' | '1y'>('6m');
  const [productChartTab, setProductChartTab] = useState<'6m' | '1y'>('6m');
  const [productForecastScreen, setProductForecastScreen] = useState<{
    isOpen: boolean;
    loading: boolean;
    error?: string | null;
    productId?: number;
    productName?: string | null;
    vendorCode?: string | null;
    category?: string | null;
    forecast?: ProductForecastResponse | null;
    charts?: ProductCharts | null;
    proof?: ProductProof | null;
  }>({
    isOpen: false,
    loading: false,
    error: null,
    forecast: null,
    charts: null,
    proof: null,
  });
  const [scorePanel, setScorePanel] = useState<{
    open: boolean;
    loading: boolean;
    error?: string | null;
    clientId?: number;
    clientName?: string | null;
    scoreData?: ClientScoreData | null;
  }>({
    open: false,
    loading: false,
    error: null,
    scoreData: null,
  });
  const [showDashboard, setShowDashboard] = useState(false);
  const [showStoragePanel, setShowStoragePanel] = useState(false);
  const [showManagerPanel, setShowManagerPanel] = useState(false);
  const [showOrderRecommendations, setShowOrderRecommendations] = useState(false);
  const scorePanelController = useRef<AbortController | null>(null);
  const productForecastController = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const insightRequestRef = useRef(0);
  const insightControllers = useRef<AbortController[]>([]);

  const { status: apiStatus } = useApiStatus();
  const { processQuery } = useQueryHandler();

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  const handleNewChat = useCallback(() => {
    setMessages([]);
  }, []);

  const toggleLanguage = useCallback(() => {
    setLanguage((prev) => (prev === 'uk' ? 'en' : 'uk'));
  }, []);

  const handleSendMessage = useCallback(
    async (content: string) => {
      if (isLoading) return;

      const controller = new AbortController();
      abortControllerRef.current = controller;

      // Add user message
      const userMessage: MessageType = {
        id: `user-${Date.now()}`,
        role: 'user',
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      try {
        const result = await processQuery(content, controller.signal);

        // Ensure result has valid structure
        const structured = result?.structured || { sections: [] };
        if (!Array.isArray(structured.sections)) {
          structured.sections = [];
        }

        // Add assistant message with structured content
        const assistantMessage: MessageType = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          structuredContent: structured,
          sourceQuery: content,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error) {
        if (isAbortError(error)) {
          return;
        }
        const errorTitle = language === 'uk' ? 'Щось пішло не так' : 'Something went wrong';
        const errorMsg = language === 'uk' ? 'Не вдалося обробити запит.' : 'We could not process the request.';
        const errorMessage: MessageType = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          structuredContent: {
            sections: [
              {
                type: 'error',
                title: errorTitle,
                message: error instanceof Error ? error.message : errorMsg,
                retryable: true,
              },
            ],
          },
          sourceQuery: content,
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
        setIsLoading(false);
      }
    },
    [isLoading, processQuery, language]
  );

  const handleCancel = useCallback(() => {
    if (!abortControllerRef.current) return;
    abortControllerRef.current.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
  }, []);

  const handleReact = useCallback((messageId: string, reaction: 'like' | 'dislike') => {
    setMessages((prev) =>
      prev.map((message) =>
        message.id === messageId
          ? { ...message, reaction: message.reaction === reaction ? undefined : reaction }
          : message
      )
    );
  }, []);

  const handleRetry = useCallback(
    (query: string) => {
      if (!query) return;
      handleSendMessage(query);
    },
    [handleSendMessage]
  );

  const handleQuickQuery = useCallback(
    (query: string) => {
      handleSendMessage(query);
    },
    [handleSendMessage]
  );

  const withTimeout = useCallback(
    async <T,>(factory: (controller: AbortController) => Promise<T>, ms: number, label: string): Promise<T> => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), ms);
      try {
        return await factory(controller);
      } catch (err) {
        if ((err as any)?.name === 'AbortError') {
          throw new Error(`${label} timed out`);
        }
        throw err;
      } finally {
        clearTimeout(timer);
      }
    },
    []
  );

  // Extract numeric ID from row - prioritizes exact matches and numeric values
  const extractNumericId = useCallback((row: Record<string, unknown>, candidates: string[]): number | undefined => {
    // First pass: look for exact column matches with numeric values
    for (const [key, value] of Object.entries(row)) {
      const lower = key.toLowerCase();
      if (candidates.some((c) => lower === c)) {
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value;
        }
        if (typeof value === 'string') {
          const num = parseInt(value, 10);
          if (Number.isFinite(num) && String(num) === value.trim()) {
            return num;
          }
        }
      }
    }
    // Second pass: look for partial matches with numeric values
    for (const [key, value] of Object.entries(row)) {
      const lower = key.toLowerCase();
      // Avoid matching columns that are likely names (contains 'name')
      if (lower.includes('name')) continue;
      if (candidates.some((c) => lower.includes(c) && !lower.includes('name'))) {
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value;
        }
        if (typeof value === 'string') {
          const num = parseInt(value, 10);
          if (Number.isFinite(num) && String(num) === value.trim()) {
            return num;
          }
        }
      }
    }
    return undefined;
  }, []);

  // Extract name/text value from row
  const extractName = useCallback((row: Record<string, unknown>, candidates: string[]): string | undefined => {
    for (const [key, value] of Object.entries(row)) {
      const lower = key.toLowerCase();
      if (candidates.some((c) => lower === c || lower.includes(c))) {
        if (typeof value === 'string' && value.length > 0) {
          return value;
        }
      }
    }
    return undefined;
  }, []);

  const handleRowClick = useCallback(
    async (row: Record<string, unknown>) => {
      // Detect context by looking for product-specific or client-specific columns
      const keys = Object.keys(row).map(k => k.toLowerCase());

      // Product context indicators
      const hasProductContext = keys.some(k =>
        k.includes('vendorcode') || k.includes('vendor_code') ||
        k.includes('productgroup') || k.includes('product_group') ||
        k.includes('sku') || k.includes('productid') || k.includes('product_id') ||
        k.includes('price') || k.includes('qty') || k.includes('quantity') ||
        k.includes('totalsold') || k.includes('total_sold') ||
        k.includes('productname') || k.includes('product_name') ||
        k === 'product' || k === 'товар' || k === 'артикул' || k === 'originalvendorcode'
      );

      // Check for VendorCode value pattern (looks like product SKU: letters+numbers like "MG35693-1")
      const hasVendorCodeValue = Object.values(row).some(v =>
        typeof v === 'string' && /^[A-Z]{2,}[\d-]+[A-Z\d]*$/i.test(v.trim())
      );

      // Check for client-like name values (Ukrainian names with ФОП, ТОВ, ПП, etc.)
      const hasClientNameValue = Object.values(row).some(v =>
        typeof v === 'string' && (
          /\b(ФОП|ТОВ|ПП|ТзОВ|ПАТ|ПрАТ|КП|ОП)\b/i.test(v) ||  // Ukrainian business types
          /^[А-ЯІЇЄҐ][а-яіїєґ']+\s+[А-ЯІЇЄҐ][а-яіїєґ']+/u.test(v.trim())  // Ukrainian name pattern
        )
      );

      // Client context indicators
      const hasClientContext = keys.some(k =>
        k.includes('clientagreement') || k.includes('client_agreement') ||
        k.includes('segment') || k.includes('clientid') || k.includes('client_id') ||
        k.includes('customer') || k.includes('клієнт') || k.includes('client') ||
        k === 'fullname' || k === 'firstname' || k === 'lastname'
      ) || hasClientNameValue;  // Also treat as client if name looks like a client name

      // Check if looks like a Product table query (has ID + VendorCode pattern, but NOT client-like names)
      const looksLikeProductTable = (keys.includes('id') || keys.includes('rank')) &&
        (keys.includes('vendorcode') || hasVendorCodeValue) &&
        !hasClientContext && !hasClientNameValue;

      // Extract numeric IDs for API calls
      let clientId: number | undefined;
      let productId: number | undefined;

      // First, try explicit column names
      clientId = extractNumericId(row, ['client_id', 'clientid', 'customer_id', 'customerid', 'clientagreementid', 'agreement_id', 'cid', 'c_id']);
      productId = extractNumericId(row, ['product_id', 'productid', 'pid', 'p_id', 'item_id']);

      // If still no ID, use context detection for generic 'id' column
      if (!clientId && !productId) {
        const genericId = extractNumericId(row, ['id']);
        if (genericId) {
          // Priority: VendorCode (definitive product) > business name > column-based > default
          if (hasVendorCodeValue) {
            // Has VendorCode pattern (like MG35693) - DEFINITELY a product!
            productId = genericId;
          } else if (hasClientNameValue) {
            // Ukrainian business name or person name - definitely a client
            clientId = genericId;
          } else if ((hasProductContext || looksLikeProductTable) && !hasClientContext) {
            productId = genericId;
          } else if (hasClientContext) {
            clientId = genericId;
          } else {
            // Default to client context if truly ambiguous
            clientId = genericId;
          }
        }
      }

      // Extract names for display (if available from row)
      const rowProductName = extractName(row, ['product_name', 'productname', 'name', 'product']);
      const rowClientName = extractName(row, ['client_name', 'clientname', 'customer_name', 'name', 'client']);

      if (!clientId && !productId) return;

      // If ONLY product ID (no client ID) - open full-screen ProductForecastScreen
      if (productId && !clientId) {
        // Abort any previous product forecast request
        productForecastController.current?.abort();
        const controller = new AbortController();
        productForecastController.current = controller;

        setProductForecastScreen({
          isOpen: true,
          loading: true,
          error: null,
          productId,
          productName: rowProductName || null,
          vendorCode: null,
          category: null,
          forecast: null,
          charts: null,
          proof: null,
        });

        try {
          const forecast = await fetchForecastForProduct(productId, controller.signal);
          if (!controller.signal.aborted) {
            setProductForecastScreen((prev) => ({
              ...prev,
              loading: false,
              productName: forecast?.product_name || prev.productName,
              vendorCode: forecast?.vendor_code || null,
              category: forecast?.category || null,
              forecast,
              charts: forecast?.charts || null,
              proof: forecast?.proof || null,
            }));
          }
        } catch (err) {
          if (!controller.signal.aborted) {
            setProductForecastScreen((prev) => ({
              ...prev,
              loading: false,
              error: err instanceof Error ? err.message : 'Failed to load forecast',
            }));
          }
        }
        return;
      }

      // CLIENT context - use the insight panel as before
      // cancel any previous insight fetches
      insightControllers.current.forEach((c) => c.abort());
      insightControllers.current = [];
      const requestId = Date.now();
      insightRequestRef.current = requestId;

      setInsight({
        open: true,
        loading: true,
        clientId,
        clientName: rowClientName || null,
        segment: null,
        productId,
        productName: rowProductName || null,
        vendorCode: null,
        category: null,
        recommendations: [],
        products: [],
        forecast: null,
        charts: null,
        proof: null,
        productCharts: null,
        productProof: null,
        recError: null,
        productError: null,
        forecastError: null,
        error: null,
      });

      // Also open the score panel on the LEFT side for client payment score
      if (clientId !== undefined) {
        // Abort previous score request
        scorePanelController.current?.abort();
        const scoreController = new AbortController();
        scorePanelController.current = scoreController;

        setScorePanel({
          open: true,
          loading: true,
          error: null,
          clientId,
          clientName: rowClientName || null,
          scoreData: null,
        });

        // Fetch client score in parallel
        fetchClientScore(clientId, scoreController.signal)
          .then((response) => {
            if (!scoreController.signal.aborted && response) {
              setScorePanel((prev) => ({
                ...prev,
                loading: false,
                clientName: response.client_name || prev.clientName,
                scoreData: response.score,
              }));
            }
          })
          .catch((err) => {
            if (!scoreController.signal.aborted) {
              setScorePanel((prev) => ({
                ...prev,
                loading: false,
                error: err instanceof Error ? err.message : 'Failed to load score',
              }));
            }
          });
      }

      // Fetch client name IMMEDIATELY (fast) - don't wait for recommendations
      if (clientId !== undefined) {
        fetchClientById(clientId).then((clientDetails) => {
          if (insightRequestRef.current === requestId && clientDetails) {
            const clientName = (clientDetails?.FullName as string) || (clientDetails?.Name as string) || undefined;
            setInsight((prev) => ({
              ...prev,
              clientName,
            }));
            // Also update score panel client name
            setScorePanel((prev) => ({
              ...prev,
              clientName: prev.clientName || clientName,
            }));
          }
        }).catch(() => {});
      }

      try {
        // Fetch full recommendations (with charts/proof) and forecast (slow) in parallel
        const [fullRecs, forecast] = await Promise.all([
          clientId !== undefined
            ? withTimeout(
                (controller) => {
                  insightControllers.current.push(controller);
                  return fetchFullRecommendations(clientId, controller.signal);
                },
                90000,  // 90 seconds - recommendations can be slow without Redis cache
                'Recommendations'
              ).catch((err) => { console.error('Recs error:', err); return null as FullRecommendationResponse | null; })
            : Promise.resolve(null as FullRecommendationResponse | null),
          productId !== undefined
            ? withTimeout(
                (controller) => {
                  insightControllers.current.push(controller);
                  return fetchForecastForProduct(productId, controller.signal);
                },
                15000,
                'Forecast'
              ).catch(() => null)
            : Promise.resolve(null),
        ]);

        if (insightRequestRef.current !== requestId) return;

        // Extract recommendations array and metadata from full response
        const recs = fullRecs?.recommendations || [];
        const charts = fullRecs?.charts || null;
        const proof = fullRecs?.proof || null;
        const segment = fullRecs?.segment || null;
        const apiClientName = fullRecs?.client_name || null;

        let products: Record<string, unknown>[] = [];
        let productError: string | null = null;
        const productIds = recs.map((r: any) => r.product_id).filter((id: number) => Number.isFinite(id)).slice(0, 10);
        if (productIds.length > 0) {
          try {
            products = await withTimeout(
              (controller) => {
                insightControllers.current.push(controller);
                return fetchProductsByIds(productIds, controller.signal);
              },
              8000,
              'Product details'
            );
          } catch (detailErr) {
            console.error('Product detail fetch failed', detailErr);
            productError = detailErr instanceof Error ? detailErr.message : 'Failed to load product details';
          }
        }

        // Extract product data from forecast
        const productName = forecast?.product_name || null;
        const vendorCode = forecast?.vendor_code || null;
        const category = forecast?.category || null;
        const productCharts = forecast?.charts || null;
        const productProof = forecast?.proof || null;

        setInsight((prev) => ({
          ...prev,
          loading: false,
          clientName: prev.clientName || apiClientName,
          segment,
          productName,
          vendorCode,
          category,
          recommendations: recs,
          products,
          forecast,
          charts,
          proof,
          productCharts,
          productProof,
          productError,
          error: null,
        }));
      } catch (error) {
        console.error('Insight fetch failed', error);
        if (insightRequestRef.current === requestId) {
          setInsight((prev) => ({
            ...prev,
            loading: false,
            error: error instanceof Error ? error.message : 'Failed to load insights',
          }));
        }
      }
    },
    [extractNumericId, extractName, withTimeout]
  );

  const showWelcome = messages.length === 0;
  const hasMessages = messages.length > 0;
  const lastUserMessage =
    messages.slice().reverse().find((message) => message.role === 'user')?.content || '';
  const recentLabel = lastUserMessage
    ? lastUserMessage.length > 44
      ? `${lastUserMessage.slice(0, 44)}...`
      : lastUserMessage
    : language === 'uk' ? 'Почніть новий чат' : 'Start a new chat';

  return (
    <div className="app-shell grok-shell light-theme min-h-screen h-screen overflow-hidden relative">
      {/* Background grid layer */}
      <div className="absolute inset-0 bg-gradient-grok pointer-events-none z-0" />

      <div className="relative z-10 flex min-h-screen h-screen">
        {/* Sidebar Overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Collapsible Sidebar */}
        <aside
          className={`
            fixed left-0 top-0 h-full z-50 flex flex-col bg-white border-r border-slate-200 shadow-xl
            transition-all duration-300 ease-in-out
            ${sidebarOpen ? 'w-72 translate-x-0' : 'w-16 translate-x-0'}
          `}
        >
          {/* Toggle Button */}
          <div className="p-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-10 h-10 rounded-lg flex items-center justify-center text-slate-600 hover:bg-slate-100 transition-colors"
              aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>

          {/* New Chat Button */}
          <div className={`px-3 ${sidebarOpen ? '' : 'flex justify-center'}`}>
            <button
              onClick={() => { handleNewChat(); setSidebarOpen(false); }}
              className={`
                flex items-center gap-3 rounded-lg transition-all
                ${sidebarOpen
                  ? 'w-full px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-100'
                  : 'w-10 h-10 justify-center text-slate-600 hover:bg-slate-100'
                }
              `}
              title={language === 'uk' ? 'Новий чат' : 'New chat'}
            >
              <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 5v14M5 12h14" />
              </svg>
              {sidebarOpen && <span>{language === 'uk' ? 'Новий чат' : 'New chat'}</span>}
            </button>
          </div>

          {/* Live Dashboard Button */}
          <div className={`px-3 mt-1 ${sidebarOpen ? '' : 'flex justify-center'}`}>
            <button
              onClick={() => { setShowDashboard(true); setSidebarOpen(false); }}
              className={`
                flex items-center gap-3 rounded-lg transition-all
                ${sidebarOpen
                  ? 'w-full px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-100'
                  : 'w-10 h-10 justify-center text-slate-600 hover:bg-slate-100'
                }
              `}
              title={language === 'uk' ? 'Live Dashboard' : 'Live Dashboard'}
            >
              <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              {sidebarOpen && <span>Live Dashboard</span>}
            </button>
          </div>

          {/* Storage List Button */}
          <div className={`px-3 mt-1 ${sidebarOpen ? '' : 'flex justify-center'}`}>
            <button
              onClick={() => { setShowStoragePanel(!showStoragePanel); }}
              className={`
                flex items-center gap-3 rounded-lg transition-all
                ${showStoragePanel
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-slate-600 hover:bg-slate-100'
                }
                ${sidebarOpen
                  ? 'w-full px-4 py-2.5 text-sm font-medium'
                  : 'w-10 h-10 justify-center'
                }
              `}
              title={language === 'uk' ? 'Склади' : 'Storages'}
            >
              <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
              {sidebarOpen && <span>{language === 'uk' ? 'Склади' : 'Storages'}</span>}
            </button>
          </div>

          {/* Manager List Button */}
          <div className={`px-3 mt-1 ${sidebarOpen ? '' : 'flex justify-center'}`}>
            <button
              onClick={() => { setShowManagerPanel(!showManagerPanel); }}
              className={`
                flex items-center gap-3 rounded-lg transition-all
                ${showManagerPanel
                  ? 'bg-violet-50 text-violet-700'
                  : 'text-slate-600 hover:bg-slate-100'
                }
                ${sidebarOpen
                  ? 'w-full px-4 py-2.5 text-sm font-medium'
                  : 'w-10 h-10 justify-center'
                }
              `}
              title={language === 'uk' ? 'Менеджери' : 'Managers'}
            >
              <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              {sidebarOpen && <span>{language === 'uk' ? 'Менеджери' : 'Managers'}</span>}
            </button>
          </div>

          {/* Recent Chats - Only show when expanded */}
          {sidebarOpen && (
            <div className="flex-1 overflow-y-auto px-3 mt-4">
              <div className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 px-2">
                {language === 'uk' ? 'Останні' : 'Recent'}
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="w-full rounded-lg px-3 py-2.5 text-left text-sm text-slate-600 hover:bg-slate-100 transition-colors"
              >
                <div className="font-medium text-slate-700 truncate">{recentLabel}</div>
                <div className="mt-0.5 text-xs text-slate-400">
                  {hasMessages
                    ? language === 'uk'
                      ? `${messages.length} повідомлень`
                      : `${messages.length} messages`
                    : language === 'uk'
                      ? 'Ще немає повідомлень'
                      : 'No messages yet'}
                </div>
              </button>
            </div>
          )}

          {/* Footer - API Status */}
          <div className={`mt-auto border-t border-slate-200 p-3 ${sidebarOpen ? '' : 'flex justify-center'}`}>
            <div className={`flex items-center gap-2 text-xs text-slate-500 ${sidebarOpen ? 'px-2' : ''}`}>
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${apiStatus.online ? 'bg-emerald-500' : 'bg-rose-500'}`} />
              {sidebarOpen && (
                <span>
                  {apiStatus.online
                    ? language === 'uk' ? 'API онлайн' : 'API online'
                    : language === 'uk' ? 'API офлайн' : 'API offline'}
                </span>
              )}
            </div>
          </div>
        </aside>

        {/* Client Score Panel (left side) */}
        <ClientScorePanel
          open={scorePanel.open}
          loading={scorePanel.loading}
          error={scorePanel.error}
          clientId={scorePanel.clientId}
          clientName={scorePanel.clientName}
          scoreData={scorePanel.scoreData}
          sidebarOpen={sidebarOpen}
          onClose={() => setScorePanel((prev) => ({ ...prev, open: false }))}
        />

        {/* Live Dashboard Modal */}
        {showDashboard && (
          <LiveDashboard onClose={() => setShowDashboard(false)} />
        )}

        {/* Order Recommendations Panel */}
        <OrderRecommendationsPanel
          open={showOrderRecommendations}
          onClose={() => setShowOrderRecommendations(false)}
          language={language}
        />

        {/* Storage List Panel (left side) */}
        <StorageListPanel
          open={showStoragePanel}
          sidebarOpen={sidebarOpen}
          onStorageSelect={(storage) => {
            console.log('Selected storage:', storage);
          }}
          onClose={() => setShowStoragePanel(false)}
        />

        {/* Manager List Panel (left side, positioned below storage if both open) */}
        <ManagerListPanel
          open={showManagerPanel}
          sidebarOpen={sidebarOpen}
          storagesPanelOpen={showStoragePanel}
          onManagerSelect={(manager) => {
            console.log('Selected manager:', manager);
          }}
          onClose={() => setShowManagerPanel(false)}
        />

        {/* Insight Panel (right side) */}
        {insight.open && (
          <>
            {/* Mobile overlay backdrop */}
            <div
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
              onClick={() => setInsight((prev) => ({ ...prev, open: false }))}
            />

            <div
              className="fixed right-6 top-6 bottom-6 w-[90vw] max-w-[680px] overflow-y-auto bg-white border border-slate-200 rounded-2xl shadow-2xl p-5 z-50 flex flex-col"
            >
              <div className="flex items-start justify-between gap-2 mb-4">
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <h2 className="text-lg font-semibold text-slate-800">
                      {insight.clientId
                        ? (insight.clientName || `${language === 'uk' ? 'Клієнт' : 'Client'} #${insight.clientId}`)
                        : insight.productId
                          ? (insight.productName || `${language === 'uk' ? 'Товар' : 'Product'} #${insight.productId}`)
                          : (language === 'uk' ? 'Аналітика' : 'Analytics')}
                    </h2>
                    {insight.segment && (
                      <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                        insight.segment === 'HEAVY' ? 'bg-violet-100 text-violet-700' :
                        insight.segment === 'MEDIUM' ? 'bg-blue-100 text-blue-700' :
                        insight.segment === 'LIGHT' ? 'bg-emerald-100 text-emerald-700' :
                        insight.segment === 'DORMANT' ? 'bg-amber-100 text-amber-700' :
                        'bg-slate-100 text-slate-700'
                      }`}>
                        {insight.segment}
                      </span>
                    )}
                    {insight.category && (
                      <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-slate-100 text-slate-600">
                        {insight.category}
                      </span>
                    )}
                  </div>
                  {insight.vendorCode && <div className="text-sm text-slate-500 mt-1">{language === 'uk' ? 'Артикул' : 'SKU'}: {insight.vendorCode}</div>}
                  {insight.productId && !insight.clientId && !insight.vendorCode && <div className="text-sm text-slate-500 mt-1">{language === 'uk' ? 'Товар ID' : 'Product ID'}: {insight.productId}</div>}
                </div>
                <button
                  onClick={() => setInsight((prev) => ({ ...prev, open: false }))}
                  className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors"
                  aria-label="Close insights"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M18 6L6 18" />
                    <path d="M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {insight.loading ? (
                <div className="flex-1 flex items-center justify-center gap-3 text-sm text-slate-500">
                  <div className="w-6 h-6 border-2 border-slate-300 border-t-violet-600 rounded-full animate-spin" />
                  {language === 'uk' ? 'Завантаження рекомендацій...' : 'Loading recommendations...'}
                </div>
              ) : (
                <div className="flex-1 flex flex-col space-y-4 overflow-y-auto">
                  {/* Proof Metrics Grid */}
                  {insight.proof && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? 'Статистика клієнта' : 'Customer Proof'}
                      </div>
                      <div className="grid grid-cols-4 gap-2">
                        <div className="p-2 bg-violet-50 rounded-lg border border-violet-100 text-center">
                          <div className="text-lg font-bold text-violet-700">
                            {insight.proof.total_orders?.toLocaleString() || 0}
                          </div>
                          <div className="text-[10px] text-violet-600">{language === 'uk' ? 'Замовлень' : 'Orders'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded-lg border border-blue-100 text-center">
                          <div className="text-lg font-bold text-blue-700">
                            {insight.proof.avg_order_value ? `${(insight.proof.avg_order_value / 1000).toFixed(1)}K` : 'N/A'}
                          </div>
                          <div className="text-[10px] text-blue-600">{language === 'uk' ? 'Сер. чек' : 'Avg Order'}</div>
                        </div>
                        <div className="p-2 bg-emerald-50 rounded-lg border border-emerald-100 text-center">
                          <div className="text-lg font-bold text-emerald-700">
                            {insight.proof.days_since_last_order != null ? `${insight.proof.days_since_last_order}${language === 'uk' ? 'д' : 'd'}` : 'N/A'}
                          </div>
                          <div className="text-[10px] text-emerald-600">{language === 'uk' ? 'Останнє' : 'Last Order'}</div>
                        </div>
                        <div className="p-2 bg-amber-50 rounded-lg border border-amber-100 text-center">
                          <div className="text-lg font-bold text-amber-700">
                            {insight.proof.model_confidence ? `${(insight.proof.model_confidence * 100).toFixed(0)}%` : 'N/A'}
                          </div>
                          <div className="text-[10px] text-amber-600">{language === 'uk' ? 'Точність' : 'Confidence'}</div>
                        </div>
                      </div>
                      {/* Last Order Date */}
                      {insight.proof.last_order_date && (
                        <div className="text-xs text-slate-500 mt-1">
                          {language === 'uk' ? 'Останнє замовлення' : 'Last order'}: <span className="font-semibold text-slate-700">{new Date(insight.proof.last_order_date).toLocaleDateString(language === 'uk' ? 'uk-UA' : 'en-US', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                        </div>
                      )}
                      {/* Loyalty Bar */}
                      {insight.proof.loyalty_score != null && (
                        <div className="mt-2">
                          <div className="flex justify-between text-xs text-slate-500 mb-1">
                            <span>{language === 'uk' ? 'Лояльність' : 'Loyalty Score'}</span>
                            <span className="font-medium">{(insight.proof.loyalty_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-violet-500 to-violet-600 rounded-full transition-all"
                              style={{ width: `${insight.proof.loyalty_score * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                      {/* Total Spent */}
                      {insight.proof.total_spent != null && insight.proof.total_spent > 0 && (
                        <div className="text-xs text-slate-500 mt-1">
                          {language === 'uk' ? 'Всього витрачено' : 'Total Spent'}: <span className="font-semibold text-slate-700">{insight.proof.total_spent.toLocaleString()} {language === 'uk' ? 'грн' : 'UAH'}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Product Proof Metrics */}
                  {insight.productProof && !insight.clientId && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? 'Статистика товару' : 'Product Stats'}
                      </div>
                      <div className="grid grid-cols-4 gap-2">
                        <div className="p-2 bg-violet-50 rounded-lg border border-violet-100 text-center">
                          <div className="text-lg font-bold text-violet-700">
                            {insight.productProof.total_orders?.toLocaleString() || 0}
                          </div>
                          <div className="text-[10px] text-violet-600">{language === 'uk' ? 'Замовлень' : 'Orders'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded-lg border border-blue-100 text-center">
                          <div className="text-lg font-bold text-blue-700">
                            {insight.productProof.total_qty_sold ? insight.productProof.total_qty_sold.toLocaleString() : 0}
                          </div>
                          <div className="text-[10px] text-blue-600">{language === 'uk' ? 'Продано' : 'Qty Sold'}</div>
                        </div>
                        <div className="p-2 bg-emerald-50 rounded-lg border border-emerald-100 text-center">
                          <div className="text-lg font-bold text-emerald-700">
                            {insight.productProof.unique_customers || 0}
                          </div>
                          <div className="text-[10px] text-emerald-600">{language === 'uk' ? 'Клієнтів' : 'Customers'}</div>
                        </div>
                        <div className="p-2 bg-amber-50 rounded-lg border border-amber-100 text-center">
                          <div className="text-lg font-bold text-amber-700">
                            {insight.productProof.days_since_last_sale != null ? `${insight.productProof.days_since_last_sale}${language === 'uk' ? 'д' : 'd'}` : 'N/A'}
                          </div>
                          <div className="text-[10px] text-amber-600">{language === 'uk' ? 'Останній' : 'Last Sale'}</div>
                        </div>
                      </div>
                      {/* Last Sale Date */}
                      {insight.productProof.last_sale_date && (
                        <div className="text-xs text-slate-500 mt-1">
                          {language === 'uk' ? 'Останній продаж' : 'Last sale'}: <span className="font-semibold text-slate-700">{new Date(insight.productProof.last_sale_date).toLocaleDateString(language === 'uk' ? 'uk-UA' : 'en-US', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                        </div>
                      )}
                      {/* Total Revenue */}
                      {insight.productProof.total_revenue != null && insight.productProof.total_revenue > 0 && (
                        <div className="text-xs text-slate-500 mt-1">
                          {language === 'uk' ? 'Загальний дохід' : 'Total Revenue'}: <span className="font-semibold text-slate-700">{insight.productProof.total_revenue.toLocaleString()} {language === 'uk' ? 'грн' : 'UAH'}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Product Sales History Chart with Tabs */}
                  {insight.productCharts?.sales_history && insight.productCharts.sales_history.length > 0 && !insight.clientId && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                          {language === 'uk' ? 'Історія продажів' : 'Sales History'}
                        </div>
                        <div className="flex gap-1 bg-slate-100 rounded-lg p-0.5">
                          <button
                            onClick={() => setProductChartTab('6m')}
                            className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                              productChartTab === '6m' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                            }`}
                          >
                            {language === 'uk' ? '6 міс' : '6 mo'}
                          </button>
                          <button
                            onClick={() => setProductChartTab('1y')}
                            className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                              productChartTab === '1y' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                            }`}
                          >
                            {language === 'uk' ? '1 рік' : '1 year'}
                          </button>
                        </div>
                      </div>
                      {(() => {
                        const data = productChartTab === '6m'
                          ? insight.productCharts!.sales_history.slice(-6)
                          : insight.productCharts!.sales_history;
                        const maxQty = Math.max(...data.map((p: any) => p.qty || 0));

                        if (productChartTab === '6m') {
                          return (
                            <div className="h-36 flex items-end gap-1">
                              {data.map((item: any, idx: number) => {
                                const height = maxQty > 0 ? ((item.qty || 0) / maxQty) * 100 : 0;
                                return (
                                  <div key={idx} className="flex-1 flex flex-col items-center gap-1">
                                    <div className="text-[9px] text-slate-500 font-medium">
                                      {item.qty ? item.qty.toLocaleString() : '0'}
                                    </div>
                                    <div
                                      className="w-full bg-gradient-to-t from-emerald-500 to-emerald-400 rounded-t transition-all hover:from-emerald-600 hover:to-emerald-500 cursor-pointer"
                                      style={{ height: `${Math.max(height, 4)}%` }}
                                      title={`${item.month}: ${item.orders} ${language === 'uk' ? 'замовлень' : 'orders'}, ${item.qty?.toLocaleString()} ${language === 'uk' ? 'шт' : 'qty'}`}
                                    />
                                    <div className="text-[9px] text-slate-400">{item.month?.slice(5) || ''}</div>
                                  </div>
                                );
                              })}
                            </div>
                          );
                        } else {
                          const points = data.map((item: any, idx: number) => {
                            const x = (idx / Math.max(data.length - 1, 1)) * 100;
                            const y = maxQty > 0 ? 100 - ((item.qty || 0) / maxQty) * 100 : 100;
                            return `${x},${y}`;
                          }).join(' ');
                          const areaPoints = `0,100 ${points} 100,100`;

                          return (
                            <div className="h-36 relative">
                              <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                <line x1="0" y1="25" x2="100" y2="25" stroke="#e2e8f0" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                                <line x1="0" y1="50" x2="100" y2="50" stroke="#e2e8f0" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                                <line x1="0" y1="75" x2="100" y2="75" stroke="#e2e8f0" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                                <polygon points={areaPoints} fill="url(#productAreaGradient)" />
                                <polyline points={points} fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />

                                <defs>
                                  <linearGradient id="productAreaGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
                                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.05" />
                                  </linearGradient>
                                </defs>
                              </svg>
                              {/* Crisp dot markers */}
                              {data.map((item: any, idx: number) => {
                                const xPct = (idx / Math.max(data.length - 1, 1)) * 100;
                                const yPct = maxQty > 0 ? ((item.qty || 0) / maxQty) * 100 : 0;
                                return (
                                  <div
                                    key={idx}
                                    className="absolute w-2 h-2 bg-emerald-500 rounded-full border border-white shadow-sm hover:scale-150 transition-transform cursor-pointer"
                                    style={{ left: `${xPct}%`, bottom: `${yPct}%`, transform: 'translate(-50%, 50%)' }}
                                    title={`${item.month}: ${item.orders} ${language === 'uk' ? 'замовлень' : 'orders'}, ${item.qty?.toLocaleString()} ${language === 'uk' ? 'шт' : 'qty'}`}
                                  />
                                );
                              })}
                              <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[8px] text-slate-400 -mb-4">
                                {data.filter((_: any, i: number) => i === 0 || i === Math.floor(data.length / 2) || i === data.length - 1).map((item: any, idx: number) => (
                                  <span key={idx}>{item.month?.slice(2, 7) || ''}</span>
                                ))}
                              </div>
                            </div>
                          );
                        }
                      })()}
                    </div>
                  )}

                  {/* Top Customers for Product */}
                  {insight.productCharts?.top_customers && insight.productCharts.top_customers.length > 0 && !insight.clientId && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? 'Топ покупці' : 'Top Customers'}
                      </div>
                      <div className="divide-y divide-slate-100 border border-slate-200 rounded-lg overflow-hidden">
                        {insight.productCharts.top_customers.slice(0, 5).map((customer: any, idx: number) => (
                          <div
                            key={customer.customer_id || idx}
                            className="flex items-center justify-between p-2 bg-white hover:bg-slate-50 cursor-pointer transition-colors"
                            onClick={() => handleRowClick({ client_id: customer.customer_id })}
                          >
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-slate-800 truncate">
                                {customer.customer_name || `${language === 'uk' ? 'Клієнт' : 'Customer'} #${customer.customer_id}`}
                              </div>
                              <div className="text-xs text-slate-500">
                                {customer.order_count} {language === 'uk' ? 'замовлень' : 'orders'}
                              </div>
                            </div>
                            <div className="text-right ml-2">
                              <div className="text-sm font-bold text-emerald-600">{customer.total_qty?.toLocaleString()}</div>
                              <div className="text-[10px] text-slate-400">{language === 'uk' ? 'шт' : 'qty'}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Purchase History Chart with Tabs */}
                  {insight.charts?.purchase_history && insight.charts.purchase_history.length > 0 && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                          {language === 'uk' ? 'Історія покупок' : 'Purchase History'}
                        </div>
                        <div className="flex gap-1 bg-slate-100 rounded-lg p-0.5">
                          <button
                            onClick={() => setChartTab('6m')}
                            className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                              chartTab === '6m' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                            }`}
                          >
                            {language === 'uk' ? '6 міс' : '6 mo'}
                          </button>
                          <button
                            onClick={() => setChartTab('1y')}
                            className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                              chartTab === '1y' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                            }`}
                          >
                            {language === 'uk' ? '1 рік' : '1 year'}
                          </button>
                        </div>
                      </div>
                      {(() => {
                        const data = chartTab === '6m'
                          ? insight.charts!.purchase_history.slice(-6)
                          : insight.charts!.purchase_history;
                        const maxAmount = Math.max(...data.map((p: any) => p.amount || 0));

                        if (chartTab === '6m') {
                          // Bar chart for 6 months
                          return (
                            <div className="h-36 flex items-end gap-1">
                              {data.map((item: any, idx: number) => {
                                const height = maxAmount > 0 ? ((item.amount || 0) / maxAmount) * 100 : 0;
                                return (
                                  <div key={idx} className="flex-1 flex flex-col items-center gap-1">
                                    <div className="text-[9px] text-slate-500 font-medium">
                                      {item.amount ? `${(item.amount / 1000).toFixed(0)}K` : '0'}
                                    </div>
                                    <div
                                      className="w-full bg-gradient-to-t from-blue-500 to-blue-400 rounded-t transition-all hover:from-blue-600 hover:to-blue-500 cursor-pointer"
                                      style={{ height: `${Math.max(height, 4)}%` }}
                                      title={`${item.month}: ${item.orders} ${language === 'uk' ? 'замовлень' : 'orders'}, ${item.amount?.toLocaleString()} ${language === 'uk' ? 'грн' : 'UAH'}`}
                                    />
                                    <div className="text-[9px] text-slate-400">{item.month?.slice(5) || ''}</div>
                                  </div>
                                );
                              })}
                            </div>
                          );
                        } else {
                          // Line chart for 1 year
                          const points = data.map((item: any, idx: number) => {
                            const x = (idx / Math.max(data.length - 1, 1)) * 100;
                            const y = maxAmount > 0 ? 100 - ((item.amount || 0) / maxAmount) * 100 : 100;
                            return `${x},${y}`;
                          }).join(' ');
                          const areaPoints = `0,100 ${points} 100,100`;

                          return (
                            <div className="h-36 relative">
                              <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                {/* Grid lines */}
                                <line x1="0" y1="25" x2="100" y2="25" stroke="#e2e8f0" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                                <line x1="0" y1="50" x2="100" y2="50" stroke="#e2e8f0" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                                <line x1="0" y1="75" x2="100" y2="75" stroke="#e2e8f0" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                                {/* Area fill */}
                                <polygon points={areaPoints} fill="url(#areaGradient)" />
                                {/* Line */}
                                <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />

                                <defs>
                                  <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
                                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.05" />
                                  </linearGradient>
                                </defs>
                              </svg>
                              {/* Crisp dot markers */}
                              {data.map((item: any, idx: number) => {
                                const xPct = (idx / Math.max(data.length - 1, 1)) * 100;
                                const yPct = maxAmount > 0 ? ((item.amount || 0) / maxAmount) * 100 : 0;
                                return (
                                  <div
                                    key={idx}
                                    className="absolute w-2 h-2 bg-blue-500 rounded-full border border-white shadow-sm hover:scale-150 transition-transform cursor-pointer"
                                    style={{ left: `${xPct}%`, bottom: `${yPct}%`, transform: 'translate(-50%, 50%)' }}
                                    title={`${item.month}: ${item.orders} ${language === 'uk' ? 'замовлень' : 'orders'}, ${item.amount?.toLocaleString()} ${language === 'uk' ? 'грн' : 'UAH'}`}
                                  />
                                );
                              })}
                              {/* X-axis labels */}
                              <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[8px] text-slate-400 -mb-4">
                                {data.filter((_: any, i: number) => i === 0 || i === Math.floor(data.length / 2) || i === data.length - 1).map((item: any, idx: number) => (
                                  <span key={idx}>{item.month?.slice(2, 7) || ''}</span>
                                ))}
                              </div>
                            </div>
                          );
                        }
                      })()}
                    </div>
                  )}

                  {/* Recommendation Sources */}
                  {insight.charts?.recommendation_sources && insight.charts.recommendation_sources.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? 'Джерела рекомендацій' : 'Recommendation Sources'}
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {insight.charts.recommendation_sources.map((src: any, idx: number) => (
                          <div key={idx} className={`px-2 py-1 text-xs rounded-full font-medium ${
                            src.source === 'repurchase' ? 'bg-emerald-100 text-emerald-700' :
                            src.source === 'co-purchase' ? 'bg-blue-100 text-blue-700' :
                            src.source === 'discovery' ? 'bg-violet-100 text-violet-700' :
                            src.source === 'similar' ? 'bg-amber-100 text-amber-700' :
                            'bg-slate-100 text-slate-700'
                          }`}>
                            {src.source}: {src.count}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Top Categories */}
                  {insight.charts?.top_categories && insight.charts.top_categories.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? 'Топ категорії' : 'Top Categories'}
                      </div>
                      <div className="space-y-1">
                        {insight.charts.top_categories.slice(0, 5).map((cat: any, idx: number) => (
                          <div key={idx} className="flex items-center gap-2">
                            <div className="flex-1 h-4 bg-slate-100 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-slate-400 to-slate-500 rounded-full"
                                style={{ width: `${cat.percentage || 0}%` }}
                              />
                            </div>
                            <div className="text-xs text-slate-600 w-24 truncate">{cat.category}</div>
                            <div className="text-xs font-medium text-slate-700 w-10 text-right">{cat.percentage}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recommended Products */}
                  {!!insight.products.length && (
                    <div className="space-y-1">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? `Рекомендовані товари (${insight.products.length})` : `Recommended Products (${insight.products.length})`}
                      </div>
                      <div className="divide-y divide-slate-100 border border-slate-200 rounded-lg overflow-hidden">
                        {insight.products.map((product: any, idx: number) => {
                          const rec = insight.recommendations.find((r: any) => r.product_id === product.ID);
                          return (
                            <div
                              key={product.ID || idx}
                              className="flex items-center justify-between p-2 bg-white hover:bg-slate-50 cursor-pointer transition-colors"
                              onClick={() => handleRowClick({ product_id: product.ID })}
                            >
                              <div className="flex-1 min-w-0">
                                <div className="text-sm font-medium text-slate-800 truncate">
                                  {product.Name || product.ProductName || 'Unknown Product'}
                                </div>
                                <div className="flex items-center gap-2 text-xs text-slate-500">
                                  <span>{product.VendorCode || '-'}</span>
                                  {rec?.source && (
                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                      rec.source === 'repurchase' ? 'bg-emerald-50 text-emerald-600' :
                                      rec.source === 'co-purchase' ? 'bg-blue-50 text-blue-600' :
                                      rec.source === 'discovery' ? 'bg-violet-50 text-violet-600' :
                                      'bg-slate-50 text-slate-600'
                                    }`}>
                                      {rec.source}
                                    </span>
                                  )}
                                </div>
                              </div>
                              <div className="text-right ml-2">
                                <div className="text-xs font-medium text-slate-700">
                                  {rec?.score ? `${(rec.score * 100).toFixed(0)}%` : `#${idx + 1}`}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {!!insight.recommendations.length && !insight.products.length && (
                    <div className="flex items-center justify-center gap-3 py-8 text-sm text-slate-500">
                      <div className="w-5 h-5 border-2 border-slate-300 border-t-violet-600 rounded-full animate-spin" />
                      {language === 'uk' ? 'Завантаження товарів...' : 'Loading product details...'}
                    </div>
                  )}

                  {insight.forecast && (
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        {language === 'uk' ? 'Прогноз попиту' : 'Demand Forecast'}
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-blue-50 rounded-lg border border-blue-100">
                          <div className="text-lg font-bold text-blue-700">
                            {insight.forecast.summary?.total_predicted_quantity?.toLocaleString() || 'N/A'}
                          </div>
                          <div className="text-xs text-blue-600">{language === 'uk' ? 'Прогноз к-сть' : 'Predicted Qty'}</div>
                        </div>
                        <div className="p-2 bg-amber-50 rounded-lg border border-amber-100">
                          <div className="text-lg font-bold text-amber-700">
                            {insight.forecast.summary?.at_risk_customers || 0}
                          </div>
                          <div className="text-xs text-amber-600">{language === 'uk' ? 'Під ризиком' : 'At Risk'}</div>
                        </div>
                        <div className="p-2 bg-emerald-50 rounded-lg border border-emerald-100">
                          <div className="text-lg font-bold text-emerald-700">
                            {insight.forecast.summary?.active_customers || 0}
                          </div>
                          <div className="text-xs text-emerald-600">{language === 'uk' ? 'Активних' : 'Active'}</div>
                        </div>
                        <div className="p-2 bg-violet-50 rounded-lg border border-violet-100">
                          <div className="text-lg font-bold text-violet-700">
                            {insight.forecast.summary?.average_weekly_quantity?.toFixed(0) || 'N/A'}
                          </div>
                          <div className="text-xs text-violet-600">{language === 'uk' ? 'Сер./тижд' : 'Avg/Week'}</div>
                        </div>
                      </div>
                      {/* Forecast Timeline */}
                      {insight.forecast.weekly_data && insight.forecast.weekly_data.length > 0 && (
                        <div className="mt-3">
                          <div className="text-[10px] text-slate-400 mb-1">{language === 'uk' ? 'Тижневий прогноз' : 'Weekly Timeline'}</div>
                          <div className="h-20 flex items-end gap-0.5">
                            {insight.forecast.weekly_data.slice(0, 13).map((week: any, idx: number) => {
                              const qty = week.data_type === 'actual' ? (week.quantity || 0) : (week.predicted_quantity || 0);
                              const maxQty = Math.max(...insight.forecast!.weekly_data.slice(0, 13).map((w: any) =>
                                w.data_type === 'actual' ? (w.quantity || 0) : (w.predicted_quantity || 0)
                              ));
                              const height = maxQty > 0 ? (qty / maxQty) * 100 : 0;
                              const isPredicted = week.data_type === 'predicted';
                              return (
                                <div key={idx} className="flex-1 flex flex-col items-center gap-0.5">
                                  <div
                                    className={`w-full rounded-t transition-all cursor-pointer ${
                                      isPredicted
                                        ? 'bg-gradient-to-t from-blue-400 to-blue-300 opacity-70'
                                        : 'bg-gradient-to-t from-slate-500 to-slate-400'
                                    }`}
                                    style={{ height: `${Math.max(height, 4)}%` }}
                                    title={`${week.week_start}: ${qty.toLocaleString()} ${language === 'uk' ? 'шт' : 'qty'} (${isPredicted ? (language === 'uk' ? 'прогноз' : 'forecast') : (language === 'uk' ? 'факт' : 'actual')})`}
                                  />
                                </div>
                              );
                            })}
                          </div>
                          <div className="flex justify-between text-[8px] text-slate-400 mt-1">
                            <span>{language === 'uk' ? 'Факт' : 'Actual'}</span>
                            <span>{language === 'uk' ? 'Прогноз' : 'Forecast'}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {!insight.loading && !insight.recommendations.length && !insight.products.length && !insight.forecast && !insight.proof && (
                    <div className="text-xs text-slate-500">
                      {language === 'uk' ? 'Немає даних для цього рядка.' : 'No insight data for this row.'}
                    </div>
                  )}

                  {insight.error && (
                    <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded-lg p-2">
                      {insight.error}
                    </div>
                  )}
                </div>
              )}
            </div>
          </>
        )}

        {/* Main Column */}
        <div className={`flex-1 flex flex-col transition-all duration-300 ${sidebarOpen ? 'ml-72' : 'ml-16'}`}>
          <header className="flex-shrink-0 border-b border-slate-200 bg-white/85 backdrop-blur-xl sticky top-0 z-10">
            <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between text-slate-900">
              <div className="flex items-center gap-3">
                <img src={logoSvg} alt="BI logo" className="h-8" />
                <h1 className="text-base font-semibold text-slate-900">Business Intelligent</h1>
              </div>

              <div className="flex items-center gap-3">
                <div className="hidden sm:flex items-center gap-2 text-xs text-slate-600">
                  <span className={`w-2 h-2 rounded-full ${apiStatus.online ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                  {apiStatus.online ? 'Online' : 'Offline'}
                </div>

                <button
                  onClick={() => setShowDashboard(true)}
                  className="flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700
                             hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 transition-colors"
                  title="Live Dashboard"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <span className="hidden sm:inline">Live</span>
                </button>

                <button
                  onClick={() => setShowOrderRecommendations(true)}
                  className="flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700
                             hover:border-violet-300 hover:bg-violet-50 hover:text-violet-700 transition-colors"
                  title={language === 'uk' ? 'Рекомендації замовлень' : 'Order Recommendations'}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                  </svg>
                  <span className="hidden sm:inline">{language === 'uk' ? 'Замовлення' : 'Orders'}</span>
                </button>

                <button
                  onClick={toggleLanguage}
                  className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700
                             hover:border-slate-300 hover:bg-slate-50 transition-colors"
                  title={language === 'uk' ? 'Switch to English' : 'Перейти на українську'}
                >
                  {language === 'uk' ? 'UA' : 'EN'}
                </button>
              </div>
            </div>
          </header>

          {/* Main Chat Area */}
          <main className="flex-1 overflow-hidden">
            <div className="max-w-5xl mx-auto px-4 py-8 h-full flex flex-col">
              {showWelcome ? (
                <WelcomeMessage onQuickQuery={handleQuickQuery} language={language} />
              ) : (
                <div className="space-y-6 pb-8 chat-scroll">
                  {messages.map((message, index) => (
                    <div
                      key={message.id}
                      className="animate-rise"
                      style={{ animationDelay: `${Math.min(index * 0.04, 0.4)}s` }}
                    >
                      <Message
                        message={message}
                        onReact={handleReact}
                        onRetry={handleRetry}
                        disableRetry={isLoading}
                        language={language}
                        onRowClick={handleRowClick}
                      />
                    </div>
                  ))}
                  {isLoading && (
                    <div className="animate-rise">
                      <LoadingMessage onCancel={handleCancel} language={language} />
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          </main>

          {/* Chat Input */}
          <ChatInput onSend={handleSendMessage} disabled={isLoading} />
        </div>
      </div>

      {/* Product Forecast Full Screen */}
      <ProductForecastScreen
        isOpen={productForecastScreen.isOpen}
        onClose={() => setProductForecastScreen((prev) => ({ ...prev, isOpen: false }))}
        language={language}
        loading={productForecastScreen.loading}
        error={productForecastScreen.error}
        productId={productForecastScreen.productId}
        productName={productForecastScreen.productName}
        vendorCode={productForecastScreen.vendorCode}
        category={productForecastScreen.category}
        forecast={productForecastScreen.forecast}
        charts={productForecastScreen.charts}
        proof={productForecastScreen.proof}
      />
    </div>
  );
}

export default App;
