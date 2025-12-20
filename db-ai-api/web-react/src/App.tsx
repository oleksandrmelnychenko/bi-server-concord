import { useState, useCallback, useRef, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';
import { Sidebar } from './components/Sidebar';
import { WelcomeMessage } from './components/WelcomeMessage';
import { Message, LoadingMessage } from './components/Message';
import { ChatInput } from './components/ChatInput';
import { ChartModal } from './components/ChartModal';
import { useApiStatus } from './hooks/useApiStatus';
import { useQueryHandler } from './hooks/useQueryHandler';
import type { Message as MessageType, ChartType } from './types';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

function App() {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeChart, setActiveChart] = useState<ChartType | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const { status: apiStatus } = useApiStatus();
  const { processQuery, getChartData } = useQueryHandler();

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  // Initialize inline charts after message renders
  useEffect(() => {
    const initCharts = () => {
      const chartData = getChartData();
      Object.entries(chartData).forEach(([chartId, config]) => {
        const canvas = document.getElementById(chartId) as HTMLCanvasElement;
        if (canvas && !canvas.dataset.initialized) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            const chartConfig = config as {
              type: string;
              data: unknown;
              indexAxis?: string;
            };
            new ChartJS(ctx, {
              type: chartConfig.type as 'bar' | 'doughnut',
              data: chartConfig.data as never,
              options: getChartOptions(chartConfig.type, chartConfig.indexAxis),
            });
            canvas.dataset.initialized = 'true';
          }
        }
      });
    };

    const timeout = setTimeout(initCharts, 100);
    return () => clearTimeout(timeout);
  }, [messages, getChartData]);

  const handleNewChat = useCallback(() => {
    setMessages([]);
  }, []);

  const handleSendMessage = useCallback(
    async (content: string) => {
      if (isLoading) return;

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
        const response = await processQuery(content);

        // Add assistant message
        const assistantMessage: MessageType = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error) {
        const errorMessage: MessageType = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Помилка: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, processQuery]
  );

  const handleQuickQuery = useCallback(
    (query: string) => {
      handleSendMessage(query);
    },
    [handleSendMessage]
  );

  const handleShowChart = useCallback((chartType: ChartType) => {
    setActiveChart(chartType);
  }, []);

  const handleCloseChart = useCallback(() => {
    setActiveChart(null);
  }, []);

  const showWelcome = messages.length === 0;

  return (
    <div className="flex h-screen bg-white">
      <Sidebar
        apiStatus={apiStatus}
        onNewChat={handleNewChat}
        onQuickQuery={handleQuickQuery}
        onShowChart={handleShowChart}
      />

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <div className="flex-1 overflow-y-auto px-6 py-8" ref={chatContainerRef}>
          {showWelcome && <WelcomeMessage onQuickQuery={handleQuickQuery} />}

          <div className="max-w-4xl mx-auto space-y-6 pb-6">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            {isLoading && <LoadingMessage />}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <ChatInput onSend={handleSendMessage} disabled={isLoading} />
      </main>

      {activeChart && <ChartModal chartType={activeChart} onClose={handleCloseChart} />}
    </div>
  );
}

function getChartOptions(type: string, indexAxis?: string) {
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: type === 'doughnut',
        position: 'bottom' as const,
        labels: { color: '#6b7280', padding: 20 },
      },
    },
  };

  if (type === 'doughnut') {
    return {
      ...baseOptions,
      cutout: '60%',
    };
  }

  return {
    ...baseOptions,
    indexAxis: (indexAxis || 'x') as 'x' | 'y',
    scales: {
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(0, 0, 0, 0.05)' },
        ticks: { color: '#6b7280' },
      },
      x: {
        grid: { color: 'rgba(0, 0, 0, 0.05)' },
        ticks: { color: '#6b7280' },
      },
    },
  };
}

export default App;
