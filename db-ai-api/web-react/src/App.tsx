import { useState, useCallback, useRef, useEffect } from 'react';
import { WelcomeMessage } from './components/WelcomeMessage';
import { Message, LoadingMessage } from './components/Message';
import { ChatInput } from './components/ChatInput';
import { useApiStatus } from './hooks/useApiStatus';
import { useQueryHandler } from './hooks/useQueryHandler';
import type { Message as MessageType } from './types';

const isAbortError = (error: unknown): boolean => {
  if (error instanceof DOMException) {
    return error.name === 'AbortError';
  }
  return (error as Error | undefined)?.name === 'AbortError';
};

function App() {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

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
        const errorMessage: MessageType = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          structuredContent: {
            sections: [
              {
                type: 'error',
                title: 'Something went wrong',
                message: error instanceof Error ? error.message : 'We could not process the request.',
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
    [isLoading, processQuery]
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

  const showWelcome = messages.length === 0;
  const hasMessages = messages.length > 0;
  const lastUserMessage =
    messages.slice().reverse().find((message) => message.role === 'user')?.content || '';
  const recentLabel = lastUserMessage
    ? lastUserMessage.length > 44
      ? `${lastUserMessage.slice(0, 44)}...`
      : lastUserMessage
    : 'Start a new chat';

  return (
    <div className="app-shell grok-shell min-h-screen">
      <div className="flex min-h-screen bg-gradient-grok">
        {/* Left Rail */}
        <aside className="hidden lg:flex w-72 flex-col border-r border-white/10 bg-surface-secondary text-slate-100">
          <div className="p-6">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-2xl border border-sky-500/30 bg-white/5 flex items-center justify-center">
                <svg className="w-5 h-5 text-sky-300 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M12 4.5l1.7 4.2L18 10l-4.3 1.3L12 16.5l-1.7-4.2L6 10l4.3-1.3L12 4.5z" />
                  <path d="M5 17l.9 2.2L8 20l-2.1.8L5 23l-.9-2.2L2 20l2.1-.8L5 17z" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-slate-100">Concord Business Intelligent</h1>
                <p className="text-xs text-slate-500">Business Intelligence</p>
              </div>
            </div>

            <button
              onClick={handleNewChat}
              className="mt-6 w-full rounded-full border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-100
                         hover:border-sky-300/60 hover:bg-white/10 transition-colors"
            >
              New chat
            </button>
          </div>

          <div className="px-6">
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Recent</div>
            <button
              onClick={handleNewChat}
              className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm text-slate-200
                         hover:border-sky-300/50 hover:bg-white/10 transition-colors"
            >
              <div className="font-medium text-slate-100">{recentLabel}</div>
              <div className="mt-1 text-xs text-slate-500">
                {hasMessages ? `${messages.length} messages` : 'No messages yet'}
              </div>
            </button>
          </div>

          <div className="mt-auto p-6 border-t border-white/10">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span className={`w-2 h-2 rounded-full ${apiStatus.online ? 'bg-emerald-500' : 'bg-rose-500'}`} />
              {apiStatus.online ? 'API online' : 'API offline'}
            </div>
          </div>
        </aside>

        {/* Main Column */}
        <div className="flex-1 flex flex-col">
          <header className="flex-shrink-0 border-b border-white/10 bg-surface-tertiary/80 backdrop-blur-xl sticky top-0 z-10">
            <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between text-slate-100">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-2xl border border-sky-400/40 bg-white/5 flex items-center justify-center">
                  <svg className="w-5 h-5 text-sky-300 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M12 4.5l1.7 4.2L18 10l-4.3 1.3L12 16.5l-1.7-4.2L6 10l4.3-1.3L12 4.5z" />
                    <path d="M5 17l.9 2.2L8 20l-2.1.8L5 23l-.9-2.2L2 20l2.1-.8L5 17z" />
                  </svg>
                </div>
                <h1 className="text-base font-semibold text-slate-100">Concord Business Intelligent</h1>
              </div>

              <div className="flex items-center gap-3">
                <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400">
                  <span className={`w-2 h-2 rounded-full ${apiStatus.online ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                  {apiStatus.online ? 'Online' : 'Offline'}
                </div>

                {hasMessages && (
                  <button
                    onClick={handleNewChat}
                    className="rounded-full border border-white/10 px-3 py-1.5 text-xs uppercase tracking-[0.2em] text-slate-200
                               hover:border-sky-300/50 hover:bg-white/10 transition-colors"
                  >
                    New chat
                  </button>
                )}
              </div>
            </div>
          </header>

          {/* Main Chat Area */}
          <main className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-4 py-8">
              {showWelcome ? (
                <WelcomeMessage onQuickQuery={handleQuickQuery} />
              ) : (
                <div className="space-y-6 pb-6">
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
                      />
                    </div>
                  ))}
                  {isLoading && (
                    <div className="animate-rise">
                      <LoadingMessage onCancel={handleCancel} />
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
    </div>
  );
}

export default App;
