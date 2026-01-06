import React from 'react';
import type { Message as MessageType } from '../types';
import { ResponseRenderer } from './responses';
import type { Language } from './WelcomeMessage';

interface MessageProps {
  message: MessageType;
  onReact?: (messageId: string, reaction: 'like' | 'dislike') => void;
  onRetry?: (query: string) => void;
  disableRetry?: boolean;
  language?: Language;
  onRowClick?: (row: Record<string, unknown>) => void;
}

export const Message: React.FC<MessageProps> = ({ message, onReact, onRetry, disableRetry = false, language = 'uk', onRowClick }) => {
  const isUser = message.role === 'user';
  const hasStructuredContent = !isUser && message.structuredContent;
  const canRetry = Boolean(message.sourceQuery) && Boolean(onRetry);
  const rowClass = isUser ? 'flex-row-reverse gap-2' : 'flex-row gap-4';
  const contentClass = isUser ? 'max-w-[80%] w-fit' : 'flex-1 min-w-0';

  return (
    <div className={`flex ${rowClass}`}>
      {/* Avatar */}
      <div className="flex-shrink-0 pt-0.5">
        {isUser ? (
          <div className="w-9 h-9 rounded-full border border-white/10 bg-white/5 flex items-center justify-center">
            <svg className="w-4 h-4 text-slate-300 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 12a4 4 0 1 0-4-4 4 4 0 0 0 4 4z" />
              <path d="M4 20a8 8 0 0 1 16 0" />
            </svg>
          </div>
        ) : (
          <div className="relative w-9 h-9 rounded-full border border-slate-200 bg-white flex items-center justify-center">
            <svg className="w-4 h-4 text-slate-600 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 4.5l1.7 4.2L18 10l-4.3 1.3L12 16.5l-1.7-4.2L6 10l4.3-1.3L12 4.5z" />
            </svg>
            {hasStructuredContent && (
              <span className="absolute right-0 bottom-0 w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-white" aria-label="Response ready" />
            )}
          </div>
        )}
      </div>

      {/* Message Content */}
      <div className={contentClass}>
        {hasStructuredContent && message.structuredContent ? (
          <div className="rounded-2xl border border-white/20 bg-white overflow-hidden">
            <div className="p-4">
              <ResponseRenderer response={message.structuredContent} language={language} onRowClick={onRowClick} />
            </div>
          </div>
        ) : (
          <div
            className={`
              inline-block rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap shadow-lg
              ${isUser
                ? 'bg-amber-100 text-purple-900 border border-amber-200 ml-auto'
                : 'bg-surface-tertiary border border-white/10 text-slate-100'
              }
            `}
          >
            {message.content}
          </div>
        )}

        {!isUser && (
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
            <button
              type="button"
              onClick={() => onReact?.(message.id, 'like')}
              aria-label="Like response"
              aria-pressed={message.reaction === 'like'}
              title="Like"
              className={`h-8 w-8 rounded-full border transition-colors backdrop-blur
                ${message.reaction === 'like'
                  ? 'border-emerald-300/50 bg-emerald-500/15 text-emerald-200'
                  : 'border-white/10 text-slate-400 hover:border-white/30 hover:text-slate-200'
                }
              `}
            >
              <svg className="mx-auto h-4 w-4 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M14 9V4a2 2 0 0 0-2-2l-4 7v11h9.5a2 2 0 0 0 2-1.5l1-5a2 2 0 0 0-2-2.5H14z" />
                <path d="M7 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h3" />
              </svg>
            </button>

            <button
              type="button"
              onClick={() => onReact?.(message.id, 'dislike')}
              aria-label="Dislike response"
              aria-pressed={message.reaction === 'dislike'}
              title="Dislike"
              className={`h-8 w-8 rounded-full border transition-colors backdrop-blur
                ${message.reaction === 'dislike'
                  ? 'border-rose-300/50 bg-rose-500/15 text-rose-200'
                  : 'border-white/10 text-slate-400 hover:border-white/30 hover:text-slate-200'
                }
              `}
            >
              <svg className="mx-auto h-4 w-4 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M10 15v5a2 2 0 0 0 2 2l4-7V4H6.5a2 2 0 0 0-2 1.5l-1 5a2 2 0 0 0 2 2.5H10z" />
                <path d="M17 2h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-3" />
              </svg>
            </button>

            <button
              type="button"
              onClick={() => message.sourceQuery && onRetry?.(message.sourceQuery)}
              aria-label="Retry response"
              title="Retry"
              disabled={!canRetry || disableRetry}
              className="h-8 w-8 rounded-full border border-white/10 text-slate-400 transition-colors
                         hover:border-white/30 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <svg className="mx-auto h-4 w-4 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 12a9 9 0 1 1-2.64-6.36" />
                <path d="M21 3v6h-6" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// Loading indicator component
interface LoadingMessageProps {
  onCancel?: () => void;
  language?: Language;
}

const thinkingPhrases = [
  'Concording',
  'Reasoning',
  'Thinking',
  'Analyzing',
  'Processing',
];

export const LoadingMessage: React.FC<LoadingMessageProps> = ({ onCancel, language = 'uk' }) => {
  const [phrase] = React.useState(() =>
    thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)]
  );

  return (
    <div className="flex items-center gap-4">
      {/* Elegant thinking indicator */}
      <div className="relative flex items-center">
        <span className="text-[15px] font-light tracking-wide text-slate-500 italic">
          {phrase}
          <span className="inline-flex ml-0.5">
            <span className="animate-[fade_1.4s_ease-in-out_infinite]" style={{ animationDelay: '0ms' }}>.</span>
            <span className="animate-[fade_1.4s_ease-in-out_infinite]" style={{ animationDelay: '200ms' }}>.</span>
            <span className="animate-[fade_1.4s_ease-in-out_infinite]" style={{ animationDelay: '400ms' }}>.</span>
          </span>
        </span>
      </div>

      {/* Minimal stop button */}
      {onCancel && (
        <button
          type="button"
          onClick={onCancel}
          className="group flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs text-slate-400 transition-all hover:text-slate-600 hover:bg-slate-100"
          title={language === 'uk' ? 'Зупинити' : 'Stop'}
        >
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
          <span className="font-medium">{language === 'uk' ? 'Стоп' : 'Stop'}</span>
        </button>
      )}
    </div>
  );
};
