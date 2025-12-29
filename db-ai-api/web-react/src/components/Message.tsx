import React from 'react';
import type { Message as MessageType } from '../types';
import { ResponseRenderer } from './responses';

interface MessageProps {
  message: MessageType;
  onReact?: (messageId: string, reaction: 'like' | 'dislike') => void;
  onRetry?: (query: string) => void;
  disableRetry?: boolean;
}

export const Message: React.FC<MessageProps> = ({ message, onReact, onRetry, disableRetry = false }) => {
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
          <div className="w-9 h-9 rounded-full border border-sky-400/40 bg-white/5 flex items-center justify-center">
            <svg className="w-4 h-4 text-sky-300 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 4.5l1.7 4.2L18 10l-4.3 1.3L12 16.5l-1.7-4.2L6 10l4.3-1.3L12 4.5z" />
            </svg>
          </div>
        )}
      </div>

      {/* Message Content */}
      <div className={contentClass}>
        {hasStructuredContent && message.structuredContent ? (
          <div className="rounded-2xl border border-white/10 bg-surface-tertiary overflow-hidden shadow-lg">
            <div className="px-4 py-3 border-b border-white/10 text-xs uppercase tracking-[0.2em] text-slate-400 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              Concord Business Intelligent
            </div>
            <div className="p-4">
              <ResponseRenderer response={message.structuredContent} />
            </div>
          </div>
        ) : (
          <div
            className={`
              inline-block rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap shadow-lg
              ${isUser
                ? 'bg-gradient-to-r from-sky-500 to-cyan-400 text-white border border-sky-300/50 ml-auto'
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
}

export const LoadingMessage: React.FC<LoadingMessageProps> = ({ onCancel }) => {
  return (
    <div className="flex gap-4">
      {/* AI Avatar */}
      <div className="flex-shrink-0 pt-0.5">
        <div className="w-9 h-9 rounded-full border border-sky-400/40 bg-white/5 flex items-center justify-center">
          <svg className="w-4 h-4 text-sky-300 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M12 4.5l1.7 4.2L18 10l-4.3 1.3L12 16.5l-1.7-4.2L6 10l4.3-1.3L12 4.5z" />
          </svg>
        </div>
      </div>

      {/* Loading Bubble */}
      <div className="bg-surface-tertiary border border-white/10 rounded-2xl px-4 py-3 shadow-lg">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-sky-400 rounded-full animate-pulse-dot" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-sky-400 rounded-full animate-pulse-dot" style={{ animationDelay: '180ms' }} />
          <span className="w-2 h-2 bg-sky-400 rounded-full animate-pulse-dot" style={{ animationDelay: '360ms' }} />
        </div>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="mt-3 inline-flex items-center gap-2 rounded-full border border-rose-300/50 bg-rose-500/15 px-3 py-1 text-xs font-medium text-rose-200 transition-colors
                       hover:border-rose-200 hover:bg-rose-500/20"
          >
            <svg className="h-3.5 w-3.5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M18 6L6 18" />
              <path d="M6 6l12 12" />
            </svg>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};
