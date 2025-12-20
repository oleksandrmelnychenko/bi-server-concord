import React from 'react';
import type { Message as MessageType } from '../types';

interface MessageProps {
  message: MessageType;
}

export const Message: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`animate-fade-in flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex gap-3 max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`
          w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0
          text-xs font-semibold
          ${isUser
            ? 'bg-primary-500 text-white'
            : 'bg-content-primary text-white'
          }
        `}>
          {isUser ? 'Ви' : 'AI'}
        </div>

        {/* Message Bubble */}
        <div className={`
          rounded-2xl px-4 py-3
          ${isUser
            ? 'bg-primary-500 text-white rounded-br-md'
            : 'bg-surface-secondary border border-surface-border rounded-bl-md'
          }
        `}>
          <div
            className={isUser ? '' : 'prose prose-sm max-w-none text-content-primary'}
            dangerouslySetInnerHTML={{
              __html: isUser ? escapeHtml(message.content) : message.content,
            }}
          />
        </div>
      </div>
    </div>
  );
};

// Loading indicator component
export const LoadingMessage: React.FC = () => {
  return (
    <div className="animate-fade-in flex justify-start">
      <div className="flex gap-3 max-w-[85%]">
        {/* AI Avatar */}
        <div className="w-9 h-9 rounded-full bg-content-primary
                      flex items-center justify-center text-white text-xs font-semibold">
          AI
        </div>

        {/* Loading Bubble */}
        <div className="bg-surface-secondary border border-surface-border rounded-2xl rounded-bl-md px-5 py-4">
          <div className="flex gap-1.5">
            <span
              className="w-2.5 h-2.5 bg-gray-400 rounded-full animate-bounce-dot"
              style={{ animationDelay: '-0.32s' }}
            />
            <span
              className="w-2.5 h-2.5 bg-gray-400 rounded-full animate-bounce-dot"
              style={{ animationDelay: '-0.16s' }}
            />
            <span className="w-2.5 h-2.5 bg-gray-400 rounded-full animate-bounce-dot" />
          </div>
        </div>
      </div>
    </div>
  );
};

function escapeHtml(text: string): string {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
