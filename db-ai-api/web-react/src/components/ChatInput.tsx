import React, { useState, useRef, useCallback } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled = false }) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleResize = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    }
  }, []);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;

    onSend(trimmed);
    setValue('');

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value);
      handleResize();
    },
    [handleResize]
  );

  return (
    <div className="flex-shrink-0 bg-gradient-to-t from-surface-primary via-surface-tertiary/70 to-transparent pt-4 pb-6 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Input Container */}
        <div
          className={`
            relative rounded-2xl border shadow-lg
            ${disabled
              ? 'border-white/10 bg-surface-secondary/70'
              : 'border-white/10 bg-surface-secondary hover:border-sky-300/60 focus-within:border-sky-400/80 focus-within:shadow-[0_0_0_1px_rgba(56,189,248,0.35)]'
            }
            transition-all duration-200
          `}
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Message Concord Business Intelligent..."
            rows={1}
            disabled={disabled}
            className="w-full bg-transparent px-4 py-3.5 pr-14 resize-none outline-none
                     text-slate-100 placeholder:text-slate-500 text-sm leading-relaxed
                     disabled:cursor-not-allowed disabled:opacity-60
                     max-h-40"
          />

          {/* Send Button */}
          <button
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            className={`
              absolute right-2.5 bottom-2.5 w-10 h-10 rounded-full
              flex items-center justify-center transition-all duration-200 border
              ${value.trim() && !disabled
                ? 'border-sky-300/60 bg-sky-500 text-white shadow-md hover:bg-sky-600'
                : 'border-white/10 bg-white/5 text-slate-500 cursor-not-allowed'
              }
            `}
            aria-label="Send message"
          >
            <svg className="w-4 h-4 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M4 12l15-7-6 15-2.5-5.5L4 12z" />
              <path d="M11 14l8-9" />
            </svg>
          </button>
        </div>

        {/* Hint text */}
        <p className="text-center text-xs text-slate-400 mt-3">
          Enter to send. Shift + Enter for a new line.
        </p>
      </div>
    </div>
  );
};
