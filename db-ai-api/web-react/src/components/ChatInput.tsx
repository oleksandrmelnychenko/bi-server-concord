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
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
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
    <div className="p-3 pb-4 bg-white border-t border-surface-border">
      <div className="max-w-3xl mx-auto">
        <div className={`
          relative bg-white rounded-2xl border
          ${disabled ? 'border-gray-200' : 'border-gray-300 hover:border-gray-400'}
          focus-within:border-gray-400
          transition-all duration-150
        `}>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Напишіть ваше питання..."
            rows={1}
            disabled={disabled}
            className="w-full bg-transparent px-4 py-3 pr-12 resize-none outline-none
                     text-content-primary placeholder:text-content-light font-sans text-sm
                     disabled:cursor-not-allowed disabled:opacity-60
                     max-h-48"
          />
          <button
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            className={`
              absolute right-2 bottom-2 w-8 h-8 rounded-lg
              flex items-center justify-center transition-all duration-150
              ${value.trim() && !disabled
                ? 'bg-black text-white hover:bg-gray-800'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }
            `}
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13" />
              <path d="M22 2L15 22L11 13L2 9L22 2Z" />
            </svg>
          </button>
        </div>
        <p className="text-center text-xs text-content-muted mt-2">
          Пошук по базі даних ConcordDb з українською підтримкою
        </p>
      </div>
    </div>
  );
};
