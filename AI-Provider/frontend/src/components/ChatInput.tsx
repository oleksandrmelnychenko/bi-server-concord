import React, { useState, useRef, useCallback } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

type ModelMode = 'Auto' | 'Fast' | 'Precise';

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled = false }) => {
  const [value, setValue] = useState('');
  const [mode, setMode] = useState<ModelMode>('Auto');
  const [showModeDropdown, setShowModeDropdown] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowModeDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const modes: { value: ModelMode; label: string; icon: React.ReactNode }[] = [
    {
      value: 'Auto',
      label: 'Auto',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      ),
    },
    {
      value: 'Fast',
      label: 'Fast',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
        </svg>
      ),
    },
    {
      value: 'Precise',
      label: 'Precise',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="12" r="6" />
          <circle cx="12" cy="12" r="2" />
        </svg>
      ),
    },
  ];


  return (
    <div className="flex-shrink-0 pt-6 pb-8 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Input Container */}
        <div
          className={`
            relative rounded-full border bg-white min-h-[56px]
            ${disabled
              ? 'border-slate-200 bg-slate-50'
              : 'border-slate-200 hover:border-slate-300 focus-within:border-slate-400 focus-within:shadow-[0_2px_12px_rgba(0,0,0,0.08)]'
            }
            transition-all duration-200 ease-out
          `}
        >
          {/* Attachment Icon */}
          <button
            className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors p-1"
            aria-label="Attach file"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
            </svg>
          </button>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="What do you want to know?"
            rows={1}
            disabled={disabled}
            className="w-full bg-transparent pl-14 pr-32 py-4 resize-none outline-none
                     text-slate-900 placeholder:text-slate-400 text-base leading-relaxed
                     disabled:cursor-not-allowed disabled:opacity-60
                     max-h-40"
          />

          {/* Right side controls */}
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {/* Mode Dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setShowModeDropdown(!showModeDropdown)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-slate-600 hover:text-slate-900 transition-colors rounded-full hover:bg-slate-100"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                  <path d="M2 12l10 5 10-5" />
                </svg>
                <span>{mode}</span>
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </button>

              {showModeDropdown && (
                <div className="absolute bottom-full right-0 mb-2 bg-white rounded-xl shadow-lg border border-slate-200 py-1 min-w-[140px] z-50">
                  {modes.map((m) => (
                    <button
                      key={m.value}
                      onClick={() => {
                        setMode(m.value);
                        setShowModeDropdown(false);
                      }}
                      className={`w-full flex items-center gap-2 px-4 py-2 text-sm transition-colors
                        ${mode === m.value
                          ? 'text-slate-900 bg-slate-50'
                          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                        }`}
                    >
                      {m.icon}
                      <span>{m.label}</span>
                      {mode === m.value && (
                        <svg className="w-4 h-4 ml-auto text-slate-900" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M20 6L9 17l-5-5" />
                        </svg>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Send Button */}
            <button
              onClick={handleSend}
              disabled={disabled || !value.trim()}
              className={`
                w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200
                ${value.trim() && !disabled
                  ? 'bg-black hover:bg-slate-800'
                  : 'bg-slate-100 cursor-not-allowed'
                }
              `}
              aria-label="Send message"
            >
              <svg
                className={`w-5 h-5 ${value.trim() && !disabled ? 'fill-white' : 'fill-slate-400'}`}
                viewBox="0 0 24 24"
              >
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            </button>
          </div>
        </div>

        {/* Disclaimer text */}
        <p className="text-center text-xs text-slate-400 mt-4">
          By using this assistant, you agree to our <a href="#" style={{ color: '#6b7280' }} className="hover:text-gray-700 hover:underline transition-colors">Terms</a> and <a href="#" style={{ color: '#6b7280' }} className="hover:text-gray-700 hover:underline transition-colors">Privacy Policy</a>.
        </p>
      </div>
    </div>
  );
};
