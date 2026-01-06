import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ErrorStateProps {
  title?: string;
  message: string;
  details?: string;
  retryable?: boolean;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Помилка',
  message,
  details,
  retryable = true,
  onRetry,
  className = '',
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);

  const handleRetry = async () => {
    if (onRetry) {
      setIsRetrying(true);
      try {
        await onRetry();
      } finally {
        setIsRetrying(false);
      }
    }
  };

  return (
    <motion.div
      className={`rounded-xl border border-red-500/20 bg-red-500/5 p-6 ${className}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-start gap-4">
        {/* Error Icon */}
        <motion.div
          className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center"
          initial={{ rotate: -10 }}
          animate={{ rotate: [0, -5, 5, -5, 0] }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </motion.div>

        {/* Error Content */}
        <div className="flex-1 min-w-0">
          <h3 className="text-red-400 font-medium mb-1">{title}</h3>
          <p className="text-white/70 text-sm">{message}</p>

          {/* Details Expansion */}
          {details && (
            <div className="mt-3">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-xs text-white/50 hover:text-white/70 flex items-center gap-1 transition-colors"
              >
                <motion.svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  animate={{ rotate: showDetails ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </motion.svg>
                {showDetails ? 'Сховати деталі' : 'Показати деталі'}
              </button>

              <AnimatePresence>
                {showDetails && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <pre className="mt-2 p-3 bg-black/30 rounded-lg text-xs text-white/50 overflow-x-auto whitespace-pre-wrap break-all">
                      {details}
                    </pre>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Retry Button */}
          {retryable && onRetry && (
            <motion.button
              onClick={handleRetry}
              disabled={isRetrying}
              className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              whileHover={{ scale: isRetrying ? 1 : 1.02 }}
              whileTap={{ scale: isRetrying ? 1 : 0.98 }}
            >
              {isRetrying ? (
                <>
                  <motion.svg
                    className="w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </motion.svg>
                  Повторюю...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Спробувати ще раз
                </>
              )}
            </motion.button>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default ErrorState;
