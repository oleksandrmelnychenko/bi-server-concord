import React from 'react';

interface WelcomeMessageProps {
  onQuickQuery: (query: string) => void;
}

const welcomeExamples = [
  { label: 'Найкращий клієнт', query: 'найкращий клієнт хто купив найбільше' },
  { label: 'Найпопулярніший товар', query: 'який товар продали найбільше' },
  { label: 'Клієнти з Хмельницького', query: 'клієнт із хмельницького' },
];

export const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ onQuickQuery }) => {
  return (
    <div className="max-w-2xl mx-auto text-center py-16 animate-slide-up">
      {/* Icon */}
      <div className="mb-8">
        <div className="w-20 h-20 mx-auto rounded-full bg-content-primary
                      flex items-center justify-center
                      transform hover:scale-105 transition-transform duration-300">
          <svg className="w-10 h-10 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
      </div>

      {/* Heading */}
      <h2 className="text-4xl font-bold mb-4 text-content-primary">
        Вітаю! Я Concord AI
      </h2>

      <p className="text-lg text-content-secondary mb-10 max-w-md mx-auto">
        Ваш асистент для бізнес-аналітики. Задайте питання про продажі, клієнтів, товари або борги.
      </p>

      {/* Example Buttons */}
      <div className="flex flex-wrap justify-center gap-3">
        {welcomeExamples.map((example, index) => (
          <button
            key={example.query}
            onClick={() => onQuickQuery(example.query)}
            className="px-5 py-2.5 rounded-full text-sm font-medium
                     bg-surface-secondary border border-surface-border text-content-secondary
                     hover:border-gray-300 hover:bg-surface-hover
                     transition-all duration-200
                     animate-fade-in"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            {example.label}
          </button>
        ))}
      </div>
    </div>
  );
};
