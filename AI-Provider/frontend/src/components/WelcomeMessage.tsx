import React from 'react';

export type Language = 'uk' | 'en';

interface WelcomeMessageProps {
  onQuickQuery: (query: string) => void;
  language?: Language;
}

const suggestionsEn = [
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M4 19h16" />
        <path d="M6 16V8" />
        <path d="M10 16V5" />
        <path d="M14 16v-6" />
        <path d="M18 16v-3" />
      </svg>
    ),
    title: 'Revenue pulse',
    description: 'Show yearly sales as a chart',
    query: 'Show yearly sales as a chart',
  },
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M3 9h18" />
        <path d="M9 9v11" />
      </svg>
    ),
    title: 'Top products',
    description: 'List the top 10 products as a table',
    query: 'Top 10 products as a table',
  },
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M16 11a4 4 0 1 0-8 0" />
        <path d="M6 20a6 6 0 0 1 12 0" />
      </svg>
    ),
    title: 'Best customers',
    description: 'Top customers by revenue in a table',
    query: 'Top customers by revenue in a table',
  },
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M12 6v6l4 2" />
        <path d="M4.5 19a9 9 0 1 0 0-14" />
      </svg>
    ),
    title: 'Debt snapshot',
    description: 'Debt summary chart',
    query: 'Debt summary chart',
  },
];

const suggestionsUk = [
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M4 19h16" />
        <path d="M6 16V8" />
        <path d="M10 16V5" />
        <path d="M14 16v-6" />
        <path d="M18 16v-3" />
      </svg>
    ),
    title: 'Динаміка продажів',
    description: 'Покажи продажі по роках',
    query: 'Покажи продажі по роках',
  },
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M3 9h18" />
        <path d="M9 9v11" />
      </svg>
    ),
    title: 'Топ товарів',
    description: 'Топ 10 товарів по продажах',
    query: 'Топ 10 товарів по продажах',
  },
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M16 11a4 4 0 1 0-8 0" />
        <path d="M6 20a6 6 0 0 1 12 0" />
      </svg>
    ),
    title: 'Найкращі клієнти',
    description: 'Топ клієнтів по виручці',
    query: 'Топ клієнтів по виручці',
  },
  {
    icon: (
      <svg className="w-5 h-5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M12 6v6l4 2" />
        <path d="M4.5 19a9 9 0 1 0 0-14" />
      </svg>
    ),
    title: 'Борги',
    description: 'Загальна сума боргів',
    query: 'Загальна сума боргів',
  },
];

const translations = {
  en: {
    heading: 'Ask anything about your business data.',
    subheading: 'Get fast answers, charts, and tables powered by AI.',
  },
  uk: {
    heading: 'Запитайте будь-що про ваші бізнес-дані.',
    subheading: 'Отримуйте швидкі відповіді, графіки та таблиці на основі ШІ.',
  },
};

export const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ onQuickQuery, language = 'uk' }) => {
  const suggestions = language === 'uk' ? suggestionsUk : suggestionsEn;
  const t = translations[language];

  return (
    <div className="flex flex-col items-center justify-center min-h-[65vh] py-10 text-center text-slate-100">
      {/* Emblem */}
      <div className="mb-6">
        <div className="w-16 h-16 rounded-2xl border border-sky-400/40 bg-white/5 flex items-center justify-center shadow-lg">
          <svg className="w-7 h-7 text-sky-300 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M12 4.5l1.7 4.2L18 10l-4.3 1.3L12 16.5l-1.7-4.2L6 10l4.3-1.3L12 4.5z" />
          </svg>
        </div>
      </div>

      {/* Heading */}
      <h1 className="text-3xl sm:text-4xl font-semibold text-white mb-4">
        {t.heading}
      </h1>
      <p className="text-slate-400 text-sm sm:text-base max-w-xl mb-10">
        {t.subheading}
      </p>

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl">
        {suggestions.map((item, index) => (
          <button
            key={item.query}
            onClick={() => onQuickQuery(item.query)}
            className="rounded-2xl border border-white/10 bg-white/5 flex items-start gap-3 p-4 text-left text-sm text-slate-100
                       hover:border-sky-300/50 hover:bg-white/10 transition-colors animate-rise shadow-lg"
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <div className="flex-shrink-0 w-10 h-10 rounded-xl border border-sky-400/40 bg-white/5 text-sky-200 flex items-center justify-center">
              {item.icon}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-medium text-slate-100">{item.title}</h3>
              <p className="text-xs text-slate-400 mt-1">{item.description}</p>
            </div>
            <svg className="w-4 h-4 text-slate-400 thin-icon mt-1" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M8 5l8 7-8 7" />
            </svg>
          </button>
        ))}
      </div>
    </div>
  );
};
