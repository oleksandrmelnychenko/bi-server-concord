import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  StructuredResponse,
  ResponseSection,
  DataTableResponse,
  StatisticsResponse,
  ChartResponse,
  MapResponse,
  SqlResultResponse,
  GroupedResultsResponse,
  TextResponse,
  ErrorResponse,
} from '../../types/responses';
import { DataTable } from './DataTable/DataTable';
import { StatisticsGrid } from './Statistics/StatisticsGrid';
import { SmartChart } from './Charts/SmartChart';
import { RegionMap } from './Map/RegionMap';
import { SqlResultCard } from './SqlResult/SqlResultCard';
import { GroupedResults } from './GroupedResults/GroupedResults';
import { ErrorState, EmptyState } from '../ui';
import { staggerContainer, fadeInUp } from '../../utils/animations';
import type { Language } from '../WelcomeMessage';

const translations = {
  uk: {
    processing: 'Обробка запиту...',
    noResponse: 'Немає відповіді',
    noData: 'Немає даних для відображення',
    emptyResponse: 'Порожня відповідь',
    queryDoneNoData: 'Запит виконано, але дані відсутні',
    renderError: 'Помилка відображення',
    renderErrorSection: 'Не вдалося відобразити розділ',
    error: 'Помилка',
  },
  en: {
    processing: 'Processing request...',
    noResponse: 'No response',
    noData: 'No data to display',
    emptyResponse: 'Empty response',
    queryDoneNoData: 'Query completed, but no data returned',
    renderError: 'Render error',
    renderErrorSection: 'Failed to render section',
    error: 'Error',
  },
};

interface ResponseRendererProps {
  response: StructuredResponse;
  isLoading?: boolean;
  onRetry?: () => void;
  language?: Language;
  onRowClick?: (row: Record<string, unknown>) => void;
}

// Error Boundary class component
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class SectionErrorBoundary extends React.Component<
  { children: React.ReactNode; sectionType: string; language: Language },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode; sectionType: string; language: Language }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      const t = translations[this.props.language];
      return (
        <ErrorState
          title={`${t.renderError}: ${this.props.sectionType}`}
          message={t.renderErrorSection}
          details={this.state.error?.message}
          retryable={false}
        />
      );
    }
    return this.props.children;
  }
}

// Render individual section based on type
const renderSection = (
  section: ResponseSection,
  index: number,
  language: Language,
  onRowClick?: (row: Record<string, unknown>) => void
): React.ReactNode => {
  if (!section || !section.type) return null;

  const key = `section-${index}`;

  let component: React.ReactNode = null;

  switch (section.type) {
    case 'data_table':
      component = <DataTable {...(section as DataTableResponse)} language={language} onRowClick={onRowClick} />;
      break;
    case 'statistics':
      component = <StatisticsGrid {...(section as StatisticsResponse)} language={language} />;
      break;
    case 'chart':
      component = <SmartChart {...(section as ChartResponse)} language={language} />;
      break;
    case 'map':
      component = <RegionMap {...(section as MapResponse)} />;
      break;
    case 'sql_result':
      component = <SqlResultCard {...(section as SqlResultResponse)} language={language} />;
      break;
    case 'grouped_results':
      component = <GroupedResults {...(section as GroupedResultsResponse)} />;
      break;
    case 'text':
      component = <TextBlock {...(section as TextResponse)} />;
      break;
    case 'error':
      component = <ErrorBlock {...(section as ErrorResponse)} language={language} />;
      break;
    default:
      return null;
  }

  return (
    <motion.div
      key={key}
      variants={fadeInUp}
      layout
    >
      <SectionErrorBoundary sectionType={section.type} language={language}>
        {component}
      </SectionErrorBoundary>
    </motion.div>
  );
};

// Simple text block component
const TextBlock: React.FC<TextResponse> = ({ content, variant = 'default' }) => {
  const variantConfig = {
    default: {
      bg: 'bg-white',
      border: 'border-slate-200',
      text: 'text-slate-800',
      icon: null,
    },
    success: {
      bg: 'bg-emerald-50',
      border: 'border-emerald-200',
      text: 'text-emerald-800',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
    warning: {
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      text: 'text-amber-800',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
        </svg>
      ),
    },
    error: {
      bg: 'bg-rose-50',
      border: 'border-rose-200',
      text: 'text-rose-800',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
    info: {
      bg: 'bg-violet-50',
      border: 'border-violet-200',
      text: 'text-violet-800',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
        </svg>
      ),
    },
  };

  const config = variantConfig[variant] || variantConfig.default;

  return (
    <motion.div
      className={`p-4 rounded-xl border ${config.bg} ${config.border}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className={`flex gap-3 ${config.text}`}>
        {config.icon && (
          <div className="flex-shrink-0 mt-0.5">
            {config.icon}
          </div>
        )}
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      </div>
    </motion.div>
  );
};

// Error block component (uses ErrorState)
const ErrorBlock: React.FC<ErrorResponse & { onRetry?: () => void; language?: Language }> = ({
  title,
  message,
  details,
  retryable,
  onRetry,
  language = 'uk',
}) => {
  const t = translations[language];
  return (
    <ErrorState
      title={title || t.error}
      message={message}
      details={details}
      retryable={retryable}
      onRetry={onRetry}
    />
  );
};

// Main ResponseRenderer component
export const ResponseRenderer: React.FC<ResponseRendererProps> = ({
  response,
  isLoading = false,
  onRetry,
  language = 'uk',
  onRowClick,
}) => {
  const t = translations[language];

  if (isLoading) {
    return (
      <motion.div
        className="flex items-center justify-center py-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="flex flex-col items-center gap-4">
          <motion.div
            className="w-10 h-10 border-2 border-violet-400/40 border-t-violet-600 rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          />
          <span className="text-sm text-slate-600">{t.processing}</span>
        </div>
      </motion.div>
    );
  }

  if (!response) {
    return (
      <EmptyState
        icon="data"
        title={t.noResponse}
        description={t.noData}
      />
    );
  }

  const sections = Array.isArray(response.sections) ? response.sections : [];

  if (sections.length === 0) {
    return (
      <EmptyState
        icon="data"
        title={t.emptyResponse}
        description={t.queryDoneNoData}
      />
    );
  }

  return (
    <motion.div
      className="response-renderer space-y-4"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <AnimatePresence mode="popLayout">
        {sections.map((section, index) => {
          try {
            return renderSection(section, index, language, onRowClick);
          } catch (err) {
            console.error('Error rendering section:', section, err);
            return (
              <motion.div
                key={`error-${index}`}
                variants={fadeInUp}
              >
                <ErrorState
                  title={t.renderError}
                  message={`${t.renderErrorSection} ${index + 1}`}
                  details={err instanceof Error ? err.message : String(err)}
                  retryable={!!onRetry}
                  onRetry={onRetry}
                />
              </motion.div>
            );
          }
        })}
      </AnimatePresence>
    </motion.div>
  );
};

export default ResponseRenderer;
