import React from 'react';
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

interface ResponseRendererProps {
  response: StructuredResponse;
}

// Error Boundary class component
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class SectionErrorBoundary extends React.Component<
  { children: React.ReactNode; sectionType: string },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode; sectionType: string }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 rounded-xl border border-rose-200 bg-rose-50">
          <h4 className="font-semibold text-rose-700 mb-2">
            Failed to render section: {this.props.sectionType}
          </h4>
          <p className="text-rose-600 text-sm">{this.state.error?.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

// Render individual section based on type
const renderSection = (section: ResponseSection, index: number): React.ReactNode => {
  if (!section || !section.type) return null;

  const key = `section-${index}`;

  let component: React.ReactNode = null;

  switch (section.type) {
    case 'data_table':
      component = <DataTable {...(section as DataTableResponse)} />;
      break;
    case 'statistics':
      component = <StatisticsGrid {...(section as StatisticsResponse)} />;
      break;
    case 'chart':
      component = <SmartChart {...(section as ChartResponse)} />;
      break;
    case 'map':
      component = <RegionMap {...(section as MapResponse)} />;
      break;
    case 'sql_result':
      component = <SqlResultCard {...(section as SqlResultResponse)} />;
      break;
    case 'grouped_results':
      component = <GroupedResults {...(section as GroupedResultsResponse)} />;
      break;
    case 'text':
      component = <TextBlock {...(section as TextResponse)} />;
      break;
    case 'error':
      component = <ErrorBlock {...(section as ErrorResponse)} />;
      break;
    default:
      return null;
  }

  return (
    <SectionErrorBoundary key={key} sectionType={section.type}>
      {component}
    </SectionErrorBoundary>
  );
};

// Simple text block component
const TextBlock: React.FC<TextResponse> = ({ content, variant = 'default' }) => {
  if (variant === 'default') {
    return (
      <div className="text-response">
        <p className="whitespace-pre-wrap">{content}</p>
      </div>
    );
  }

  const variantClasses = {
    default: '',
    success: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    warning: 'text-amber-700 bg-amber-50 border-amber-200',
    error: 'text-rose-700 bg-rose-50 border-rose-200',
    info: 'text-sky-700 bg-sky-50 border-sky-200',
  };

  return (
    <div className={`p-3 rounded-xl border text-sm leading-relaxed ${variantClasses[variant]}`}>
      <p className="whitespace-pre-wrap">{content}</p>
    </div>
  );
};

// Error block component
const ErrorBlock: React.FC<ErrorResponse> = ({ title, message, details, retryable }) => {
  return (
    <div className="p-4 rounded-xl border border-rose-200 bg-rose-50">
      {title && (
        <h4 className="font-semibold text-rose-700 mb-2 flex items-center gap-2">
          <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M12 9v4" />
            <path d="M12 17h.01" />
            <circle cx="12" cy="12" r="9" />
          </svg>
          {title}
        </h4>
      )}
      <p className="text-rose-600">{message}</p>
      {details && (
        <details className="mt-2">
          <summary className="text-sm text-rose-500 cursor-pointer hover:text-rose-700">
            Show details
          </summary>
          <pre className="mt-2 p-3 bg-rose-100 rounded text-xs overflow-x-auto text-rose-800">
            {details}
          </pre>
        </details>
      )}
      {retryable && (
        <button className="mt-3 px-4 py-2 bg-rose-500 text-white rounded-lg hover:bg-rose-600 transition-colors">
          Try again
        </button>
      )}
    </div>
  );
};

// Main ResponseRenderer component
export const ResponseRenderer: React.FC<ResponseRendererProps> = ({ response }) => {
  if (!response) {
    return (
      <div className="text-slate-400 italic p-4">
        No response available.
      </div>
    );
  }

  const sections = Array.isArray(response.sections) ? response.sections : [];

  if (sections.length === 0) {
    return (
      <div className="text-slate-400 italic p-4">
        No response available.
      </div>
    );
  }

  return (
    <div className="response-renderer space-y-4">
      {sections.map((section, index) => {
        try {
          return renderSection(section, index);
        } catch (err) {
          console.error('Error rendering section:', section, err);
          return (
            <div key={`error-${index}`} className="p-4 rounded-xl border border-rose-200 bg-rose-50">
              <p className="text-rose-600">Failed to render section {index + 1}.</p>
            </div>
          );
        }
      })}
    </div>
  );
};

export default ResponseRenderer;
