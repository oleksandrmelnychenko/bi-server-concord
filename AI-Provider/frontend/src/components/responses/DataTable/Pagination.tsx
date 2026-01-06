import React from 'react';
import { motion } from 'framer-motion';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalItems: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  pageSizeOptions?: number[];
  compact?: boolean;
}

export const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  pageSize,
  totalItems,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 25, 50, 100],
  compact = false,
}) => {
  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  // Generate page numbers to show
  const getPageNumbers = () => {
    const pages: (number | 'ellipsis')[] = [];
    const showPages = compact ? 3 : 5;
    const halfShow = Math.floor(showPages / 2);

    if (totalPages <= showPages + 2) {
      // Show all pages
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);

      // Calculate start and end of middle section
      let start = Math.max(2, currentPage - halfShow);
      let end = Math.min(totalPages - 1, currentPage + halfShow);

      // Adjust if at edges
      if (currentPage <= halfShow + 1) {
        end = showPages;
      } else if (currentPage >= totalPages - halfShow) {
        start = totalPages - showPages + 1;
      }

      // Add ellipsis before middle section if needed
      if (start > 2) {
        pages.push('ellipsis');
      }

      // Add middle pages
      for (let i = start; i <= end; i++) {
        pages.push(i);
      }

      // Add ellipsis after middle section if needed
      if (end < totalPages - 1) {
        pages.push('ellipsis');
      }

      // Always show last page
      if (totalPages > 1) {
        pages.push(totalPages);
      }
    }

    return pages;
  };

  const PageButton: React.FC<{
    page: number | 'ellipsis';
    isActive?: boolean;
  }> = ({ page, isActive }) => {
    if (page === 'ellipsis') {
      return (
        <span className="px-2 text-white/30">...</span>
      );
    }

    return (
      <motion.button
        onClick={() => onPageChange(page)}
        className={`
          min-w-[36px] h-9 px-3 rounded-lg text-sm font-medium transition-colors
          ${isActive
            ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
            : 'text-white/60 hover:text-white hover:bg-white/5'
          }
        `}
        whileHover={{ scale: isActive ? 1 : 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {page}
      </motion.button>
    );
  };

  const NavButton: React.FC<{
    onClick: () => void;
    disabled: boolean;
    children: React.ReactNode;
    title: string;
  }> = ({ onClick, disabled, children, title }) => (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`
        w-9 h-9 rounded-lg flex items-center justify-center transition-colors
        ${disabled
          ? 'text-white/20 cursor-not-allowed'
          : 'text-white/60 hover:text-white hover:bg-white/5'
        }
      `}
      whileHover={disabled ? {} : { scale: 1.1 }}
      whileTap={disabled ? {} : { scale: 0.9 }}
    >
      {children}
    </motion.button>
  );

  if (compact) {
    // Compact mobile view
    return (
      <div className="flex items-center justify-between gap-2 py-3 px-2">
        <span className="text-xs text-white/50">
          {startItem}-{endItem} з {totalItems}
        </span>
        <div className="flex items-center gap-1">
          <NavButton
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
            title="Попередня"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </NavButton>
          <span className="px-2 text-sm text-white/70">
            {currentPage} / {totalPages}
          </span>
          <NavButton
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            title="Наступна"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </NavButton>
        </div>
      </div>
    );
  }

  // Full desktop view
  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-3 px-4 border-t border-white/10">
      {/* Info and page size */}
      <div className="flex items-center gap-4">
        <span className="text-sm text-white/50">
          Показано {startItem}-{endItem} з {totalItems.toLocaleString('uk-UA')}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-sm text-white/50">Рядків:</span>
          <select
            value={pageSize}
            onChange={(e) => onPageSizeChange(Number(e.target.value))}
            className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-sm text-white/80 focus:outline-none focus:border-sky-500/50"
          >
            {pageSizeOptions.map((size) => (
              <option key={size} value={size} className="bg-gray-900">
                {size}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Page navigation */}
      <div className="flex items-center gap-1">
        {/* First page */}
        <NavButton
          onClick={() => onPageChange(1)}
          disabled={currentPage === 1}
          title="Перша сторінка"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </NavButton>

        {/* Previous page */}
        <NavButton
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          title="Попередня сторінка"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </NavButton>

        {/* Page numbers */}
        <div className="flex items-center gap-1 mx-2">
          {getPageNumbers().map((page, index) => (
            <PageButton
              key={page === 'ellipsis' ? `ellipsis-${index}` : page}
              page={page}
              isActive={page === currentPage}
            />
          ))}
        </div>

        {/* Next page */}
        <NavButton
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          title="Наступна сторінка"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </NavButton>

        {/* Last page */}
        <NavButton
          onClick={() => onPageChange(totalPages)}
          disabled={currentPage === totalPages}
          title="Остання сторінка"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
        </NavButton>
      </div>

      {/* Jump to page */}
      <div className="hidden lg:flex items-center gap-2">
        <span className="text-sm text-white/50">Перейти:</span>
        <input
          type="number"
          min={1}
          max={totalPages}
          value={currentPage}
          onChange={(e) => {
            const page = parseInt(e.target.value);
            if (page >= 1 && page <= totalPages) {
              onPageChange(page);
            }
          }}
          className="w-16 bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-sm text-white/80 text-center focus:outline-none focus:border-sky-500/50"
        />
      </div>
    </div>
  );
};

export default Pagination;
