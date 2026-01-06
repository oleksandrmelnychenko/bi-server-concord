import React from 'react';
import { motion } from 'framer-motion';

interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  animate?: boolean;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  width,
  height,
  rounded = 'md',
  animate = true,
}) => {
  const roundedClasses = {
    none: 'rounded-none',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    full: 'rounded-full',
  };

  const style: React.CSSProperties = {
    width: width,
    height: height,
  };

  if (animate) {
    return (
      <motion.div
        className={`bg-white/5 ${roundedClasses[rounded]} ${className}`}
        style={style}
        animate={{
          opacity: [0.5, 0.8, 0.5],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    );
  }

  return (
    <div
      className={`bg-white/5 animate-pulse ${roundedClasses[rounded]} ${className}`}
      style={style}
    />
  );
};

// Table Skeleton
interface TableSkeletonProps {
  rows?: number;
  columns?: number;
  showHeader?: boolean;
}

export const TableSkeleton: React.FC<TableSkeletonProps> = ({
  rows = 5,
  columns = 4,
  showHeader = true,
}) => {
  return (
    <div className="w-full overflow-hidden rounded-lg border border-white/10">
      {showHeader && (
        <div className="flex gap-4 p-4 bg-white/5 border-b border-white/10">
          {Array.from({ length: columns }).map((_, i) => (
            <Skeleton key={`header-${i}`} height={20} className="flex-1" />
          ))}
        </div>
      )}
      <div className="divide-y divide-white/5">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <motion.div
            key={`row-${rowIndex}`}
            className="flex gap-4 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: rowIndex * 0.05 }}
          >
            {Array.from({ length: columns }).map((_, colIndex) => (
              <Skeleton
                key={`cell-${rowIndex}-${colIndex}`}
                height={16}
                className="flex-1"
              />
            ))}
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// Chart Skeleton
interface ChartSkeletonProps {
  type?: 'bar' | 'line' | 'pie';
  height?: number;
}

export const ChartSkeleton: React.FC<ChartSkeletonProps> = ({
  type = 'bar',
  height = 300,
}) => {
  if (type === 'pie') {
    return (
      <div className="flex items-center justify-center p-8" style={{ height }}>
        <Skeleton width={200} height={200} rounded="full" />
      </div>
    );
  }

  if (type === 'line') {
    return (
      <div className="p-4" style={{ height }}>
        <div className="h-full flex items-end gap-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="flex-1 flex flex-col justify-end">
              <motion.div
                className="w-full bg-white/10 rounded-t"
                initial={{ height: 0 }}
                animate={{ height: `${30 + Math.random() * 60}%` }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Bar chart skeleton
  return (
    <div className="p-4" style={{ height }}>
      <div className="h-full flex items-end gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <motion.div
            key={i}
            className="flex-1 bg-white/10 rounded-t"
            initial={{ height: 0 }}
            animate={{ height: `${20 + Math.random() * 70}%` }}
            transition={{ delay: i * 0.1, duration: 0.5 }}
          />
        ))}
      </div>
    </div>
  );
};

// Card Skeleton
interface CardSkeletonProps {
  showIcon?: boolean;
  showTrend?: boolean;
}

export const CardSkeleton: React.FC<CardSkeletonProps> = ({
  showIcon = true,
  showTrend = true,
}) => {
  return (
    <div className="p-4 rounded-xl bg-white/5 border border-white/10">
      <div className="flex items-start justify-between mb-3">
        {showIcon && <Skeleton width={40} height={40} rounded="lg" />}
        {showTrend && <Skeleton width={60} height={20} rounded="full" />}
      </div>
      <Skeleton width="60%" height={32} className="mb-2" />
      <Skeleton width="80%" height={16} />
    </div>
  );
};

// Statistics Grid Skeleton
interface StatisticsGridSkeletonProps {
  cards?: number;
  columns?: 2 | 3 | 4;
}

export const StatisticsGridSkeleton: React.FC<StatisticsGridSkeletonProps> = ({
  cards = 4,
  columns = 4,
}) => {
  const gridCols = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={`grid ${gridCols[columns]} gap-4`}>
      {Array.from({ length: cards }).map((_, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1 }}
        >
          <CardSkeleton />
        </motion.div>
      ))}
    </div>
  );
};

export default Skeleton;
