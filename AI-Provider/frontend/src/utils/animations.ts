import { Variants } from 'framer-motion';

// Fade in with upward movement
export const fadeInUp: Variants = {
  initial: {
    opacity: 0,
    y: 20,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: {
      duration: 0.2,
    },
  },
};

// Fade in with scale
export const scaleIn: Variants = {
  initial: {
    opacity: 0,
    scale: 0.9,
  },
  animate: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    transition: {
      duration: 0.2,
    },
  },
};

// Slide in from right
export const slideInFromRight: Variants = {
  initial: {
    opacity: 0,
    x: 30,
  },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    x: -20,
    transition: {
      duration: 0.2,
    },
  },
};

// Slide in from left
export const slideInFromLeft: Variants = {
  initial: {
    opacity: 0,
    x: -30,
  },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    x: 20,
    transition: {
      duration: 0.2,
    },
  },
};

// Container for staggered children
export const staggerContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.05,
    },
  },
  exit: {
    transition: {
      staggerChildren: 0.05,
      staggerDirection: -1,
    },
  },
};

// Fast stagger for lists
export const fastStaggerContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.02,
    },
  },
};

// List item animation
export const listItem: Variants = {
  initial: {
    opacity: 0,
    x: -10,
  },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.2,
    },
  },
  exit: {
    opacity: 0,
    x: 10,
    transition: {
      duration: 0.15,
    },
  },
};

// Card hover effect
export const cardHover: Variants = {
  initial: {
    scale: 1,
  },
  hover: {
    scale: 1.02,
    transition: {
      duration: 0.2,
      ease: 'easeInOut',
    },
  },
  tap: {
    scale: 0.98,
  },
};

// Pulse animation for highlighting
export const pulse: Variants = {
  initial: {
    scale: 1,
  },
  animate: {
    scale: [1, 1.05, 1],
    transition: {
      duration: 0.6,
      repeat: Infinity,
      repeatDelay: 2,
    },
  },
};

// Shimmer effect for loading states
export const shimmer: Variants = {
  initial: {
    backgroundPosition: '-200% 0',
  },
  animate: {
    backgroundPosition: '200% 0',
    transition: {
      duration: 1.5,
      repeat: Infinity,
      ease: 'linear',
    },
  },
};

// Flip animation for cards
export const flipCard: Variants = {
  initial: {
    rotateY: 0,
  },
  flipped: {
    rotateY: 180,
    transition: {
      duration: 0.6,
      ease: 'easeInOut',
    },
  },
};

// Chart bar animation
export const chartBar: Variants = {
  initial: {
    scaleY: 0,
    originY: 1,
  },
  animate: (i: number) => ({
    scaleY: 1,
    transition: {
      duration: 0.5,
      delay: i * 0.1,
      ease: 'easeOut',
    },
  }),
};

// Table row animation
export const tableRow: Variants = {
  initial: {
    opacity: 0,
    backgroundColor: 'rgba(255, 255, 255, 0)',
  },
  animate: {
    opacity: 1,
    transition: {
      duration: 0.2,
    },
  },
  hover: {
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
  },
  exit: {
    opacity: 0,
    transition: {
      duration: 0.15,
    },
  },
};

// Expand/collapse animation
export const expandCollapse: Variants = {
  initial: {
    height: 0,
    opacity: 0,
  },
  animate: {
    height: 'auto',
    opacity: 1,
    transition: {
      height: {
        duration: 0.3,
        ease: 'easeOut',
      },
      opacity: {
        duration: 0.2,
        delay: 0.1,
      },
    },
  },
  exit: {
    height: 0,
    opacity: 0,
    transition: {
      height: {
        duration: 0.2,
      },
      opacity: {
        duration: 0.1,
      },
    },
  },
};

// Notification/toast animation
export const notification: Variants = {
  initial: {
    opacity: 0,
    y: -20,
    scale: 0.95,
  },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    y: -10,
    scale: 0.95,
    transition: {
      duration: 0.2,
    },
  },
};

// Page transition
export const pageTransition: Variants = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
    transition: {
      duration: 0.3,
      when: 'beforeChildren',
    },
  },
  exit: {
    opacity: 0,
    transition: {
      duration: 0.2,
      when: 'afterChildren',
    },
  },
};

// Spring animation settings
export const springConfig = {
  soft: { type: 'spring', stiffness: 100, damping: 20 },
  medium: { type: 'spring', stiffness: 200, damping: 25 },
  stiff: { type: 'spring', stiffness: 300, damping: 30 },
  bouncy: { type: 'spring', stiffness: 400, damping: 10 },
};

// Utility function to create delayed animation
export const withDelay = (variants: Variants, delay: number): Variants => ({
  ...variants,
  animate: {
    ...variants.animate,
    transition: {
      ...(typeof variants.animate === 'object' && 'transition' in variants.animate
        ? variants.animate.transition
        : {}),
      delay,
    },
  },
});

// Utility function to create staggered item
export const createStaggerItem = (index: number, baseDelay = 0.05): Variants => ({
  initial: { opacity: 0, y: 10 },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.3,
      delay: index * baseDelay,
    },
  },
});
