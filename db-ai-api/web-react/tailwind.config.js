/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
        },
        surface: {
          primary: '#0f1218',
          secondary: '#151a22',
          tertiary: '#1b2230',
          hover: '#202838',
          border: 'rgba(148, 163, 184, 0.18)',
        },
        content: {
          primary: '#e2e8f0',
          secondary: '#cbd5e1',
          muted: '#94a3b8',
          light: '#64748b',
        },
      },
      fontFamily: {
        sans: ['"Space Grotesk"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        'grok': '0 12px 30px rgba(0, 0, 0, 0.25)',
        'grok-lg': '0 20px 50px rgba(0, 0, 0, 0.35)',
        'grok-input': '0 0 0 1px rgba(125, 211, 252, 0.4)',
      },
      backgroundImage: {
        'gradient-grok': 'radial-gradient(1200px 500px at 10% -10%, rgba(56, 189, 248, 0.2), transparent 60%)',
        'gradient-accent': 'linear-gradient(135deg, #38bdf8 0%, #22d3ee 50%, #0ea5e9 100%)',
      },
      animation: {
        'bounce-dot': 'bounce-dot 1.4s infinite ease-in-out both',
        'fade-in': 'fade-in 0.4s ease-out both',
        'slide-up': 'slide-up 0.35s ease-out both',
        'pulse-soft': 'pulse-soft 2.4s ease-in-out infinite',
      },
      keyframes: {
        'bounce-dot': {
          '0%, 80%, 100%': { transform: 'scale(0.8)', opacity: '0.45' },
          '40%': { transform: 'scale(1)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(18px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
    },
  },
  plugins: [],
}
