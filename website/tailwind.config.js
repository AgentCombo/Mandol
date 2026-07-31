/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}', './docs/**/*.mdx', './blog/**/*.mdx'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        // Blue-cyan accent scale
        primary: {
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
        },
        accent: {
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
        },
        // Deep blue backgrounds
        deep: {
          900: '#060f1a',
          800: '#0a1628',
          700: '#0f2038',
          600: '#122544',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'glow-pulse': 'glowPulse 2.5s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'count-up': 'countUp 0.3s ease-out forwards',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)', filter: 'blur(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)', filter: 'blur(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 8px rgba(59, 130, 246, 0.3), 0 0 20px rgba(59, 130, 246, 0.1)' },
          '50%': { boxShadow: '0 0 16px rgba(59, 130, 246, 0.5), 0 0 40px rgba(59, 130, 246, 0.2)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backgroundImage: {
        'hero-glow': 'radial-gradient(ellipse 80% 60% at 30% 20%, rgba(59, 130, 246, 0.12) 0%, transparent 60%), radial-gradient(ellipse 60% 50% at 70% 60%, rgba(6, 182, 212, 0.08) 0%, transparent 60%)',
        'card-glow': 'radial-gradient(ellipse at top, rgba(59, 130, 246, 0.06) 0%, transparent 70%)',
      },
    },
  },
  plugins: [],
};
