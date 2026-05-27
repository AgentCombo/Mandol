/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}', './docs/**/*.mdx', './blog/**/*.mdx'],
  theme: {
    extend: {
      colors: {
        primary: {
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        accent: {
          500: '#06b6d4',
        },
      },
    },
  },
  plugins: [],
};
