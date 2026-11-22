/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        mist: '#F4F7FA',
        navy: '#17394D',
        slateText: '#475569',
        tealAccent: '#0D9488',
        indigoAccent: '#4F46E5',
        amberAccent: '#D97706',
      },
      boxShadow: {
        card: '0 4px 20px -2px rgba(23, 57, 77, 0.06), 0 1px 3px 0 rgba(23, 57, 77, 0.04)',
        'card-hover': '0 14px 30px -6px rgba(23, 57, 77, 0.12), 0 4px 10px -2px rgba(23, 57, 77, 0.05)',
        drawer: '-10px 0 40px rgba(23, 57, 77, 0.14)',
      },
    },
  },
  plugins: [],
};
