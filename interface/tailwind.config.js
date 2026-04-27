/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        emerald: {
          DEFAULT: '#10b981',
          dim: 'rgba(16,185,129,0.12)',
        },
      },
    },
  },
  plugins: [],
};
