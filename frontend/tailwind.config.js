/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ai: {
          bg: '#05080F',
          surface: '#1F2B2D',
          card: '#23717B',
          accent: '#0D8A9E',
          highlight: '#12B2C1',
          text: '#E5F9F8',
          muted: '#A8CFCF',
        },
      },
      boxShadow: {
        'ai-glow': '0 0 18px rgba(18,178,193,0.6)',
      },
    },
  },
  plugins: [],
}
