/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: { deep: '#050510', card: '#0a0a20', raised: '#0f0f28' },
        accent: { cyan: '#00d4ff', violet: '#7b2fff', green: '#00ff9d', amber: '#ffb700', danger: '#ff4d6d' },
      },
      fontFamily: {
        sans: ['Syne', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
