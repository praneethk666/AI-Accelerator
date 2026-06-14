/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: "#1e1e2e",
        card: "#2d2d44",
        accent: "#6c63ff",
        success: "#4caf50",
        warning: "#ff9800",
      },
    },
  },
  plugins: [],
}