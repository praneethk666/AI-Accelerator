/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        slacc: {
          primary: '#4a154b',
          'primary-deep': '#481a54',
          'primary-press': '#611f69',
          'primary-tint': '#592466',
          'link-blue': '#1264a3',
          'link-hover': '#3860be',
          canvas: '#ffffff',
          'canvas-cream': '#f4ede4',
          'canvas-lavender': '#f9f0ff',
          'surface-aubergine': '#4a154b',
          hairline: '#e6e6e6',
          ink: '#1d1d1d',
          'ink-mute': '#696969',
          'on-primary': '#ffffff',
          'on-aubergine-mute': '#d9bdde',
          'semantic-error': '#cc4117',
          'semantic-success': '#007a5a',
        },
        dark: "#0f172a",
        card: "#ffffff",
        accent: "#6366f1",
        success: "#10b981",
        warning: "#f59e0b",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      borderRadius: {
        pill: "90px",
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}