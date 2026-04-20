/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Neutral grayscale palette with a single accent for live/accepted.
        ink: {
          50: "#f7f7f8",
          100: "#eceef0",
          200: "#d4d7dc",
          300: "#9ea3ad",
          400: "#6b7078",
          500: "#474a52",
          600: "#2f3238",
          700: "#1f2126",
          800: "#141619",
          900: "#0b0c0e",
        },
        accent: {
          500: "#4ade80",
          600: "#22c55e",
        },
        warn: { 500: "#f59e0b" },
        bad: { 500: "#ef4444" },
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
