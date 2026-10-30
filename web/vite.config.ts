import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs separately (qanun serve, port 8000); Vite
// forwards API paths to it so the browser sees a single origin.
const api = process.env.QANUN_API ?? "http://127.0.0.1:8000";
const apiPaths = ["/ask", "/search", "/articles", "/laws", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(apiPaths.map((path) => [path, api])),
  },
});
