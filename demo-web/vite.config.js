import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    /** Keep 5173 fixed so bookmarks match the dev proxy (avoids stale ports returning HTML). */
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_API ?? "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_API ?? "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
