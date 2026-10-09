import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function readPort(raw: string | undefined, name: string, fallback: number): number {
  if (!raw) return fallback;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${name} must be an integer from 1 to 65535, got ${raw}`);
  }
  return port;
}

export default defineConfig(({ mode }) => {
  const fileEnv = loadEnv(mode, process.cwd(), "");
  const port = readPort(process.env.VITE_PORT || fileEnv.VITE_PORT, "VITE_PORT", 5180);
  const target = process.env.VITE_API_TARGET || fileEnv.VITE_API_TARGET || "http://127.0.0.1:8100";
  return {
    plugins: [react()],
    server: {
      port,
      strictPort: true,
      proxy: { "/api": target },
    },
  };
});
