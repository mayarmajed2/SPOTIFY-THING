import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

function copyManifestPlugin() {
  return {
    name: "copy-extension-manifest",
    closeBundle() {
      const distDir = resolve(__dirname, "dist");
      if (!existsSync(distDir)) {
        mkdirSync(distDir, { recursive: true });
      }
      copyFileSync(resolve(__dirname, "manifest.json"), resolve(distDir, "manifest.json"));
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), copyManifestPlugin()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      input: {
        popup: resolve(__dirname, "popup.html"),
        "service-worker": resolve(__dirname, "src/background/service-worker.ts"),
      },
      output: {
        entryFileNames: (chunkInfo) =>
          chunkInfo.name === "service-worker" ? "service-worker.js" : "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
