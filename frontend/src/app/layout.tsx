import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Resonance AI — Intelligent Music Recommendation & Adaptive Shuffle",
  description:
    "Deep Learning music personalization, Familiarity-Discovery control, and Constrained Probabilistic Adaptive Shuffle.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased selection:bg-emerald-500/30 selection:text-emerald-200">
        {children}
      </body>
    </html>
  );
}
