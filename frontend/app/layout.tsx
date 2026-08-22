import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HHGoa 2026 — Voice & Intelligence Assistant",
  description: "Editorial voice & knowledge assistant for Hacker House Goa 2026",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-black text-[#fcfdff] selection:bg-white/20 selection:text-white font-sans">
        {children}
      </body>
    </html>
  );
}
