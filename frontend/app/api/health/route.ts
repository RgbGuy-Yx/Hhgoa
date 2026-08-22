import { NextResponse } from "next/server";

// Use a server-only env var (no NEXT_PUBLIC_ prefix) for the backend URL.
// This runs server-side only — the browser never sees the Render URL.
// Set BACKEND_URL in Vercel environment variables to https://hhgoa-f93l.onrender.com
const BACKEND_URL =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ status: "down" }, { status: 503 });
  }
}
