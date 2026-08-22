import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

// Proxy POST /api/ask → backend /ask
// Forwards the multipart audio blob server-side so the browser only
// ever POSTs to its own origin — avoids CORS and ad-blocker blocks.
export async function POST(req: NextRequest) {
  try {
    const tts = req.nextUrl.searchParams.get("tts") ?? "true";
    const body = await req.blob();

    const res = await fetch(`${BACKEND_URL}/ask?tts=${tts}`, {
      method: "POST",
      body,
      headers: {
        // Forward content-type (multipart/form-data with boundary)
        "content-type": req.headers.get("content-type") ?? "application/octet-stream",
      },
      signal: AbortSignal.timeout(60000), // voice pipeline can take a few seconds
    });

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    return NextResponse.json(
      { error: String(err) },
      { status: 502 }
    );
  }
}
