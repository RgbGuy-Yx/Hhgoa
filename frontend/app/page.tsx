"use client";

import { useRef, useState, useCallback, useEffect } from "react";

// ─── types ────────────────────────────────────────────────────────────────────

interface AgentResponse {
  transcript: string;
  answer: string;
  used_retrieval: boolean;
  source_docs: string[];
  is_refusal: boolean;
  audio_b64: string | null;
  guardrail_triggered: boolean;
  guardrail_reason: string;
}

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  source_docs?: string[];
  used_retrieval?: boolean;
  is_refusal?: boolean;
  audio_b64?: string | null;
  guardrail_triggered?: boolean;
  guardrail_reason?: string;
}

type RecordingState = "idle" | "recording" | "processing";
type BackendStatus = "checking" | "up" | "down";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

let msgId = 0;
const nextId = () => ++msgId;

// ─── helpers ──────────────────────────────────────────────────────────────────

function friendlyError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  if (
    msg.toLowerCase().includes("failed to fetch") ||
    msg.toLowerCase().includes("networkerror") ||
    msg.toLowerCase().includes("load failed")
  ) {
    return (
      "Cannot reach the backend.\n\n" +
      "Start it with:\n" +
      "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    );
  }
  if (msg.includes("413")) return "Audio file too large. Try a shorter recording.";
  if (msg.includes("400")) return "Bad request — check your input and try again.";
  if (msg.includes("502") || msg.includes("503"))
    return "Backend is temporarily unavailable. Please try again in a moment.";
  return `Something went wrong: ${msg}`;
}

// ─── global audio player singleton ───────────────────────────────────────────
// Tracks the one Audio instance that may be playing at any time.
// Stopping it before playing a new one prevents overlapping voices.

type AudioListener = (playing: boolean) => void;

const audioPlayer = (() => {
  let current: HTMLAudioElement | null = null;
  const listeners = new Set<AudioListener>();

  function notify(playing: boolean) {
    listeners.forEach((fn) => fn(playing));
  }

  return {
    play(b64: string) {
      // stop whatever is currently playing
      if (current) {
        current.pause();
        current.currentTime = 0;
        current = null;
        notify(false);
      }
      const audio = new Audio(`data:audio/wav;base64,${b64}`);
      current = audio;
      notify(true);
      audio.onended = () => {
        current = null;
        notify(false);
      };
      audio.onerror = () => {
        current = null;
        notify(false);
      };
      audio.play().catch(() => {
        current = null;
        notify(false);
      });
    },
    stop() {
      if (current) {
        current.pause();
        current.currentTime = 0;
        current = null;
        notify(false);
      }
    },
    isPlaying() {
      return current !== null;
    },
    subscribe(fn: AudioListener) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
})();

// ─── sub-components ───────────────────────────────────────────────────────────

function BackendBadge({ status }: { status: BackendStatus }) {
  if (status === "checking") {
    return (
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
        <span className="text-xs text-zinc-400">connecting…</span>
      </div>
    );
  }
  if (status === "up") {
    return (
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full bg-green-400 animate-pulse" />
        <span className="text-xs text-zinc-500 dark:text-zinc-400">backend live</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full bg-red-500" />
      <span className="text-xs text-red-500 dark:text-red-400">backend offline</span>
    </div>
  );
}

function SourceBadges({ docs }: { docs: string[] }) {
  if (!docs.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1">
      {docs.map((d) => (
        <span
          key={d}
          className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300"
        >
          {d}
        </span>
      ))}
    </div>
  );
}

function PlayButton({ b64 }: { b64: string }) {
  // Track whether THIS message's audio is currently playing
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    // Subscribe to global player state changes
    const unsub = audioPlayer.subscribe((isPlaying) => {
      // If something else started playing, we're no longer "playing"
      // We rely on the play/stop calls below to set our own state correctly,
      // but if another message's button starts audio we need to reset too.
      if (!isPlaying) setPlaying(false);
    });
    return unsub;
  }, []);

  const handleClick = () => {
    if (playing) {
      audioPlayer.stop();
      setPlaying(false);
    } else {
      // Stop any other playing audio first
      audioPlayer.play(b64);
      setPlaying(true);
    }
  };

  return (
    <button
      onClick={handleClick}
      aria-label={playing ? "Stop audio" : "Play voice response"}
      className={`mt-2 flex items-center gap-1.5 text-xs transition ${
        playing
          ? "text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
          : "text-indigo-500 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-200"
      }`}
    >
      {playing ? (
        <>
          <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
          Stop
        </>
      ) : (
        <>
          <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
          Play voice response
        </>
      )}
    </button>
  );
}

function ChatBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? "rounded-br-sm bg-indigo-600 text-white"
            : msg.is_refusal
            ? "rounded-bl-sm border border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200"
            : "rounded-bl-sm bg-white text-zinc-800 dark:bg-zinc-800 dark:text-zinc-100"
        }`}
      >
        <p className="whitespace-pre-wrap">{msg.text}</p>
        {!isUser && (
          <>
            <SourceBadges docs={msg.source_docs ?? []} />
            {msg.guardrail_triggered && msg.guardrail_reason && (
              <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-400">
                ⚠ guardrail: {msg.guardrail_reason}
              </p>
            )}
            {msg.audio_b64 && <PlayButton b64={msg.audio_b64} />}
          </>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-sm bg-white px-4 py-3 shadow-sm dark:bg-zinc-800">
        <div className="flex gap-1.5 items-center h-4">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce dark:bg-zinc-500"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── main page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: nextId(),
      role: "assistant",
      text: "Hey! I'm your Hacker House Goa 2026 assistant.\nAsk me anything — dates, tasks, rules, schedule, prizes — or just say hi 👋\n\nType below or tap the mic to speak in any Indian language.",
    },
  ]);
  const [input, setInput] = useState("");
  const [recording, setRecording] = useState<RecordingState>("idle");
  const [loading, setLoading] = useState(false);
  const [autoplay, setAutoplay] = useState(true);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [audioPlaying, setAudioPlaying] = useState(false);

  // Keep audioPlaying in sync with the global player
  useEffect(() => {
    const unsub = audioPlayer.subscribe(setAudioPlaying);
    return unsub;
  }, []);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // ── health-check on mount + every 15 s ─────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    async function ping() {
      try {
        const res = await fetch(`${API_BASE}/health`, {
          signal: AbortSignal.timeout(4000),
        });
        if (!cancelled) setBackendStatus(res.ok ? "up" : "down");
      } catch {
        if (!cancelled) setBackendStatus("down");
      }
    }
    ping();
    const id = setInterval(ping, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const scrollBottom = useCallback(() => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }, []);

  const pushMessage = useCallback(
    (msg: Omit<Message, "id">) => {
      setMessages((prev) => [...prev, { ...msg, id: nextId() }]);
      scrollBottom();
    },
    [scrollBottom]
  );

  const handleResponse = useCallback(
    (data: AgentResponse) => {
      pushMessage({
        role: "assistant",
        text: data.answer,
        source_docs: data.source_docs,
        used_retrieval: data.used_retrieval,
        is_refusal: data.is_refusal,
        audio_b64: data.audio_b64,
        guardrail_triggered: data.guardrail_triggered,
        guardrail_reason: data.guardrail_reason,
      });
      if (autoplay && data.audio_b64) {
        audioPlayer.play(data.audio_b64);
      }
    },
    [pushMessage, autoplay]
  );

  // ── text submit ─────────────────────────────────────────────────────────────
  const handleTextSubmit = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    pushMessage({ role: "user", text: q });
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/ask-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, tts: true }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data: AgentResponse = await res.json();
      handleResponse(data);
      setBackendStatus("up");
    } catch (err) {
      if ((err instanceof Error) && err.message.toLowerCase().includes("failed to fetch")) {
        setBackendStatus("down");
      }
      pushMessage({ role: "assistant", text: friendlyError(err), is_refusal: true });
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [input, loading, pushMessage, handleResponse]);

  // ── voice recording ─────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    if (recording !== "idle") return;

    if (backendStatus === "down") {
      pushMessage({
        role: "assistant",
        text: friendlyError(new Error("Failed to fetch")),
        is_refusal: true,
      });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const mr = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setRecording("processing");
        setLoading(true);

        const form = new FormData();
        form.append("audio", blob, "recording.webm");

        try {
          const res = await fetch(`${API_BASE}/ask?tts=true`, {
            method: "POST",
            body: form,
          });
          if (!res.ok) throw new Error(`${res.status}`);
          const data: AgentResponse = await res.json();
          pushMessage({ role: "user", text: `🎤 ${data.transcript}` });
          handleResponse(data);
          setBackendStatus("up");
        } catch (err) {
          if ((err instanceof Error) && err.message.toLowerCase().includes("failed to fetch")) {
            setBackendStatus("down");
          }
          pushMessage({ role: "assistant", text: friendlyError(err), is_refusal: true });
          console.error(err);
        } finally {
          setLoading(false);
          setRecording("idle");
        }
      };

      mediaRecorderRef.current = mr;
      mr.start();
      setRecording("recording");
    } catch {
      alert("Microphone access denied. Please allow mic access and try again.");
    }
  }, [recording, backendStatus, pushMessage, handleResponse]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit();
    }
  };

  // ─── render ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">

      {/* ── header ── */}
      <header className="shrink-0 border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-2xl items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-indigo-600 text-white font-bold text-sm select-none">
            HG
          </div>
          <div>
            <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Hey Goa</p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              HHGoa 2026 · saaras:v3 STT · bulbul:v3 TTS
            </p>
          </div>

          <div className="ml-auto flex items-center gap-3">
            {/* backend status */}
            <BackendBadge status={backendStatus} />

            {/* global stop button — visible only while audio is playing */}
            {audioPlaying && (
              <button
                onClick={() => audioPlayer.stop()}
                aria-label="Stop audio"
                title="Stop playing"
                className="flex h-7 items-center gap-1 rounded-lg bg-red-100 px-2 text-xs font-medium text-red-600 transition hover:bg-red-200 dark:bg-red-900/40 dark:text-red-300 dark:hover:bg-red-900/60"
              >
                <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
                Stop
              </button>
            )}

            {/* autoplay toggle */}
            <button
              onClick={() => {
                setAutoplay((v) => !v);
                if (audioPlaying) audioPlayer.stop();
              }}
              aria-label={autoplay ? "Mute voice responses" : "Unmute voice responses"}
              title={autoplay ? "Voice on — click to mute" : "Voice off — click to unmute"}
              className={`flex h-7 w-7 items-center justify-center rounded-lg transition ${
                autoplay
                  ? "bg-indigo-100 text-indigo-600 dark:bg-indigo-900/40 dark:text-indigo-300"
                  : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
              }`}
            >
              {autoplay ? (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M12 6v12m0 0l-3-3m3 3l3-3M9 9H5a2 2 0 00-2 2v2a2 2 0 002 2h4l5 5V4L9 9z" />
                </svg>
              ) : (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* offline banner */}
        {backendStatus === "down" && (
          <div className="mx-auto mt-2 max-w-2xl rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-300">
            ⚠ Backend offline — run:{" "}
            <code className="font-mono">
              python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
            </code>
          </div>
        )}
      </header>

      {/* ── messages ── */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-4">
          {messages.map((msg) => (
            <ChatBubble key={msg.id} msg={msg} />
          ))}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── input bar ── */}
      <div className="shrink-0 border-t border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-2xl items-end gap-2">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              backendStatus === "down"
                ? "Backend offline — start the server first"
                : "Ask about HHGoa 2026… (Enter to send)"
            }
            disabled={loading || recording !== "idle" || backendStatus === "down"}
            className="flex-1 resize-none rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-2.5 text-sm text-zinc-900 placeholder-zinc-400 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:placeholder-zinc-500 dark:focus:ring-indigo-900/40"
            style={{ maxHeight: "8rem", overflowY: "auto" }}
          />

          {/* send */}
          <button
            onClick={handleTextSubmit}
            disabled={!input.trim() || loading || recording !== "idle" || backendStatus === "down"}
            aria-label="Send message"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg className="h-4 w-4 rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          </button>

          {/* mic */}
          <button
            onClick={recording === "recording" ? stopRecording : startRecording}
            disabled={(loading && recording === "idle") || backendStatus === "down"}
            aria-label={recording === "recording" ? "Stop recording" : "Start voice input"}
            className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl transition ${
              recording === "recording"
                ? "bg-red-500 text-white animate-pulse hover:bg-red-600"
                : recording === "processing"
                ? "bg-amber-400 text-white cursor-wait"
                : "border border-zinc-200 bg-zinc-50 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
            } disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {recording === "processing" ? (
              <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={4} />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" />
              </svg>
            )}
          </button>
        </div>

        {recording === "recording" && (
          <p className="mt-2 text-center text-xs text-red-500 dark:text-red-400">
            🔴 Recording… tap mic again to stop and send
          </p>
        )}
      </div>
    </div>
  );
}
