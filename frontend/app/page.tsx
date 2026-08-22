"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";

/* ─── Types & Models ─────────────────────────────────────────────────────── */

interface AgentResponse {
  transcript: string;
  answer: string;
  used_retrieval: boolean;
  source_docs?: string[];
  is_refusal?: boolean;
  audio_b64?: string | null;
  guardrail_triggered?: boolean;
  guardrail_reason?: string;
  language?: string;
}

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  isVoice?: boolean;
  timestamp: string;
  source_docs?: string[];
  is_refusal?: boolean;
  audio_b64?: string | null;
  guardrail_triggered?: boolean;
  guardrail_reason?: string;
}

type RecordingState = "idle" | "recording" | "processing";
type BackendStatus = "checking" | "up" | "down";
type TabView = "landing" | "chat" | "specs";

// All API calls go through Next.js proxy routes (/api/ask, /api/ask-text).
// In dev: proxy routes forward to localhost:8000 via BACKEND_URL env var.
// In prod: proxy routes forward to Render via BACKEND_URL set in Vercel.
// The browser ALWAYS calls same-origin /api/* — never hits Render directly.
// This solves ERR_BLOCKED_BY_CLIENT from Brave/uBlock on cross-origin fetches.
let globalMsgCounter = 0;
const nextMsgId = () => ++globalMsgCounter;

function getTimestamp(): string {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function parseFriendlyError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  if (
    msg.toLowerCase().includes("failed to fetch") ||
    msg.toLowerCase().includes("networkerror") ||
    msg.toLowerCase().includes("load failed")
  ) {
    return "Cannot reach the backend server.\n\nStart it with:\n```bash\npython -m uvicorn main:app --reload --host 0.0.0.0 --port 8000\n```";
  }
  if (msg.includes("413")) return "Audio recording too large for processing.";
  if (msg.includes("400")) return "Bad request — please enter a valid query.";
  return `Service error: ${msg}`;
}

/* ─── Singleton Audio Engine ─────────────────────────────────────────────── */

type AudioListener = (playing: boolean, progress: number) => void;

const audioEngine = (() => {
  let audio: HTMLAudioElement | null = null;
  let animId: number | null = null;
  let activeId: number | null = null;
  const listeners = new Set<AudioListener>();

  const notify = (playing: boolean, prog: number = 0) => {
    listeners.forEach((fn) => fn(playing, prog));
  };

  const trackProgress = () => {
    if (audio && !audio.paused && audio.duration) {
      notify(true, (audio.currentTime / audio.duration) * 100);
      animId = requestAnimationFrame(trackProgress);
    }
  };

  return {
    play(b64: string, msgId: number) {
      this.stop();
      activeId = msgId;
      audio = new Audio(`data:audio/wav;base64,${b64}`);
      audio.onplay = () => {
        notify(true, 0);
        trackProgress();
      };
      audio.onended = audio.onerror = () => {
        this.stop();
      };
      audio.play().catch(() => {
        this.stop();
      });
    },
    stop() {
      if (animId) cancelAnimationFrame(animId);
      if (audio) {
        audio.pause();
        audio = null;
      }
      activeId = null;
      notify(false, 0);
    },
    getActiveId() {
      return activeId;
    },
    subscribe(fn: AudioListener) {
      listeners.add(fn);
      return () => {
        listeners.delete(fn);
      };
    },
  };
})();

/* ─── Technical Reference Snippets (Resend Code-Window) ──────────────────── */

const SPEC_FILES = {
  "architecture.json": `{
  "system": "Hacker House Goa 2026 Intelligence Engine",
  "pipeline": {
    "speech_to_text": {
      "model": "sarvam-ai/saaras:v3",
      "languages": ["hi-IN", "en-IN", "bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN", "od-IN"]
    },
    "retrieval_agent": {
      "framework": "LangChain 1.x CompiledStateGraph",
      "llm": "Mistral-Small-24B-Instruct",
      "vector_store": "ChromaDB (cosine similarity, threshold: 0.75)",
      "guardrails": ["prompt_injection", "out_of_domain", "grounding_check"]
    },
    "text_to_speech": {
      "model": "sarvam-ai/bulbul:v3",
      "codec": "wav/pcm",
      "latency_target": "<350ms"
    }
  }
}`,
  "tracks.yaml": `# Hacker House Goa 2026 — Track Allocations
tracks:
  - id: "agentic-ai"
    title: "Autonomous Multi-Agent Systems"
    bounty: "₹5,00,000"
    criteria: "Grounding, tool-use autonomy, latency under pressure"
  - id: "indic-voice"
    title: "Indic Voice & Multilingual Applications"
    bounty: "₹3,50,000"
    criteria: "Low-latency regional ASR/TTS, edge-ready pipelines"
  - id: "infra-compute"
    title: "Decentralized Compute & Edge Systems"
    bounty: "₹2,50,000"
    criteria: "Resilience, p2p consensus, resource efficiency"
`,
  "schedule.md": `### Hacker House Goa 2026 — Key Schedule

- **Day 1 (10:00 AM)**: Keynote & Problem Statements Release
- **Day 1 (02:00 PM)**: Mentorship Checkpoint #1 — Architecture Review
- **Day 2 (11:00 AM)**: Mid-Hack Prototype Demos & Guardrail Testing
- **Day 2 (08:00 PM)**: Code Freeze & Final PR Submissions
- **Day 3 (10:00 AM)**: Live Jury Pitches & Winners Ceremony
`,
};

/* ─── Topic Chips ────────────────────────────────────────────────────────── */

const TOPIC_CHIPS = [
  { label: "Prizes & Bounty Pools", query: "What are the prizes and bounty allocations for HHGoa 2026?" },
  { label: "Hackathon Schedule", query: "Can you give me the full timeline and milestone schedule?" },
  { label: "Submission Guidelines", query: "What are the rules, team limits, and submission criteria?" },
  { label: "Autonomous Agent Track", query: "What is expected in the Autonomous Multi-Agent Track?" },
  { label: "Venue & Mentors", query: "Tell me about the Goa venue, mentor checkpoints, and Wi-Fi access." },
];

/* ─── Micro Icons ────────────────────────────────────────────────────────── */

function IconMic({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" x2="12" y1="19" y2="22" />
    </svg>
  );
}

function IconSend({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function IconStop({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function IconPlay({ className = "w-3.5 h-3.5" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <polygon points="6 3 20 12 6 21 6 3" />
    </svg>
  );
}

function IconVolume({ active }: { active: boolean }) {
  return active ? (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    </svg>
  ) : (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <line x1="23" y1="9" x2="17" y2="15" />
      <line x1="17" y1="9" x2="23" y2="15" />
    </svg>
  );
}

function IconCopy({ className = "w-3.5 h-3.5" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
    </svg>
  );
}

function IconCheck({ className = "w-3.5 h-3.5" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconArrowLeft({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
    </svg>
  );
}

/* ─── Audio Bubble Player ────────────────────────────────────────────────── */

function AudioPlayerInline({ b64, msgId }: { b64: string; msgId: number }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    return audioEngine.subscribe((playing, prog) => {
      if (audioEngine.getActiveId() === msgId) {
        setIsPlaying(playing);
        setProgress(prog);
      } else {
        setIsPlaying(false);
        setProgress(0);
      }
    });
  }, [msgId]);

  const toggle = () => {
    if (isPlaying) {
      audioEngine.stop();
    } else {
      audioEngine.play(b64, msgId);
    }
  };

  return (
    <div className="mt-3.5 pt-3 border-t border-[rgba(255,255,255,0.06)] flex items-center gap-3">
      <button
        onClick={toggle}
        className={`h-7 px-2.5 rounded-md text-xs font-medium inline-flex items-center gap-1.5 transition-all cursor-pointer ${
          isPlaying
            ? "bg-[#ff2047]/15 text-[#ff2047] border border-[#ff2047]/30"
            : "bg-[#101012] hover:bg-[#161618] text-[#fcfdff] border border-[rgba(255,255,255,0.12)]"
        }`}
      >
        {isPlaying ? <IconStop className="w-3 h-3" /> : <IconPlay className="w-3 h-3" />}
        <span>{isPlaying ? "Pause audio" : "Listen voice"}</span>
      </button>

      {/* Progress track */}
      <div className="flex-1 h-1.5 bg-[#161618] rounded-full overflow-hidden border border-[rgba(255,255,255,0.06)]">
        <div
          className="h-full bg-[#fcfdff] transition-all duration-100"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Equalizer animation when playing */}
      {isPlaying && (
        <div className="flex items-center gap-0.5 h-3">
          <span className="w-0.5 bg-[#11ff99] animate-wave-1 rounded-full" />
          <span className="w-0.5 bg-[#11ff99] animate-wave-2 rounded-full" />
          <span className="w-0.5 bg-[#11ff99] animate-wave-3 rounded-full" />
          <span className="w-0.5 bg-[#11ff99] animate-wave-4 rounded-full" />
        </div>
      )}
    </div>
  );
}

/* ─── Message Bubble ─────────────────────────────────────────────────────── */

function ChatBubble({
  msg,
  onCopy,
  copiedId,
}: {
  msg: Message;
  onCopy: (text: string, id: number) => void;
  copiedId: number | null;
}) {
  const isUser = msg.role === "user";
  const isCopied = copiedId === msg.id;

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-up">
        <div className="max-w-[80%] sm:max-w-[70%] bg-[#101012] border border-[rgba(255,255,255,0.14)] rounded-[12px] p-4 text-[#fcfdff]">
          <div className="flex items-center justify-between gap-4 mb-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#888e90]">
              {msg.isVoice ? "🎤 Spoken Query" : "Prompt"}
            </span>
            <span className="text-[11px] font-mono text-[#888e90]">{msg.timestamp}</span>
          </div>
          <p className="text-[14px] leading-relaxed text-[#fcfdff] whitespace-pre-wrap">
            {msg.text}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start animate-fade-up">
      <div
        className={`max-w-[90%] sm:max-w-[82%] bg-[#0a0a0c] border rounded-[12px] p-5 ${
          msg.is_refusal
            ? "border-[#ff2047]/40 bg-[#0e0708]"
            : "border-[rgba(255,255,255,0.08)]"
        }`}
      >
        {/* Header meta */}
        <div className="flex items-center justify-between gap-3 mb-2.5 pb-2 border-b border-[rgba(255,255,255,0.04)]">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-[4px] bg-[#101012] border border-[rgba(255,255,255,0.12)] flex items-center justify-center text-[10px] font-semibold text-[#fcfdff]">
              HH
            </div>
            <span className="text-[12px] font-medium text-[#fcfdff]">HHGoa Intelligence</span>
            {msg.source_docs && msg.source_docs.length > 0 && (
              <span className="text-[10px] font-mono bg-[#101012] text-[#a1a4a5] px-1.5 py-0.5 rounded border border-[rgba(255,255,255,0.06)]">
                RAG Verified
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => onCopy(msg.text, msg.id)}
              className="text-[#888e90] hover:text-[#fcfdff] text-xs p-1 rounded transition-colors cursor-pointer"
              title="Copy message"
              aria-label="Copy text"
            >
              {isCopied ? <IconCheck className="text-[#11ff99]" /> : <IconCopy />}
            </button>
            <span className="text-[11px] font-mono text-[#888e90]">{msg.timestamp}</span>
          </div>
        </div>

        {/* Markdown content */}
        <div className="text-[14px] leading-relaxed text-[rgba(252,253,255,0.88)] space-y-3 font-normal prose-invert">
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
              ul: ({ children }) => <ul className="list-disc pl-5 space-y-1 mb-2.5">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1 mb-2.5">{children}</ol>,
              li: ({ children }) => <li className="leading-relaxed">{children}</li>,
              strong: ({ children }) => <strong className="font-semibold text-[#fcfdff]">{children}</strong>,
              code: ({ children }) => (
                <code className="px-1.5 py-0.5 rounded bg-[#06060a] border border-[rgba(255,255,255,0.08)] font-mono text-[12px] text-[#fcfdff]">
                  {children}
                </code>
              ),
              pre: ({ children }) => (
                <pre className="p-3 my-2 rounded-[8px] bg-[#06060a] border border-[rgba(255,255,255,0.1)] overflow-x-auto font-mono text-[12px] text-[#fcfdff]">
                  {children}
                </pre>
              ),
              h1: ({ children }) => <h1 className="text-[16px] font-semibold text-[#fcfdff] mt-3 mb-1.5">{children}</h1>,
              h2: ({ children }) => <h2 className="text-[15px] font-semibold text-[#fcfdff] mt-2.5 mb-1">{children}</h2>,
              h3: ({ children }) => <h3 className="text-[14px] font-semibold text-[#fcfdff] mt-2 mb-1">{children}</h3>,
            }}
          >
            {msg.text}
          </ReactMarkdown>
        </div>

        {/* References / Grounded Knowledge Files */}
        {msg.source_docs && msg.source_docs.length > 0 && (
          <div className="mt-3.5 pt-3 border-t border-[rgba(255,255,255,0.06)] flex flex-col gap-2">
            <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-wider text-[#888e90]">
              <svg className="w-3.5 h-3.5 text-[#3b9eff]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              <span>Referenced Source Files:</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {msg.source_docs.map((doc) => (
                <span
                  key={doc}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[6px] bg-[#101012] border border-[rgba(255,255,255,0.12)] text-[12px] font-mono text-[#fcfdff] hover:border-[rgba(255,255,255,0.25)] transition-colors"
                >
                  <span className="text-[#3b9eff]">📄</span>
                  <span>{doc}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Guardrail alert if triggered */}
        {msg.guardrail_triggered && msg.guardrail_reason && (
          <div className="mt-3 px-3 py-2 rounded-md bg-[#ffc53d]/10 border border-[#ffc53d]/30 text-[#ffc53d] text-xs flex items-center gap-2">
            <span>⚠</span>
            <span>Policy Guardrail: {msg.guardrail_reason}</span>
          </div>
        )}

        {/* Audio Player if TTS provided */}
        {msg.audio_b64 && <AudioPlayerInline b64={msg.audio_b64} msgId={msg.id} />}
      </div>
    </div>
  );
}

/* ─── Typing Indicator ───────────────────────────────────────────────────── */

function TypingIndicator() {
  return (
    <div className="flex justify-start animate-fade-up">
      <div className="bg-[#0a0a0c] border border-[rgba(255,255,255,0.08)] rounded-[12px] px-4 py-3 flex items-center gap-2">
        <span className="text-xs font-mono text-[#888e90] mr-1">Generating response</span>
        <span className="w-1.5 h-1.5 rounded-full bg-[#fcfdff] animate-dot-1" />
        <span className="w-1.5 h-1.5 rounded-full bg-[#fcfdff] animate-dot-2" />
        <span className="w-1.5 h-1.5 rounded-full bg-[#fcfdff] animate-dot-3" />
      </div>
    </div>
  );
}

/* ─── Main Page ──────────────────────────────────────────────────────────── */

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [recording, setRecording] = useState<RecordingState>("idle");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [loading, setLoading] = useState(false);
  const [autoplay, setAutoplay] = useState(true);
  const [backend, setBackend] = useState<BackendStatus>("checking");
  const [viewTab, setViewTab] = useState<TabView>("landing");
  const [activeSpec, setActiveSpec] = useState<keyof typeof SPEC_FILES>("architecture.json");
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<NodeJS.Timeout | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /* Health Ping — proxied through Next.js API route to avoid ad-blocker blocks */
  useEffect(() => {
    let unmounted = false;
    const checkStatus = async () => {
      try {
        // /api/health is same-origin — never blocked by Brave or uBlock
        const res = await fetch("/api/health", {
          cache: "no-store",
          signal: AbortSignal.timeout(5000),
        });
        if (!unmounted) setBackend(res.ok ? "up" : "down");
      } catch (err) {
        if (unmounted) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (
          msg.toLowerCase().includes("blocked") ||
          msg.toLowerCase().includes("err_blocked")
        ) {
          return; // ad blocker — don't flip to offline
        }
        setBackend("down");
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 12000);
    return () => {
      unmounted = true;
      clearInterval(interval);
    };
  }, []);

  /* Recording Timer */
  useEffect(() => {
    if (recording === "recording") {
      setRecordingSeconds(0);
      recordTimerRef.current = setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } else {
      if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    }
    return () => {
      if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    };
  }, [recording]);

  /* Scroll Helper */
  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 40);
  }, []);

  /* Push Message Helper */
  const pushMessage = useCallback(
    (msg: Omit<Message, "id" | "timestamp">) => {
      const newMsg: Message = {
        ...msg,
        id: nextMsgId(),
        timestamp: getTimestamp(),
      };
      setMessages((prev) => [...prev, newMsg]);
      scrollToBottom();
      return newMsg.id;
    },
    [scrollToBottom]
  );

  /* Response Handler */
  const handleAgentResponse = useCallback(
    (data: AgentResponse) => {
      const id = pushMessage({
        role: "assistant",
        text: data.answer,
        source_docs: data.source_docs,
        is_refusal: data.is_refusal,
        audio_b64: data.audio_b64,
        guardrail_triggered: data.guardrail_triggered,
        guardrail_reason: data.guardrail_reason,
      });

      if (autoplay && data.audio_b64) {
        audioEngine.play(data.audio_b64, id);
      }
    },
    [pushMessage, autoplay]
  );

  /* Send Text Query */
  const submitText = useCallback(
    async (overrideText?: string) => {
      const query = (overrideText ?? input).trim();
      if (!query || loading) return;

      setViewTab("chat");
      setInput("");
      pushMessage({ role: "user", text: query, isVoice: false });
      setLoading(true);

      try {
        const res = await fetch(`/api/ask-text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: query, tts: true }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: AgentResponse = await res.json();
        setBackend("up");
        handleAgentResponse(data);
      } catch (err) {
        if (err instanceof Error && err.message.toLowerCase().includes("failed to fetch")) {
          // Only mark offline if it's not a Brave/ad-blocker block
          setBackend("down");
        }
        pushMessage({
          role: "assistant",
          text: parseFriendlyError(err),
          is_refusal: true,
        });
      } finally {
        setLoading(false);
      }
    },
    [input, loading, pushMessage, handleAgentResponse]
  );

  /* Voice Recording Controls */
  const startRecording = useCallback(async () => {
    if (recording !== "idle") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const mr = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording("processing");
        setLoading(true);
        setViewTab("chat");

        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        try {
          const res = await fetch(`/api/ask?tts=true`, {
            method: "POST",
            body: formData,
          });

          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data: AgentResponse = await res.json();
          setBackend("up");
          pushMessage({
            role: "user",
            text: data.transcript || "Voice input",
            isVoice: true,
          });
          handleAgentResponse(data);
        } catch (err) {
          if (err instanceof Error && err.message.toLowerCase().includes("failed to fetch")) {
            setBackend("down");
          }
          pushMessage({
            role: "assistant",
            text: parseFriendlyError(err),
            is_refusal: true,
          });
        } finally {
          setLoading(false);
          setRecording("idle");
        }
      };

      mediaRecorderRef.current = mr;
      mr.start();
      setRecording("recording");
    } catch {
      alert("Microphone permission was denied or not available in your browser.");
    }
  }, [recording, pushMessage, handleAgentResponse]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && recording === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, [recording]);

  /* Keyboard shortcut Enter to submit */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitText();
    }
  };

  /* Copy message text */
  const handleCopy = (text: string, id: number) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const formatTimer = (sec: number) => {
    const m = Math.floor(sec / 60)
      .toString()
      .padStart(2, "0");
    const s = (sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <div className="flex flex-col h-[100svh] bg-black text-[#fcfdff] overflow-hidden selection:bg-white/20 selection:text-white">
      {/* ── Top Atmospheric Section Glow ──────────────────────────────────── */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-[1200px] h-[340px] glow-hero pointer-events-none z-0" />

      {/* ── Top Navigation Bar (Resend nav-bar) ────────────────────────────── */}
      <header className="relative z-10 shrink-0 h-16 border-b border-[rgba(255,255,255,0.06)] px-4 sm:px-8 flex items-center justify-between bg-black/80 backdrop-blur-md">
        {/* Left: Brand Identity */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setViewTab("landing")}
            className="flex items-center gap-2.5 group cursor-pointer text-left bg-transparent border-0 p-0"
          >
            <div className="w-7 h-7 rounded-[6px] bg-[#101012] border border-[rgba(255,255,255,0.14)] flex items-center justify-center text-xs font-bold text-[#fcfdff] transition group-hover:border-[rgba(255,255,255,0.3)]">
              HH
            </div>
            <div>
              <span className="text-[14px] font-medium tracking-tight text-[#fcfdff]">
                HHGoa <span className="text-[#888e90]">2026</span>
              </span>
            </div>
          </button>
        </div>

        {/* Center: Nav Switchers */}
        <nav className="hidden md:flex items-center gap-1 bg-[#0a0a0c] border border-[rgba(255,255,255,0.08)] rounded-full p-1">
          <button
            onClick={() => setViewTab("landing")}
            className={`px-3.5 py-1 text-xs rounded-full font-medium transition cursor-pointer ${
              viewTab === "landing"
                ? "bg-[#101012] text-[#fcfdff] border border-[rgba(255,255,255,0.12)]"
                : "text-[#888e90] hover:text-[#fcfdff]"
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setViewTab("chat")}
            className={`px-3.5 py-1 text-xs rounded-full font-medium transition cursor-pointer flex items-center gap-1.5 ${
              viewTab === "chat"
                ? "bg-[#101012] text-[#fcfdff] border border-[rgba(255,255,255,0.12)]"
                : "text-[#888e90] hover:text-[#fcfdff]"
            }`}
          >
            Assistant
            {messages.length > 0 && (
              <span className="w-4 h-4 rounded-full bg-[rgba(255,255,255,0.1)] text-[10px] flex items-center justify-center">
                {messages.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setViewTab("specs")}
            className={`px-3.5 py-1 text-xs rounded-full font-medium transition cursor-pointer ${
              viewTab === "specs"
                ? "bg-[#101012] text-[#fcfdff] border border-[rgba(255,255,255,0.12)]"
                : "text-[#888e90] hover:text-[#fcfdff]"
            }`}
          >
            Specs & Architecture
          </button>
        </nav>

        {/* Right: Status & Sound Toggle */}
        <div className="flex items-center gap-3">
          {/* Real-time backend status dot */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#0a0a0c] border border-[rgba(255,255,255,0.06)] text-[11px] font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                backend === "up"
                  ? "bg-[#11ff99] shadow-[0_0_8px_rgba(17,255,153,0.4)]"
                  : backend === "checking"
                  ? "bg-[#ffc53d] animate-pulse"
                  : "bg-[#ff2047]"
              }`}
            />
            <span
              className={
                backend === "up"
                  ? "text-[#11ff99]"
                  : backend === "checking"
                  ? "text-[#ffc53d]"
                  : "text-[#ff2047]"
              }
            >
              {backend === "up" ? "System Live" : backend === "checking" ? "Checking" : "Offline"}
            </span>
          </div>

          {/* Autoplay Audio toggle */}
          <button
            onClick={() => {
              setAutoplay((v) => !v);
              if (autoplay) audioEngine.stop();
            }}
            title={autoplay ? "Voice responses active (click to mute)" : "Voice muted (click to unmute)"}
            aria-label="Toggle voice autoplay"
            className={`w-8 h-8 rounded-md flex items-center justify-center border transition cursor-pointer ${
              autoplay
                ? "bg-[#101012] border-[rgba(255,255,255,0.14)] text-[#fcfdff]"
                : "bg-transparent border-[rgba(255,255,255,0.06)] text-[#888e90] hover:text-[#fcfdff]"
            }`}
          >
            <IconVolume active={autoplay} />
          </button>

          {/* New query CTA */}
          <button
            onClick={() => {
              setViewTab("chat");
              setTimeout(() => textareaRef.current?.focus(), 50);
            }}
            className="hidden sm:inline-flex items-center justify-center h-8 px-3 rounded-md bg-[#fcfdff] text-black text-xs font-semibold hover:bg-[#f1f7fe] transition active:scale-98 cursor-pointer"
          >
            Ask Assistant
          </button>
        </div>
      </header>

      {/* ── Main Workspace Content ────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 flex flex-col overflow-hidden">
        {/* ── VIEW: LANDING OVERVIEW ───────────────────────────────────────── */}
        {viewTab === "landing" && (
          <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-10 sm:py-16 flex flex-col items-center">
            <div className="w-full max-w-[820px] flex flex-col items-center text-center animate-fade-up">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#101012] border border-[rgba(255,255,255,0.12)] text-xs text-[#a1a4a5] mb-6">
                <span className="w-1.5 h-1.5 rounded-full bg-[#11ff99]" />
                <span>Hacker House Goa · Voice & Knowledge Retrieval</span>
              </div>

              {/* Domaine / Editorial Headline */}
              <h1 className="font-editorial text-[42px] sm:text-[68px] lg:text-[76px] font-normal tracking-tight text-[#fcfdff] max-w-[760px] leading-[1.04] mb-5">
                The intelligence layer for Hacker House Goa.
              </h1>

              {/* Subhead */}
              <p className="text-[15px] sm:text-[17px] text-[#a1a4a5] max-w-[560px] leading-relaxed mb-10 font-normal">
                Ask in any Indian voice or text. Get real-time verified answers on schedule, bounties, rules, and multi-agent challenges.
              </p>

              {/* Primary Input Dock (Resend text-input container) */}
              <div className="w-full max-w-[680px] bg-[#0a0a0c] border border-[rgba(255,255,255,0.14)] focus-within:border-[rgba(255,255,255,0.35)] rounded-[12px] p-2.5 flex flex-col gap-2 transition-all mb-8 shadow-2xl">
                <div className="flex items-center gap-3 px-2">
                  <textarea
                    ref={textareaRef}
                    rows={1}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about dates, prize pool, team rules, submission criteria…"
                    disabled={loading || recording !== "idle"}
                    className="flex-1 bg-transparent border-0 outline-none text-[14px] text-[#fcfdff] placeholder-[#888e90] resize-none py-1.5 leading-relaxed"
                  />

                  {/* Mic Button */}
                  <button
                    onClick={recording === "recording" ? stopRecording : startRecording}
                    disabled={loading && recording === "idle"}
                    title={recording === "recording" ? "Stop recording" : "Speak your query"}
                    aria-label="Voice input"
                    className={`h-9 px-3 rounded-[8px] inline-flex items-center gap-2 text-xs font-medium transition cursor-pointer ${
                      recording === "recording"
                        ? "bg-[#ff2047] text-white animate-pulse"
                        : recording === "processing"
                        ? "bg-[#ffc53d] text-black"
                        : "bg-[#101012] text-[#fcfdff] hover:bg-[#18181c] border border-[rgba(255,255,255,0.12)]"
                    }`}
                  >
                    {recording === "recording" ? (
                      <>
                        <IconStop className="w-3.5 h-3.5" />
                        <span>{formatTimer(recordingSeconds)}</span>
                      </>
                    ) : recording === "processing" ? (
                      <span>Transcribing…</span>
                    ) : (
                      <>
                        <IconMic className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Voice</span>
                      </>
                    )}
                  </button>

                  {/* Send Button */}
                  <button
                    onClick={() => submitText()}
                    disabled={!input.trim() || loading || recording !== "idle"}
                    aria-label="Submit query"
                    className={`w-9 h-9 rounded-[8px] flex items-center justify-center transition cursor-pointer ${
                      input.trim() && !loading && recording === "idle"
                        ? "bg-[#fcfdff] text-black hover:bg-[#f1f7fe]"
                        : "bg-[#101012] text-[#464a4d] border border-[rgba(255,255,255,0.06)] cursor-not-allowed"
                    }`}
                  >
                    <IconSend className="w-3.5 h-3.5" />
                  </button>
                </div>

                {recording === "recording" && (
                  <div className="flex items-center justify-between px-3 py-1.5 bg-[#ff2047]/10 border border-[#ff2047]/20 rounded-md text-xs text-[#ff2047]">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-[#ff2047] animate-ping" />
                      <span>Recording speech in progress — click to send</span>
                    </div>
                    <span className="font-mono">{formatTimer(recordingSeconds)}</span>
                  </div>
                )}
              </div>

              {/* Curated Question Chips */}
              <div className="w-full max-w-[720px] flex flex-wrap items-center justify-center gap-2 mb-14">
                {TOPIC_CHIPS.map((chip) => (
                  <button
                    key={chip.label}
                    onClick={() => submitText(chip.query)}
                    className="px-3 py-1.5 rounded-full bg-[#0a0a0c] hover:bg-[#101012] border border-[rgba(255,255,255,0.08)] hover:border-[rgba(255,255,255,0.2)] text-xs text-[#a1a4a5] hover:text-[#fcfdff] transition cursor-pointer flex items-center gap-1.5"
                  >
                    <span>{chip.label}</span>
                    <span className="text-[10px] text-[#888e90]">→</span>
                  </button>
                ))}
              </div>

              {/* Secondary Technical Spec Preview (Resend Code-Window) */}
              <div className="w-full max-w-[760px] text-left">
                <div className="flex items-center justify-between mb-3 px-1">
                  <span className="text-xs font-mono uppercase tracking-wider text-[#888e90]">
                    Live Architecture & Ground Truth
                  </span>
                  <button
                    onClick={() => setViewTab("specs")}
                    className="text-xs text-[#3b9eff] hover:underline"
                  >
                    View full specs →
                  </button>
                </div>

                <div className="surface-deep rounded-[12px] border border-[rgba(255,255,255,0.12)] p-4 sm:p-5 bg-[#06060a]">
                  {/* Traffic lights */}
                  <div className="flex items-center justify-between pb-3 mb-3 border-b border-[rgba(255,255,255,0.06)]">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#ff2047]" />
                      <span className="w-2.5 h-2.5 rounded-full bg-[#ffc53d]" />
                      <span className="w-2.5 h-2.5 rounded-full bg-[#11ff99]" />
                    </div>
                    <span className="text-[11px] font-mono text-[#888e90]">hhgoa_pipeline.json</span>
                  </div>

                  <pre className="font-mono text-[12px] text-[#a1a4a5] leading-relaxed overflow-x-auto">
                    {SPEC_FILES["architecture.json"]}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW: LIVE CHAT ─────────────────────────────────────────────── */}
        {viewTab === "chat" && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Messages Scroll Area */}
            <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6">
              <div className="max-w-[760px] mx-auto flex flex-col gap-4">
                {messages.length === 0 && (
                  <div className="text-center py-16 text-[#888e90] text-sm animate-fade-up">
                    <p className="font-editorial text-2xl text-[#fcfdff] mb-2">No active conversation</p>
                    <p className="max-w-[400px] mx-auto text-xs text-[#888e90]">
                      Type your question below or click on one of the quick suggestions to start.
                    </p>
                    <div className="flex flex-wrap justify-center gap-2 mt-6">
                      {TOPIC_CHIPS.slice(0, 3).map((chip) => (
                        <button
                          key={chip.label}
                          onClick={() => submitText(chip.query)}
                          className="px-3 py-1.5 rounded-full bg-[#0a0a0c] border border-[rgba(255,255,255,0.08)] text-xs text-[#a1a4a5] hover:text-[#fcfdff] cursor-pointer"
                        >
                          {chip.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((m) => (
                  <ChatBubble
                    key={m.id}
                    msg={m}
                    onCopy={handleCopy}
                    copiedId={copiedId}
                  />
                ))}

                {loading && <TypingIndicator />}
                <div ref={chatBottomRef} />
              </div>
            </div>

            {/* Bottom Dock Input Bar */}
            <div className="shrink-0 border-t border-[rgba(255,255,255,0.06)] bg-black/90 backdrop-blur-md px-4 sm:px-8 py-3.5">
              <div className="max-w-[760px] mx-auto flex flex-col gap-2">
                <div className="flex items-center gap-2 bg-[#0a0a0c] border border-[rgba(255,255,255,0.14)] focus-within:border-[rgba(255,255,255,0.35)] rounded-[12px] p-2">
                  <button
                    onClick={() => setViewTab("landing")}
                    title="Return to overview"
                    aria-label="Back to overview"
                    className="w-8 h-8 rounded-[8px] flex items-center justify-center text-[#888e90] hover:text-[#fcfdff] hover:bg-[#101012] transition cursor-pointer"
                  >
                    <IconArrowLeft className="w-4 h-4" />
                  </button>

                  <textarea
                    ref={textareaRef}
                    rows={1}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about HHGoa 2026… (Enter to send, Shift+Enter for newline)"
                    disabled={loading || recording !== "idle"}
                    className="flex-1 bg-transparent border-0 outline-none text-[14px] text-[#fcfdff] placeholder-[#888e90] resize-none py-1 max-h-24 overflow-y-auto"
                  />

                  {/* Voice recording button */}
                  <button
                    onClick={recording === "recording" ? stopRecording : startRecording}
                    disabled={loading && recording === "idle"}
                    title={recording === "recording" ? "Stop recording" : "Voice input"}
                    aria-label="Voice input"
                    className={`h-8 px-2.5 rounded-[8px] inline-flex items-center gap-1.5 text-xs font-medium transition cursor-pointer ${
                      recording === "recording"
                        ? "bg-[#ff2047] text-white animate-pulse"
                        : recording === "processing"
                        ? "bg-[#ffc53d] text-black"
                        : "bg-[#101012] text-[#fcfdff] hover:bg-[#161618] border border-[rgba(255,255,255,0.12)]"
                    }`}
                  >
                    {recording === "recording" ? (
                      <>
                        <IconStop className="w-3 h-3" />
                        <span>{formatTimer(recordingSeconds)}</span>
                      </>
                    ) : (
                      <>
                        <IconMic className="w-3 h-3" />
                        <span className="hidden sm:inline">Voice</span>
                      </>
                    )}
                  </button>

                  {/* Send */}
                  <button
                    onClick={() => submitText()}
                    disabled={!input.trim() || loading || recording !== "idle"}
                    aria-label="Send message"
                    className={`w-8 h-8 rounded-[8px] flex items-center justify-center transition cursor-pointer ${
                      input.trim() && !loading && recording === "idle"
                        ? "bg-[#fcfdff] text-black hover:bg-[#f1f7fe]"
                        : "bg-[#101012] text-[#464a4d] border border-[rgba(255,255,255,0.06)] cursor-not-allowed"
                    }`}
                  >
                    <IconSend className="w-3.5 h-3.5" />
                  </button>
                </div>

                {recording === "recording" && (
                  <div className="text-center text-xs text-[#ff2047] font-mono animate-pulse">
                    🔴 Recording live voice audio… Click voice button again or stop to transcribe
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW: TECHNICAL SPECS & GROUND TRUTH ─────────────────────────── */}
        {viewTab === "specs" && (
          <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-8 sm:py-12 flex justify-center">
            <div className="w-full max-w-[800px] animate-fade-up">
              <div className="mb-8">
                <span className="text-xs font-mono text-[#888e90] uppercase tracking-wider">
                  Reference Documentation
                </span>
                <h2 className="font-editorial text-3xl sm:text-4xl text-[#fcfdff] mt-1 mb-2">
                  System Architecture & Hackathon Ground Truth
                </h2>
                <p className="text-sm text-[#a1a4a5]">
                  The voice assistant utilizes Sarvam AI for multilingual ASR/TTS and a LangChain 1.x CompiledStateGraph over ChromaDB vector embeddings.
                </p>
              </div>

              {/* Code window with tabs */}
              <div className="surface-deep rounded-[12px] border border-[rgba(255,255,255,0.14)] overflow-hidden bg-[#06060a]">
                {/* Header with traffic lights & tab bar */}
                <div className="flex items-center justify-between px-4 py-3 bg-[#0a0a0c] border-b border-[rgba(255,255,255,0.06)]">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#ff2047]" />
                    <span className="w-2.5 h-2.5 rounded-full bg-[#ffc53d]" />
                    <span className="w-2.5 h-2.5 rounded-full bg-[#11ff99]" />
                  </div>

                  {/* Tabs */}
                  <div className="flex items-center gap-1 bg-[#101012] p-0.5 rounded-md border border-[rgba(255,255,255,0.08)]">
                    {(Object.keys(SPEC_FILES) as Array<keyof typeof SPEC_FILES>).map((file) => (
                      <button
                        key={file}
                        onClick={() => setActiveSpec(file)}
                        className={`px-2.5 py-1 rounded text-[11px] font-mono transition cursor-pointer ${
                          activeSpec === file
                            ? "bg-[#06060a] text-[#fcfdff] border border-[rgba(255,255,255,0.12)]"
                            : "text-[#888e90] hover:text-[#fcfdff]"
                        }`}
                      >
                        {file}
                      </button>
                    ))}
                  </div>
                </div>

                {/* File Contents */}
                <div className="p-5 font-mono text-[12.5px] leading-relaxed text-[#fcfdff] overflow-x-auto">
                  <pre className="text-[rgba(252,253,255,0.85)]">
                    {SPEC_FILES[activeSpec]}
                  </pre>
                </div>
              </div>

              {/* Quick actions */}
              <div className="mt-6 flex items-center justify-between pt-4 border-t border-[rgba(255,255,255,0.06)] text-xs">
                <span className="text-[#888e90]">Have questions about these specs?</span>
                <button
                  onClick={() => {
                    setViewTab("chat");
                    submitText(`Explain the details in ${activeSpec}`);
                  }}
                  className="px-3 py-1.5 rounded-md bg-[#101012] border border-[rgba(255,255,255,0.12)] text-[#fcfdff] hover:bg-[#161618] transition cursor-pointer"
                >
                  Ask assistant about {activeSpec} →
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
