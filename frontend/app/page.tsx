"use client";

import { useRef, useState, useCallback, useEffect } from "react";

/* ─── Types ──────────────────────────────────────────────────────────────── */

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
  is_refusal?: boolean;
  audio_b64?: string | null;
  guardrail_triggered?: boolean;
  guardrail_reason?: string;
}

type RecordingState = "idle" | "recording" | "processing";
type BackendStatus  = "checking" | "up" | "down";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
let msgId = 0;
const nextId = () => ++msgId;

function friendlyError(err: unknown): string {
  const m = err instanceof Error ? err.message : String(err);
  if (m.toLowerCase().includes("failed to fetch") || m.toLowerCase().includes("networkerror") || m.toLowerCase().includes("load failed"))
    return "Cannot reach the backend.\n\nStart it with:\npython -m uvicorn main:app --reload --host 0.0.0.0 --port 8000";
  if (m.includes("413")) return "Audio file too large.";
  if (m.includes("400")) return "Bad request — check your input.";
  return `Something went wrong: ${m}`;
}

/* ─── Audio player ───────────────────────────────────────────────────────── */

type AL = (v: boolean) => void;
const player = (() => {
  let cur: HTMLAudioElement | null = null;
  const ls = new Set<AL>();
  const emit = (v: boolean) => ls.forEach(f => f(v));
  return {
    play(b64: string) {
      cur?.pause();
      cur = new Audio(`data:audio/wav;base64,${b64}`);
      emit(true);
      cur.onended = cur.onerror = () => { cur = null; emit(false); };
      cur.play().catch(() => { cur = null; emit(false); });
    },
    stop() { cur?.pause(); cur = null; emit(false); },
    subscribe(fn: AL) { ls.add(fn); return () => { ls.delete(fn); }; },
  };
})();

/* ─── Data ───────────────────────────────────────────────────────────────── */

const CHIPS = [
  { label: "Dates & Schedule",   icon: "📅" },
  { label: "Rules & Guidelines", icon: "📋" },
  { label: "Prizes & Rewards",   icon: "🏆" },
  { label: "Tasks & Challenges", icon: "⚡" },
  { label: "Just say hi",        icon: "👋" },
];

/* ─── Design tokens ──────────────────────────────────────────────────────── */
/* Resend-inspired dark material palette */
const R = {
  bg:          "#000000",   /* true black page base              */
  surf0:       "#0A0A0A",   /* faintest surface lift             */
  surf1:       "#111111",   /* card / control surface            */
  surf2:       "#161616",   /* elevated / hover                  */
  surf3:       "#1E1E1E",   /* highest elevation                 */
  border0:     "#1A1A1A",   /* subtle divider                    */
  border1:     "#262626",   /* standard border                   */
  border2:     "#333333",   /* prominent border                  */
  text0:       "#FFFFFF",   /* primary                           */
  text1:       "#A1A1A1",   /* secondary                         */
  text2:       "#737373",   /* muted                             */
  text3:       "#525252",   /* faint                             */
  green:       "#22C55E",   /* brand accent — solid, no glow     */
  greenDim:    "#16A34A",   /* darker green for larger text      */
  red:         "#EF4444",
  redSurf:     "#1A0A0A",
  amber:       "#F59E0B",
} as const;

/* ─── Icons ──────────────────────────────────────────────────────────────── */

const IcoMic = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/>
    <path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8"/>
  </svg>
);
const IcoSend = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);
const IcoBack = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 18l-6-6 6-6"/>
  </svg>
);
const IcoVol = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
    <path d="M15.54 8.46a5 5 0 010 7.07M19.07 4.93a10 10 0 010 14.14"/>
  </svg>
);
const IcoMute = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
    <line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>
  </svg>
);
const IcoPlay = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
);
const IcoStop = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
);

/* ─── Sub-components ─────────────────────────────────────────────────────── */

function PlayButton({ b64 }: { b64: string }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    return player.subscribe(v => { if (!v) setOn(false); });
  }, []);
  return (
    <button
      onClick={() => on ? (player.stop(), setOn(false)) : (player.play(b64), setOn(true))}
      style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        marginTop: 10, padding: "4px 10px",
        fontSize: 12, fontWeight: 450,
        color: on ? R.red : R.text2,
        background: on ? R.redSurf : R.surf1,
        border: `1px solid ${on ? "#3f1010" : R.border1}`,
        borderRadius: 6, cursor: "pointer", fontFamily: "inherit",
        transition: "all 160ms",
      }}
    >
      {on ? <IcoStop/> : <IcoPlay/>}{on ? "Stop" : "Play response"}
    </button>
  );
}

function Bubble({ msg }: { msg: Message }) {
  const me = msg.role === "user";
  return (
    <div style={{ display: "flex", justifyContent: me ? "flex-end" : "flex-start" }}>
      <div style={{
        maxWidth: "72%", padding: "11px 15px",
        borderRadius: me ? "12px 12px 3px 12px" : "12px 12px 12px 3px",
        fontSize: 14, lineHeight: 1.65,
        background: me ? R.surf3 : msg.is_refusal ? R.redSurf : R.surf1,
        color: me ? R.text0 : msg.is_refusal ? R.red : R.text1,
        border: `1px solid ${me ? R.border2 : msg.is_refusal ? "#3f1010" : R.border1}`,
      }}>
        <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{msg.text}</p>
        {!me && msg.audio_b64 && <PlayButton b64={msg.audio_b64}/>}
        {!me && msg.guardrail_triggered && msg.guardrail_reason && (
          <p style={{ margin: "6px 0 0", fontSize: 12, color: R.amber }}>⚠ {msg.guardrail_reason}</p>
        )}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div style={{ display: "flex" }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 5,
        padding: "12px 15px",
        borderRadius: "12px 12px 12px 3px",
        background: R.surf1, border: `1px solid ${R.border1}`,
      }}>
        {[0,1,2].map(i => (
          <span key={i} style={{
            width: 5, height: 5, borderRadius: "50%", background: R.text3,
            display: "inline-block",
            animation: "hhDot 1.2s ease-in-out infinite",
            animationDelay: `${i*0.18}s`,
          }}/>
        ))}
      </div>
    </div>
  );
}

/* ─── Main ───────────────────────────────────────────────────────────────── */

export default function Home() {
  const [messages,  setMessages ] = useState<Message[]>([]);
  const [input,     setInput    ] = useState("");
  const [recording, setRecording] = useState<RecordingState>("idle");
  const [loading,   setLoading  ] = useState(false);
  const [autoplay,  setAutoplay ] = useState(true);
  const [backend,   setBackend  ] = useState<BackendStatus>("checking");
  const [chatMode,  setChatMode ] = useState(false);
  const [focused,   setFocused  ] = useState(false);

  const mediaRef  = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLTextAreaElement>(null);

  /* health-check */
  useEffect(() => {
    let dead = false;
    const ping = async () => {
      try {
        const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        if (!dead) setBackend(r.ok ? "up" : "down");
      } catch { if (!dead) setBackend("down"); }
    };
    ping();
    const id = setInterval(ping, 15_000);
    return () => { dead = true; clearInterval(id); };
  }, []);

  const scrollEnd  = useCallback(() => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50), []);
  const pushMsg    = useCallback((m: Omit<Message,"id">) => { setMessages(p => [...p,{...m,id:nextId()}]); scrollEnd(); }, [scrollEnd]);
  const onResponse = useCallback((d: AgentResponse) => {
    pushMsg({ role:"assistant", text:d.answer, is_refusal:d.is_refusal, audio_b64:d.audio_b64, guardrail_triggered:d.guardrail_triggered, guardrail_reason:d.guardrail_reason });
    if (autoplay && d.audio_b64) player.play(d.audio_b64);
  }, [pushMsg, autoplay]);

  const sendText = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;
    setChatMode(true); setInput(""); pushMsg({ role:"user", text:q }); setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/ask-text`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question:q,tts:true}) });
      if (!r.ok) throw new Error(`${r.status}`);
      onResponse(await r.json()); setBackend("up");
    } catch(e) {
      if (e instanceof Error && e.message.toLowerCase().includes("failed to fetch")) setBackend("down");
      pushMsg({ role:"assistant", text:friendlyError(e), is_refusal:true });
    } finally { setLoading(false); }
  }, [input, loading, pushMsg, onResponse]);

  const startRec = useCallback(async () => {
    if (recording !== "idle") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio:true });
      const mime   = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      const mr     = new MediaRecorder(stream, { mimeType:mime });
      chunksRef.current = [];
      mr.ondataavailable = e => { if (e.data.size>0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        setRecording("processing"); setLoading(true); setChatMode(true);
        const fd = new FormData();
        fd.append("audio", new Blob(chunksRef.current,{type:mime}), "recording.webm");
        try {
          const r = await fetch(`${API_BASE}/ask?tts=true`, { method:"POST", body:fd });
          if (!r.ok) throw new Error(`${r.status}`);
          const d: AgentResponse = await r.json();
          pushMsg({ role:"user", text:`🎤 ${d.transcript}` });
          onResponse(d); setBackend("up");
        } catch(e) {
          if (e instanceof Error && e.message.toLowerCase().includes("failed to fetch")) setBackend("down");
          pushMsg({ role:"assistant", text:friendlyError(e), is_refusal:true });
        } finally { setLoading(false); setRecording("idle"); }
      };
      mediaRef.current = mr; mr.start(); setRecording("recording");
    } catch { alert("Microphone access denied."); }
  }, [recording, pushMsg, onResponse]);

  const stopRec  = useCallback(() => mediaRef.current?.stop(), []);
  const onKey    = (e: React.KeyboardEvent<HTMLTextAreaElement>) => { if (e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendText();} };

  const canSend  = input.trim().length > 0 && !loading && recording === "idle";
  const micBusy  = loading && recording === "idle";

  /* ── Input bar ─────────────────────────────────────────────────────────── */
  const InputBar = ({ ph }: { ph: string }) => (
    <div style={{
      display: "flex", alignItems: "center", gap: 6,
      background: R.surf0,
      border: `1px solid ${focused ? R.border2 : R.border1}`,
      borderRadius: 12,
      padding: "10px 10px 10px 18px",
      /* very subtle inner highlight — material feel */
      backgroundImage: "linear-gradient(180deg,rgba(255,255,255,0.02) 0%,transparent 100%)",
      boxShadow: focused ? `0 0 0 1px ${R.border2}` : "none",
      transition: "border-color 180ms, box-shadow 180ms",
      width: "100%", boxSizing: "border-box" as const,
    }}>
      <textarea
        ref={inputRef}
        rows={1}
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={onKey}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        placeholder={ph}
        disabled={loading || recording !== "idle"}
        style={{
          flex:1, resize:"none", background:"transparent", border:"none", outline:"none",
          fontSize:14, color:R.text0, lineHeight:1.5, fontFamily:"inherit",
          maxHeight:120, overflowY:"auto", padding:"2px 0",
        }}
      />
      {/* Mic */}
      <button
        onClick={recording==="recording" ? stopRec : startRec}
        disabled={micBusy}
        aria-label={recording==="recording" ? "Stop recording" : "Voice input"}
        style={{
          flexShrink:0, width:36, height:36,
          display:"flex", alignItems:"center", justifyContent:"center",
          border:`1px solid ${recording==="recording" ? "#5f1010" : R.border1}`,
          borderRadius:8,
          background:recording==="recording" ? R.redSurf : R.surf2,
          color:recording==="recording" ? R.red : R.text2,
          cursor:micBusy?"not-allowed":"pointer",
          opacity:micBusy?0.35:1,
          transition:"all 180ms", fontFamily:"inherit",
        }}
      >
        {recording==="processing"
          ? <span style={{fontSize:14,animation:"hhSpin .7s linear infinite",display:"inline-block"}}>↻</span>
          : <IcoMic/>}
      </button>
      {/* Send */}
      <button
        onClick={sendText}
        disabled={!canSend}
        aria-label="Send"
        style={{
          flexShrink:0, width:36, height:36,
          display:"flex", alignItems:"center", justifyContent:"center",
          border:`1px solid ${canSend ? "rgba(255,255,255,0.2)" : R.border1}`,
          borderRadius:8,
          background:canSend ? R.text0 : R.surf2,
          color:canSend ? "#000" : R.text3,
          cursor:canSend?"pointer":"not-allowed",
          transition:"all 180ms", fontFamily:"inherit",
        }}
      >
        <IcoSend/>
      </button>
    </div>
  );

  /* ── Render ──────────────────────────────────────────────────────────────── */
  return (
    <>
      <style>{`
        *,*::before,*::after{box-sizing:border-box}
        html,body{height:100%;margin:0}
        body{
          font-family:'Inter',var(--font-geist-sans),ui-sans-serif,system-ui,-apple-system,sans-serif;
          background:#000;color:#fff;
          -webkit-font-smoothing:antialiased;
          -moz-osx-font-smoothing:grayscale;
        }
        textarea{font-family:inherit}
        textarea::placeholder{color:#525252}
        button{font-family:inherit}

        /* material chip hover */
        .hh-chip:hover{
          background:#1A1A1A!important;
          border-color:#333!important;
          color:#e0e0e0!important;
        }
        .hh-btn:hover{background:#1A1A1A!important}
        .hh-back:hover{background:#1A1A1A!important}

        @keyframes hhDot{0%,80%,100%{transform:translateY(0);opacity:.25}40%{transform:translateY(-5px);opacity:.9}}
        @keyframes hhSpin{to{transform:rotate(360deg)}}
        @keyframes hhFade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        .hh-fade{animation:hhFade 0.45s cubic-bezier(.22,1,.36,1) both}

        @media(max-width:640px){
          .hh-hero{font-size:34px!important;letter-spacing:-0.03em!important}
          .hh-wrap{padding:0 20px 64px!important}
          .hh-nav{padding:0 20px!important}
          .hh-pt{padding-top:72px!important}
        }
      `}</style>

      <div style={{ display:"flex", flexDirection:"column", height:"100svh", background:R.bg, color:R.text0 }}>

        {/* ── NAVBAR ─────────────────────────────────────────────────────── */}
        <header className="hh-nav" style={{
          flexShrink:0, height:64,
          display:"flex", alignItems:"center", justifyContent:"space-between",
          padding:"0 32px",
          borderBottom:`1px solid ${R.border0}`,
          background:R.bg,
          /* very subtle inner bottom highlight — physical surface feel */
          backgroundImage:"linear-gradient(180deg,rgba(255,255,255,0.015) 0%,transparent 100%)",
        }}>
          {/* Brand */}
          <div style={{ display:"flex", alignItems:"center", gap:11 }}>
            {/* HH mark */}
            <div style={{
              width:30, height:30, borderRadius:8,
              background:R.surf1,
              border:`1px solid ${R.border1}`,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:11, fontWeight:700, color:R.text0, letterSpacing:"-0.4px",
              /* subtle top highlight */
              backgroundImage:"linear-gradient(160deg,rgba(255,255,255,0.06) 0%,transparent 60%)",
            }}>
              HH
            </div>
            <span style={{ fontSize:14, fontWeight:500, color:R.text0, letterSpacing:"-0.2px" }}>
              HH Goa 2026
            </span>
          </div>

          {/* Right controls */}
          <div style={{ display:"flex", alignItems:"center", gap:16 }}>
            {/* Status */}
            <div style={{ display:"flex", alignItems:"center", gap:7 }}>
              <span style={{
                width:6, height:6, borderRadius:"50%",
                background: backend==="up" ? R.green : backend==="checking" ? R.amber : R.red,
              }}/>
              <span style={{
                fontSize:13, fontWeight:450, letterSpacing:"0.01em",
                color: backend==="up" ? R.green : backend==="checking" ? R.amber : R.red,
              }}>
                {backend==="up" ? "Online" : backend==="checking" ? "Connecting" : "Offline"}
              </span>
            </div>
            {/* Sound */}
            <button
              className="hh-btn"
              onClick={() => setAutoplay(v=>!v)}
              aria-label={autoplay?"Mute":"Unmute"}
              style={{
                width:34, height:34, borderRadius:8,
                display:"flex", alignItems:"center", justifyContent:"center",
                background:R.surf1, border:`1px solid ${R.border1}`,
                color:autoplay?R.text1:R.text3,
                cursor:"pointer", transition:"all 160ms",
              }}
            >
              {autoplay ? <IcoVol/> : <IcoMute/>}
            </button>
          </div>
        </header>

        {/* ── BODY ───────────────────────────────────────────────────────── */}
        <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>

          {!chatMode ? (

            /* ── LANDING ───────────────────────────────────────────────── */
            <div className="hh-wrap" style={{
              flex:1, overflowY:"auto",
              display:"flex", flexDirection:"column", alignItems:"center",
              padding:"0 32px 72px",
              /* very subtle radial material tonal variation — not a glow, just depth */
              backgroundImage:"radial-gradient(ellipse 800px 500px at 50% 30%,rgba(20,20,20,0.9) 0%,transparent 100%)",
            }}>
              <div className="hh-pt hh-fade" style={{
                width:"100%", maxWidth:760,
                display:"flex", flexDirection:"column", alignItems:"center",
                paddingTop:108,
              }}>

                {/* HG assistant mark */}
                <div style={{
                  width:52, height:52, borderRadius:13,
                  background:R.surf1,
                  border:`1px solid ${R.border2}`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontSize:15, fontWeight:700, color:R.text0, letterSpacing:"-0.5px",
                  marginBottom:36,
                  /* subtle top highlight — material depth */
                  backgroundImage:"linear-gradient(160deg,rgba(255,255,255,0.08) 0%,rgba(255,255,255,0) 55%)",
                  boxShadow:"0 1px 0 0 rgba(255,255,255,0.04) inset",
                }}>
                  HG
                </div>

                {/* Hero heading */}
                <h1 className="hh-hero" style={{
                  margin:"0 0 18px",
                  fontSize:"clamp(38px,5.5vw,58px)",
                  fontWeight:700,
                  lineHeight:1.08,
                  letterSpacing:"-0.045em",
                  color:R.text0,
                  textAlign:"center",
                  maxWidth:640,
                }}>
                  Hey! I&apos;m your<br/>
                  <span style={{ color:R.text0 }}>Hacker House </span>
                  <span style={{ color:R.green }}>Goa 2026</span>
                  <span style={{ color:R.text0 }}> assistant</span>
                </h1>

                {/* Description */}
                <p style={{
                  margin:"0 0 44px",
                  fontSize:16, color:R.text2, lineHeight:1.65,
                  textAlign:"center", maxWidth:480,
                }}>
                  Ask me anything — dates, tasks, rules, schedule, prizes — or just say hi 👋
                </p>

                {/* Input */}
                <div style={{ width:"100%", maxWidth:680, marginBottom:12 }}>
                  <InputBar ph="Type or tap the mic to speak in any Indian language…"/>
                </div>

                {recording==="recording" && (
                  <p style={{ fontSize:13, color:R.red, margin:"4px 0 0", textAlign:"center" }}>
                    Recording — tap mic to stop
                  </p>
                )}

                {/* Suggestion label */}
                <p style={{
                  margin:"36px 0 14px",
                  fontSize:12, fontWeight:500,
                  color:R.text3,
                  textAlign:"center",
                  letterSpacing:"0.08em",
                  textTransform:"uppercase",
                }}>
                  You can ask me about
                </p>

                {/* Chips */}
                <div style={{ display:"flex", flexWrap:"wrap", gap:8, justifyContent:"center", maxWidth:720 }}>
                  {CHIPS.map(c => (
                    <button
                      key={c.label}
                      className="hh-chip"
                      onClick={() => { setInput(c.label); setChatMode(true); setTimeout(()=>inputRef.current?.focus(),100); }}
                      style={{
                        display:"inline-flex", alignItems:"center", gap:7,
                        padding:"7px 13px",
                        fontSize:13, fontWeight:450, color:R.text1,
                        background:R.surf1,
                        border:`1px solid ${R.border1}`,
                        borderRadius:8, cursor:"pointer",
                        lineHeight:1, fontFamily:"inherit",
                        transition:"all 160ms",
                      }}
                    >
                      <span style={{ fontSize:13 }}>{c.icon}</span>
                      {c.label}
                    </button>
                  ))}
                </div>

              </div>
            </div>

          ) : (

            /* ── CHAT ──────────────────────────────────────────────────── */
            <>
              <div style={{ flex:1, overflowY:"auto", padding:"28px 32px 16px" }}>
                <div style={{ maxWidth:680, margin:"0 auto", display:"flex", flexDirection:"column", gap:12 }}>
                  {messages.map(m => <Bubble key={m.id} msg={m}/>)}
                  {loading && <TypingDots/>}
                  <div ref={bottomRef}/>
                </div>
              </div>

              <div style={{
                flexShrink:0, padding:"12px 32px 24px",
                borderTop:`1px solid ${R.border0}`, background:R.bg,
              }}>
                {recording==="recording" && (
                  <p style={{ fontSize:13, color:R.red, textAlign:"center", margin:"0 0 10px" }}>
                    Recording — tap mic to stop
                  </p>
                )}
                <div style={{ maxWidth:680, margin:"0 auto", display:"flex", alignItems:"center", gap:8 }}>
                  <button
                    className="hh-back"
                    onClick={() => setChatMode(false)}
                    aria-label="Back"
                    style={{
                      flexShrink:0, width:36, height:36, borderRadius:8,
                      display:"flex", alignItems:"center", justifyContent:"center",
                      background:R.surf1, border:`1px solid ${R.border1}`,
                      color:R.text2, cursor:"pointer", transition:"all 160ms",
                    }}
                  >
                    <IcoBack/>
                  </button>
                  <InputBar ph="Ask about HHGoa 2026… (Enter to send)"/>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
