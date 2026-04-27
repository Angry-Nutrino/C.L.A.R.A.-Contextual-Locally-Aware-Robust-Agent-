import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Terminal, Cpu, Send, Paperclip, X, Zap, Activity,
  Shield, User, Copy, Check, ChevronRight, Radio,
  Layers, Clock, AlertCircle
} from "lucide-react";
import useClara from "./hooks/useClara";

// ─── tiny hook: copy to clipboard ───────────────────────────────────────────
function useCopy(timeout = 1500) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback((text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), timeout);
    });
  }, [timeout]);
  return [copied, copy];
}

// ─── Task board card ─────────────────────────────────────────────────────────
function TaskCard({ task, exiting }) {
  const isBackground = task.goal.startsWith("[BACKGROUND]") || task.goal.startsWith("[ENVIRONMENT]");
  const cleanGoal = task.goal
    .replace(/^\[BACKGROUND\]\s*/, "")
    .replace(/^\[ENVIRONMENT\]\s*/, "")
    .replace(/^\[AUTONOMOUS\]\s*/, "");

  const stateConfig = {
    pending:   { dot: "bg-amber-400",   border: "border-amber-500/20",   label: "QUEUED"  },
    active:    { dot: "bg-blue-400 animate-pulse", border: "border-blue-500/30", label: "ACTIVE"  },
    running:   { dot: "bg-emerald-400 animate-pulse", border: "border-emerald-500/40 shadow-[0_0_12px_rgba(16,185,129,0.15)]", label: "RUNNING" },
    completed: { dot: "bg-emerald-500", border: "border-emerald-500/10", label: "DONE"    },
    failed:    { dot: "bg-red-500",     border: "border-red-500/30",     label: "FAILED"  },
    paused:    { dot: "bg-yellow-400",  border: "border-yellow-500/20",  label: "PAUSED"  },
  };

  const cfg = stateConfig[task.state] || stateConfig.pending;
  const priorityPct = Math.round((task.priority || 0.5) * 100);
  const priorityColor = task.priority >= 0.9 ? "bg-red-500" : task.priority >= 0.5 ? "bg-amber-400" : "bg-blue-400";

  return (
    <div className={`
      task-card relative rounded-lg border p-3 mb-2 overflow-hidden
      ${cfg.border}
      ${isBackground ? "opacity-60" : ""}
      ${exiting ? "task-card-exit" : "task-card-enter"}
      ${task.state === "failed" ? "task-card-shake" : ""}
      bg-black/30 backdrop-blur-sm transition-all duration-300
    `}>
      {/* priority bar */}
      <div className="absolute bottom-0 left-0 h-[2px] w-full bg-white/5">
        <div
          className={`h-full ${priorityColor} transition-all duration-700`}
          style={{ width: `${priorityPct}%` }}
        />
      </div>

      <div className="flex items-start gap-2">
        <span className={`mt-1 flex-shrink-0 w-2 h-2 rounded-full ${cfg.dot}`} />
        <div className="flex-1 min-w-0">
          <p className="text-[11px] text-gray-200 font-mono leading-snug truncate">{cleanGoal}</p>
          <div className="flex items-center gap-2 mt-1">
            <span className={`text-[9px] font-bold tracking-widest ${
              task.state === "running" ? "text-emerald-400" :
              task.state === "failed"  ? "text-red-400" :
              task.state === "completed" ? "text-emerald-500/60" : "text-white/30"
            }`}>{cfg.label}</span>
            {task.source === "user" && (
              <span className="text-[9px] text-purple-400/70 font-mono">USER</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Message bubble ──────────────────────────────────────────────────────────
function MessageBubble({ msg, index, messages, onQuote }) {
  const [hovered, setHovered] = useState(false);
  const [copied, copy] = useCopy();
  const isClara = msg.sender === "Clara";

  const replyTarget = isClara && msg.messageId
    ? messages.find(m => m.sender === "User" && m.messageId === msg.messageId)
    : null;

  return (
    <div
      className={`flex msg-enter ${isClara ? "justify-start" : "justify-end"}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className={`relative max-w-[80%] group`}>
        {/* hover actions */}
        <div className={`
          absolute -top-7 ${isClara ? "left-0" : "right-0"}
          flex items-center gap-1 transition-all duration-150
          ${hovered ? "opacity-100 translate-y-0" : "opacity-0 translate-y-1 pointer-events-none"}
        `}>
          <span className="text-[9px] font-mono text-white/30 px-2">{msg.time}</span>
          {isClara && (
            <button
              onClick={() => copy(msg.text)}
              className="p-1 rounded bg-black/60 border border-white/10 hover:border-emerald-500/40 transition-colors"
            >
              {copied ? <Check size={10} className="text-emerald-400" /> : <Copy size={10} className="text-white/40" />}
            </button>
          )}
        </div>

        {/* bubble */}
        <div className={`
          p-4 rounded-2xl flex flex-col gap-2 transition-all duration-150
          ${isClara
            ? "bg-gradient-to-br from-emerald-950/60 to-black/60 border border-emerald-500/20 text-emerald-50 shadow-[0_0_20px_rgba(16,185,129,0.08)] hover:shadow-[0_0_25px_rgba(16,185,129,0.12)]"
            : "bg-gradient-to-br from-[#1c1c1c] to-[#141414] border border-white/8 text-gray-200 hover:border-white/12"
          }
        `}>
          {/* image */}
          {msg.image && (
            <img
              src={msg.image}
              alt="Upload"
              className="w-full h-auto max-h-56 object-cover rounded-xl border border-white/10 cursor-zoom-in hover:brightness-110 transition-all"
            />
          )}

          {/* reply attribution */}
          {replyTarget && (
            <div className="flex items-start gap-2 pb-2 mb-1 border-b border-emerald-500/10">
              <div className="w-0.5 h-full bg-emerald-500/40 rounded-full flex-shrink-0 self-stretch min-h-3" />
              <span className="text-[10px] text-emerald-400/50 font-mono leading-relaxed italic truncate">
                {replyTarget.text.slice(0, 60)}{replyTarget.text.length > 60 ? "…" : ""}
              </span>
            </div>
          )}

          {/* content */}
          {isClara ? (
            <div className="prose prose-invert prose-sm max-w-none leading-relaxed
              prose-code:bg-black/50 prose-code:text-emerald-300 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-[11px]
              prose-pre:bg-black/70 prose-pre:border prose-pre:border-emerald-500/10 prose-pre:rounded-xl prose-pre:text-xs
              prose-a:text-emerald-400 prose-strong:text-emerald-100 prose-headings:text-white
              prose-p:text-emerald-50/90 prose-li:text-emerald-50/80">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
            </div>
          ) : (
            <p className="whitespace-pre-wrap leading-relaxed text-sm">{msg.text}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Vitals bar ──────────────────────────────────────────────────────────────
function VitalBar({ label, value, icon: Icon, color = "emerald", warn = 85 }) {
  const pct = parseFloat(value) || 0;
  const isWarn = pct >= warn;
  const barColor = isWarn
    ? "bg-amber-400"
    : color === "blue" ? "bg-blue-400"
    : color === "yellow" ? "bg-yellow-400"
    : "bg-emerald-400";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-white/40">
          <Icon size={10} className={isWarn ? "text-amber-400" : "text-white/30"} />
          <span>{label}</span>
        </div>
        <span className={`text-[10px] font-mono ${isWarn ? "text-amber-400" : "text-white/30"}`}>
          {value}
        </span>
      </div>
      <div className="h-[3px] bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} rounded-full transition-all duration-700 ease-out vital-bar-fill`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}

// ─── Main Layout ─────────────────────────────────────────────────────────────
export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen]     = useState(true);
  const [isNeuralOpen, setIsNeuralOpen]       = useState(true);
  const [viewImage, setViewImage]             = useState(null);
  const [isFocused, setIsFocused]             = useState(false);
  const [soul, setSoul]                       = useState(null);
  const [quotePopup, setQuotePopup]           = useState(null);
  const [currentMode, setCurrentMode]         = useState(null); // FAST/CHAT/DELIBERATE

  const {
    messages, thoughts, tasks, input, setInput,
    sendMessage, status, selectedImage, setSelectedImage,
    handleImageUpload, streamingContent, clearHistory, lastTokenUsage
  } = useClara();

  const chatEndRef   = useRef(null);
  const neuralEndRef = useRef(null);
  const textareaRef  = useRef(null);

  // soul vitals polling
  useEffect(() => {
    const fetchSoul = () =>
      fetch("http://localhost:8001/soul")
        .then(r => r.json())
        .then(setSoul)
        .catch(() => {});
    fetchSoul();
    const id = setInterval(fetchSoul, 5000);
    return () => clearInterval(id);
  }, []);

  // auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // auto-scroll neural
  useEffect(() => {
    neuralEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [thoughts]);

  // auto-expand textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [input]);

  // detect mode from thoughts
  useEffect(() => {
    const last = thoughts[thoughts.length - 1];
    if (!last) return;
    if (last.text?.includes("FAST")) setCurrentMode("FAST");
    else if (last.text?.includes("DELIBERATE")) setCurrentMode("DELIBERATE");
    else if (last.text?.includes("CHAT")) setCurrentMode("CHAT");
  }, [thoughts]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleQuote = (text, sender) => {
    const label = sender === "Clara" ? "[Clara]" : "[Alkama]";
    setInput(prev => `> ${label}: ${text}\n\n${prev}`);
    setQuotePopup(null);
    window.getSelection()?.removeAllRanges();
    textareaRef.current?.focus();
  };

  const modeChip = {
    FAST:       { color: "text-amber-400 border-amber-500/30 bg-amber-500/10",      pulse: false },
    CHAT:       { color: "text-blue-400 border-blue-500/30 bg-blue-500/10",         pulse: false },
    DELIBERATE: { color: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10", pulse: true  },
  };

  // parse vitals percentages
  const ramPct  = soul ? parseFloat(soul.vitals?.memory_usage)  || 0 : 0;
  const vramPct = soul ? (() => {
    const s = soul.vitals?.gpu || "";
    const m = s.match(/(\d+\.?\d*)GB\s*\/\s*(\d+\.?\d*)GB/);
    return m ? Math.round((parseFloat(m[1]) / parseFloat(m[2])) * 100) : 0;
  })() : 0;
  const cpuPct  = soul ? parseFloat(soul.vitals?.cpu) || 0 : 0;

  return (
    <div className="flex h-screen w-full bg-[#050505] text-gray-200 overflow-hidden"
      onMouseUp={() => {
        const sel = window.getSelection();
        const text = sel?.toString().trim();
        if (!text) { setQuotePopup(null); return; }
        let node = sel.anchorNode;
        while (node && !node.dataset?.msgIndex) node = node.parentElement;
        const sender = node ? messages[parseInt(node.dataset.msgIndex)]?.sender : null;
        const range = sel.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setQuotePopup({ x: rect.left + rect.width / 2, y: rect.top - 10, text, sender });
      }}
      onClick={(e) => {
        if (!window.getSelection()?.toString().trim()) setQuotePopup(null);
      }}
    >

      {/* ── ZONE A: SIDEBAR ──────────────────────────────────────────────── */}
      <aside className={`
        relative flex flex-col bg-black/50 border-r border-white/5
        backdrop-blur-xl overflow-hidden transition-all duration-300 ease-in-out hidden md:flex
        ${isSidebarOpen ? "w-72" : "w-0 border-none"}
      `}>
        {/* subtle scanline texture */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.5) 2px, rgba(255,255,255,0.5) 3px)" }}
        />

        {/* header */}
        <div className="p-5 border-b border-white/5 flex-shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="relative">
              <Terminal size={18} className="text-emerald-400" />
              <span className="absolute -bottom-0.5 -right-0.5 w-1.5 h-1.5 bg-emerald-500 rounded-full shadow-[0_0_6px_2px_rgba(16,185,129,0.6)] animate-[breathe_2.5s_ease-in-out_infinite]" />
            </div>
            <h1 className="text-sm font-bold text-white tracking-[0.15em] font-mono">C.L.A.R.A.</h1>
          </div>
          <p className="text-[9px] text-emerald-400/40 font-mono tracking-[0.25em] mt-1.5 ml-[26px]">
            SYSTEM ONLINE · {soul?.version || "v2.6"}
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6 scrollbar-thin">

          {/* identity */}
          {soul && (
            <div className="space-y-2">
              <p className="text-[9px] uppercase tracking-[0.2em] text-white/20 font-mono flex items-center gap-1.5">
                <User size={9} /> Operator
              </p>
              <div className="rounded-xl bg-white/[0.03] border border-white/5 p-3.5 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-30 transition-opacity">
                  <Shield size={20} />
                </div>
                <p className="text-base font-semibold text-white font-mono">{soul.identity.name}</p>
                <p className="text-[11px] text-emerald-400/80 mt-0.5">{soul.identity.role}</p>
                <div className="flex items-center gap-2 mt-3 pt-2.5 border-t border-white/5">
                  <span className="text-[9px] text-white/30 font-mono">{soul.identity.location}</span>
                  <span className="w-0.5 h-2.5 bg-white/10 rounded-full" />
                  <span className="text-[9px] text-white/30 font-mono">{soul.identity.clearance}</span>
                </div>
              </div>
            </div>
          )}

          {/* active context — derived from recent thoughts/tasks */}
          <div className="space-y-2">
            <p className="text-[9px] uppercase tracking-[0.2em] text-white/20 font-mono flex items-center gap-1.5">
              <Radio size={9} /> Active Context
            </p>
            <div className="rounded-xl bg-gradient-to-br from-emerald-950/20 to-transparent border border-emerald-500/10 p-3 border-l-2 border-l-emerald-500/40">
              {tasks.filter(t => t.state === "running" || t.state === "active").length > 0 ? (
                tasks.filter(t => t.state === "running" || t.state === "active").slice(0, 2).map((t, i) => (
                  <p key={i} className="text-[11px] text-emerald-100/70 font-mono leading-relaxed truncate">
                    {t.goal.replace(/^\[.*?\]\s*/, "").slice(0, 55)}
                    {t.goal.length > 55 ? "…" : ""}
                  </p>
                ))
              ) : (
                <p className="text-[11px] text-white/20 font-mono italic">Standing by</p>
              )}
              {soul?.mission?.phase && (
                <p className="text-[9px] text-white/20 font-mono mt-1.5">{soul.mission.phase}</p>
              )}
            </div>
          </div>

          {/* skills */}
          {soul?.skills?.length > 0 && (
            <div className="space-y-2">
              <p className="text-[9px] uppercase tracking-[0.2em] text-white/20 font-mono flex items-center gap-1.5">
                <Zap size={9} /> Competency Matrix
              </p>
              <div className="flex flex-wrap gap-1.5">
                {soul.skills.map((skill, i) => (
                  <span key={i} className="text-[9px] px-2 py-1 rounded-lg bg-white/[0.04] border border-white/8
                    text-white/40 hover:text-emerald-300 hover:border-emerald-500/30 hover:bg-emerald-500/5
                    transition-all duration-200 cursor-default font-mono">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* vitals footer */}
        {soul && (
          <div className="p-4 border-t border-white/5 bg-black/30 space-y-3 flex-shrink-0">
            <VitalBar label="CPU" value={`${cpuPct}%`} icon={Cpu} color="emerald" warn={80} />
            <VitalBar label="RAM" value={`${ramPct}%`} icon={Activity} color="blue" warn={90} />
            <VitalBar label="VRAM" value={`${vramPct}%`} icon={Zap} color="yellow" warn={80} />
          </div>
        )}
      </aside>

      {/* ── ZONE B: CHAT ─────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col relative h-screen overflow-hidden bg-[#080808]">

        {/* header */}
        <header className="h-13 border-b border-white/5 flex items-center justify-between px-4
          bg-[#080808]/90 backdrop-blur-md sticky top-0 z-10 flex-shrink-0">
          <button onClick={() => setIsSidebarOpen(p => !p)}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white/70">
            <Layers size={18} />
          </button>

          <div className="flex items-center gap-2">
            {/* mode chip */}
            {currentMode && status !== "idle" && status !== "disconnected" && (
              <span className={`text-[9px] font-bold font-mono px-2 py-0.5 rounded border tracking-widest
                ${modeChip[currentMode]?.color || "text-white/30 border-white/10"}
                ${modeChip[currentMode]?.pulse ? "animate-pulse" : ""}
              `}>
                {currentMode}
              </span>
            )}

            {/* status pill */}
            <span className={`text-[10px] font-bold px-3 py-1 rounded-full border transition-all
              ${status === "thinking" || status === "typing"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30 animate-pulse"
                : status === "disconnected"
                ? "bg-red-500/10 text-red-400 border-red-500/30"
                : "opacity-0 border-transparent"}`
            }>
              {status === "thinking" ? "PROCESSING" : status === "typing" ? "RESPONDING" :
               status === "disconnected" ? "OFFLINE" : "ONLINE"}
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button onClick={clearHistory}
              className="text-[9px] font-mono px-2 py-1 rounded-lg border border-white/8
              text-white/20 hover:text-red-400 hover:border-red-500/30 transition-all">
              CLEAR
            </button>
            <button onClick={() => setIsNeuralOpen(p => !p)}
              className="p-2 hover:bg-white/5 rounded-lg transition-colors text-purple-400/60 hover:text-purple-400">
              <Cpu size={18} />
            </button>
          </div>
        </header>

        {/* persistent CLARA watermark — fixed behind all content */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0 select-none">
          <h1 className="text-[11rem] font-black text-white/[0.018] tracking-[0.4em] font-mono">CLARA</h1>
        </div>

        {/* messages */}
        <div className="relative z-10 flex-1 overflow-y-auto px-5 py-6 space-y-5 scroll-smooth pb-36 scrollbar-thin">
          {messages.length === 0 && !streamingContent ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 select-none">
              <div className="relative flex items-center justify-center">
                {/* outer slow pulse ring */}
                <div className="absolute w-28 h-28 rounded-full border border-emerald-500/8 animate-[breathe_4s_ease-in-out_infinite]" />
                {/* mid ring */}
                <div className="absolute w-20 h-20 rounded-full border border-emerald-500/12 animate-[breathe_4s_ease-in-out_infinite_0.6s]" />
                {/* inner ring */}
                <div className="absolute w-12 h-12 rounded-full border border-emerald-500/20 animate-[breathe_4s_ease-in-out_infinite_1.2s]" />
                {/* name */}
                <h1 className="text-3xl font-black text-white/8 tracking-[0.3em] font-mono z-10">CLARA</h1>
              </div>
              <p className="text-[9px] text-white/12 font-mono tracking-[0.4em] mt-1">READY</p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} data-msg-index={i}>
                <MessageBubble
                  msg={msg}
                  index={i}
                  messages={messages}
                  onQuote={(text, sender) => handleQuote(text, sender)}
                />
              </div>
            ))
          )}

          {/* streaming bubble */}
          {streamingContent && (
            <div className="flex justify-start msg-enter">
              <div className="max-w-[80%] p-4 rounded-2xl bg-gradient-to-br from-emerald-950/60
                to-black/60 border border-emerald-500/20 shadow-[0_0_20px_rgba(16,185,129,0.08)]">
                {streamingContent ? (
                  <div className="prose prose-invert prose-sm max-w-none leading-relaxed
                    prose-code:bg-black/50 prose-code:text-emerald-300 prose-code:px-1.5 prose-code:rounded
                    prose-pre:bg-black/70 prose-pre:border prose-pre:border-emerald-500/10 prose-pre:rounded-xl
                    prose-a:text-emerald-400 prose-strong:text-emerald-100 prose-p:text-emerald-50/90">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingContent}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="flex gap-1 items-center py-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-[breathe_1.2s_ease-in-out_infinite]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-[breathe_1.2s_ease-in-out_infinite_0.2s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-[breathe_1.2s_ease-in-out_infinite_0.4s]" />
                  </div>
                )}
                <span className="inline-block w-1.5 h-3.5 bg-emerald-400 animate-pulse ml-0.5 align-middle" />
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* input capsule */}
        <div className="absolute bottom-0 left-0 w-full px-5 pb-5 pt-8 z-40
          bg-gradient-to-t from-[#080808] via-[#080808]/95 to-transparent">

          {/* image preview */}
          {selectedImage && (
            <div className="mb-2 ml-1 flex items-center gap-2 bg-black/80 border border-emerald-500/20
              rounded-xl px-2.5 py-1.5 w-fit">
              <img src={selectedImage} alt="Preview" onClick={() => setViewImage(selectedImage)}
                className="h-8 w-8 object-cover rounded-lg border border-white/10 cursor-zoom-in" />
              <span className="text-[10px] text-emerald-400/70 font-mono">Image attached</span>
              <button onClick={() => setSelectedImage(null)}
                className="ml-1 text-white/25 hover:text-white/60 transition-colors">
                <X size={12} />
              </button>
            </div>
          )}

          <div className={`
            relative flex items-end gap-2 px-3 py-2.5 rounded-2xl border transition-all duration-300
            ${status === "thinking" || status === "typing"
              ? "bg-emerald-950/20 border-emerald-500/30 shadow-[0_0_40px_-8px_rgba(16,185,129,0.15)]"
              : isFocused
              ? "bg-[#0f0f0f] border-white/12 shadow-[0_0_60px_-15px_rgba(16,185,129,0.08)]"
              : "bg-[#0d0d0d] border-white/6"
            }
          `}>
            <button onClick={() => document.getElementById("file-upload").click()}
              className={`p-2.5 rounded-xl transition-colors flex-shrink-0
                ${selectedImage ? "text-emerald-400 bg-emerald-900/20" : "text-white/30 hover:text-white/60 hover:bg-white/5"}`}>
              <Paperclip size={18} />
            </button>
            <input type="file" id="file-upload" className="hidden" accept="image/*" onChange={handleImageUpload} />

            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={e => {
                const items = e.clipboardData?.items;
                if (!items) return;
                for (const item of items) {
                  if (item.type.startsWith("image/")) {
                    e.preventDefault();
                    const file = item.getAsFile();
                    const reader = new FileReader();
                    reader.onload = ev => setSelectedImage(ev.target.result);
                    reader.readAsDataURL(file);
                    break;
                  }
                }
              }}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Message Clara…"
              className="flex-1 bg-transparent text-gray-200 placeholder-white/15 focus:outline-none
                resize-none py-2.5 text-sm leading-relaxed max-h-36 font-[inherit]"
              rows={1}
              style={{ minHeight: "40px" }}
            />

            <button onClick={sendMessage}
              disabled={!input.trim() && !selectedImage}
              className={`p-2.5 rounded-xl transition-all duration-200 flex-shrink-0
                ${input.trim() || selectedImage
                  ? "bg-emerald-600 text-white shadow-[0_0_20px_rgba(16,185,129,0.35)] hover:bg-emerald-500 hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] active:scale-95"
                  : "bg-white/5 text-white/20 cursor-not-allowed"
                }`}>
              <Send size={18} />
            </button>
          </div>
        </div>
      </main>

      {/* ── ZONE C: NEURAL STREAM ────────────────────────────────────────── */}
      <aside className={`
        flex flex-col border-l border-white/5 bg-[#060606] transition-all duration-300
        ${isNeuralOpen ? "w-80" : "w-0 border-none overflow-hidden"}
      `}>
        <div className="p-4 border-b border-white/5 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2 text-purple-400/70">
            <Cpu size={16} className={status === "thinking" ? "animate-[spin_3s_linear_infinite]" : ""} />
            <span className="text-xs font-bold font-mono tracking-widest">NEURAL STREAM</span>
          </div>
        </div>

        <div className="flex-1 overflow-hidden flex flex-col min-h-0">

          {/* ── TOP: TASK BOARD ── */}
          <div className="flex-shrink-0 border-b border-emerald-500/10 px-3 pt-3 pb-2">
            <p className="text-[9px] uppercase tracking-[0.2em] text-white/20 font-mono mb-2 flex items-center gap-1.5">
              <Clock size={9} /> Task Board
            </p>
            <div className="space-y-0 max-h-52 overflow-y-auto scrollbar-thin">
              {tasks.length === 0 ? (
                <p className="text-[10px] text-white/15 font-mono italic py-2">No active tasks</p>
              ) : (
                tasks.slice(-12).map(t => (
                  <TaskCard key={t.task_id} task={t} />
                ))
              )}
            </div>
          </div>

          {/* ── BOTTOM: THOUGHT STREAM ── */}
          <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3 scrollbar-thin min-h-0">
            <p className="text-[9px] uppercase tracking-[0.2em] text-white/20 font-mono mb-1 flex items-center gap-1.5 sticky top-0 bg-[#060606] py-1">
              <AlertCircle size={9} /> Thought Stream
            </p>
            {thoughts.length === 0 ? (
              <p className="text-[10px] text-white/15 font-mono italic">Idle</p>
            ) : (
              thoughts.map((t, i) => {
                const isLast = i === thoughts.length - 1;
                return (
                  <div key={i} className={`
                    border-l-2 pl-3 py-1 relative transition-all duration-400
                    ${isLast
                      ? "border-emerald-500 opacity-100"
                      : "border-purple-500/15 opacity-40 hover:opacity-70"}
                  `}>
                    <span className={`block text-[9px] font-mono mb-0.5 ${isLast ? "text-emerald-400/70" : "text-purple-400/40"}`}>
                      {t.time}
                    </span>
                    <span className={`text-[11px] leading-relaxed whitespace-pre-wrap font-mono
                      ${isLast ? "text-emerald-100/80" : "text-gray-500"}`}>
                      {t.text}
                    </span>

                  </div>
                );
              })
            )}
            {lastTokenUsage && (
              <div className="token-usage-pill">
                <span className="token-label">Last query</span>
                <span className="token-stat">
                  {lastTokenUsage.total_tokens.toLocaleString()} tokens
                </span>
                <span className="token-divider">·</span>
                <span className="token-stat">
                  {lastTokenUsage.prompt_tokens.toLocaleString()} in
                </span>
                <span className="token-divider">·</span>
                <span className="token-stat">
                  {lastTokenUsage.completion_tokens.toLocaleString()} out
                </span>
                {lastTokenUsage.cached_tokens > 0 && (
                  <>
                    <span className="token-divider">·</span>
                    <span className="token-cached">
                      {lastTokenUsage.cached_tokens.toLocaleString()} cached
                    </span>
                  </>
                )}
              </div>
            )}
            <div ref={neuralEndRef} className="h-2" />
          </div>
        </div>
      </aside>

      {/* ── QUOTE POPUP ──────────────────────────────────────────────────── */}
      {quotePopup && (
        <button
          className="fixed z-50 text-[10px] font-mono font-bold px-3 py-1.5 rounded-full shadow-xl
            -translate-x-1/2 -translate-y-full
            bg-emerald-600/95 text-white border border-emerald-400/40
            hover:bg-emerald-500 hover:shadow-[0_0_16px_rgba(16,185,129,0.5)]
            transition-all duration-150"
          style={{ left: quotePopup.x, top: quotePopup.y }}
          onMouseDown={e => {
            e.preventDefault();
            handleQuote(quotePopup.text, quotePopup.sender);
          }}
        >
          QUOTE
        </button>
      )}

      {/* ── LIGHTBOX ─────────────────────────────────────────────────────── */}
      {viewImage && (
        <div
          className="fixed inset-0 z-50 bg-black/95 backdrop-blur-sm flex items-center justify-center p-6"
          onClick={() => setViewImage(null)}
        >
          <img src={viewImage} alt="Full"
            className="max-w-full max-h-full rounded-2xl shadow-2xl border border-white/10" />
          <button className="absolute top-5 right-5 text-white/40 hover:text-white transition-colors">
            <X size={28} />
          </button>
        </div>
      )}

    </div>
  );
}
