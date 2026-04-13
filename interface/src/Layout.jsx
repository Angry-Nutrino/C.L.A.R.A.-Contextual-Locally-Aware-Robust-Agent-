import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
// 1. UPDATED IMPORTS: Added Paperclip and X for the file upload UI
import { Terminal, Cpu, MessageSquare, Menu, Send, Paperclip, X , Zap, Activity, MapPin,
  Shield, Target, User, Disc
} from "lucide-react";
import useClara from "./hooks/useClara";
import Typewriter from "./components/Typewriter";

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isBrainOpen, setIsBrainOpen] = useState(true);
  const [viewImage, setViewImage] = useState(null);
  const [isFocused, setIsFocused] = useState(false);
  const [soul, setSoul] = useState(null);
  const [quotePopup, setQuotePopup] = useState(null); // { x, y, text, sender }

  useEffect(() => {
    // Poll the soul every 5 seconds to keep vitals "alive"
    const fetchSoul = () => {
      fetch("http://localhost:8001/soul")
        .then(res => res.json())
        .then(data => setSoul(data))
        .catch(err => console.error("Soul fetch error:", err));
    };
    
    fetchSoul(); // Initial fetch
    const interval = setInterval(fetchSoul, 5000); // Live update
    return () => clearInterval(interval);
  }, []);
  
  // 2. UPDATED HOOK: Getting the image tools from useClara
  const {
    messages, thoughts, input, setInput, sendMessage, status,
    selectedImage, setSelectedImage, handleImageUpload,
    streamingContent, clearHistory
  } = useClara();
  
  const chatEndRef = useRef(null);
  const brainEndRef = useRef(null);
  const textareaRef = useRef(null); // Ref for the auto-expanding box

  // Auto-scroll logic for Chat
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  
  // Auto-scroll logic for Brain
  useEffect(() => {
    brainEndRef.current?.scrollIntoView({ behavior: "smooth" });
    const timeoutId = setTimeout(() => {
      brainEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [thoughts]);

  // 3. AUTO-EXPAND LOGIC: Grow the text box as you type
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // Reset
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px'; // Grow to fit content
    }
  }, [input]);

  // 4. UPDATED KEY HANDLER: Shift+Enter = New Line, Enter = Send
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Stop new line
      sendMessage();
    }
  };

  return (
    <div className="flex h-screen w-full bg-[var(--bg-depth)] text-gray-200 overflow-hidden font-sans">
      
      {/* --- ZONE A: LEFT SIDEBAR --- */}
      <aside 
        className={`
          bg-black/40 border-r border-white/5 flex-col hidden md:flex backdrop-blur-md relative overflow-hidden transition-all duration-300 ease-in-out
          ${isSidebarOpen ? "w-72 translate-x-0 opacity-100" : "w-0 -translate-x-full opacity-0 border-none"}
        `}
      >
        
        {/* BACKGROUND NOISE (Optional Texture) */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-5 pointer-events-none"></div>

        {/* HEADER */}
        <div className="p-6 border-b border-white/5">
          <h1 className="text-xl font-bold text-emerald-500 tracking-wider flex items-center gap-2 font-mono">
            <Terminal size={20} />
            C.L.A.R.A.
          </h1>
          <div className="flex items-center gap-2 mt-2">
             <span className="relative flex h-2 w-2">
               <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
               <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
             </span>
             <p className="text-[10px] text-emerald-400/60 font-mono tracking-widest">SYSTEM ONLINE // V2.0.4</p>
          </div>
        </div>

        {/* DYNAMIC CONTENT */}
        {soul && (
          <div className="flex-1 overflow-y-auto p-6 space-y-8 scrollbar-hide">
            
            {/* 1. IDENTITY SECTION */}
            <div className="space-y-3">
              <h3 className="text-[10px] uppercase tracking-[0.2em] text-white/30 font-bold flex items-center gap-2">
                <User size={10} /> Operator Identity
              </h3>
              <div className="bg-white/5 rounded-lg p-4 border border-white/5 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-2 opacity-20 group-hover:opacity-100 transition-opacity">
                  <Shield size={16} />
                </div>
                <div className="text-lg font-medium text-white font-mono">{soul.identity.name}</div>
                <div className="text-xs text-emerald-400 font-mono mt-1">{soul.identity.role}</div>
                <div className="flex items-center gap-3 mt-4 pt-3 border-t border-white/5">
                  <div className="flex items-center gap-1 text-[10px] text-gray-400">
                    <MapPin size={10} /> {soul.identity.location}
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-gray-400">
                    <Disc size={10} /> {soul.identity.clearance}
                  </div>
                </div>
              </div>
            </div>

            {/* 2. MISSION STATUS */}
            <div className="space-y-3">
              <h3 className="text-[10px] uppercase tracking-[0.2em] text-white/30 font-bold flex items-center gap-2">
                <Target size={10} /> Current Objective
              </h3>
              <div className="bg-gradient-to-r from-emerald-900/10 to-transparent p-4 rounded border-l-2 border-emerald-500 relative">
                 <div className="text-xs text-emerald-100 font-medium">{soul.mission.current}</div>
                 <div className="flex justify-between items-center mt-2">
                    <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">
                      {soul.mission.status}
                    </span>
                    <span className="text-[10px] text-white/20 font-mono">{soul.mission.phase}</span>
                 </div>
              </div>
            </div>

            {/* 3. SKILL MATRIX (Tags) */}
            <div className="space-y-3">
              <h3 className="text-[10px] uppercase tracking-[0.2em] text-white/30 font-bold flex items-center gap-2">
                <Zap size={10} /> Competency Matrix
              </h3>
              <div className="flex flex-wrap gap-2">
                {soul.skills?.map((skill, i) => (
                  <span key={i} className="text-[10px] px-2 py-1 rounded bg-[#111] text-gray-300 border border-white/10 hover:border-emerald-500/50 hover:text-emerald-400 transition-colors cursor-default">
                    {skill}
                  </span>
                ))}
              </div>
            </div>

          </div>
        )}
        
        {/* 4. SYSTEM VITALS (Footer) */}
        {soul && (
          <div className="p-4 border-t border-white/5 bg-black/20 backdrop-blur-lg">
             {/* CPU ROW (Full Width) */}
             <div className="flex items-center gap-2 text-[10px] font-mono text-emerald-500/80 mb-2 border-b border-white/5 pb-2">
                <Cpu size={12} />
                <span className="truncate">{soul.vitals.cpu}</span>
             </div>

             {/* GPU & RAM ROW (Split) */}
             <div className="grid grid-cols-2 gap-2 text-[10px] font-mono text-white/40">
                <div className="flex items-center gap-2">
                  <Zap size={12} className="text-yellow-500/50"/> 
                  <span>{soul.vitals.gpu}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Activity size={12} className="text-blue-500/50"/> 
                  <span>RAM: {soul.vitals.memory_usage}</span>
                </div>
             </div>
          </div>
        )}
      </aside>

      {/* --- ZONE B: MAIN CHAT --- */}
      <main className="flex-1 flex flex-col relative h-screen overflow-hidden bg-[#0a0a0a]">
        
        {/* HEADER */}
        <header className="h-14 border-b border-[var(--border-subtle)] flex items-center justify-between px-4 bg-[var(--bg-depth)]/80 backdrop-blur-md sticky top-0 z-10">
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-white/5 rounded"><Menu size={20} /></button>
          
          <span className={`text-xs font-bold px-3 py-1 rounded-full transition-all border
            ${status === 'thinking' || status === 'typing'
              ? 'bg-emerald-500/20 text-emerald-400 animate-pulse border-emerald-500/50'
              : status === 'disconnected'
              ? 'bg-red-500/20 text-red-400 border-red-500/50 animate-pulse'
              : 'opacity-30 border-transparent'}`}>
            {status === 'thinking' || status === 'typing'
              ? 'PROCESSING...'
              : status === 'disconnected'
              ? 'DISCONNECTED'
              : 'IDLE'}
          </span>
          
          <div className="flex items-center gap-1">
            <button
              onClick={clearHistory}
              className="text-[10px] font-mono px-2 py-1 rounded border border-white/10 text-white/30 hover:text-red-400 hover:border-red-500/40 hover:bg-red-500/10 transition-all duration-200"
            >
              CLEAR
            </button>
            <button onClick={() => setIsBrainOpen(!isBrainOpen)} className="p-2 hover:bg-white/5 rounded text-purple-400"><Cpu size={20} /></button>
          </div>
        </header>

        {/* MESSAGES AREA */}
        <div
          className="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth pb-32"
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
          onClick={() => { if (!window.getSelection()?.toString().trim()) setQuotePopup(null); }}
        >
          {messages.length === 0 && !streamingContent ? (
            <div className="text-center mt-20 opacity-30">
              <h1 className="text-4xl font-black mb-2">INITIALIZED</h1>
              <p>Waiting for input...</p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} data-msg-index={i} className={`flex ${msg.sender === "User" ? "justify-end" : "justify-start"}`}>
                
                {/* 1. THE BUBBLE CONTAINER (Wrap everything here) */}
                <div className={`max-w-[80%] p-4 rounded-xl flex flex-col gap-2 
                  ${msg.sender === "User" 
                    ? "bg-[#222] border border-[#333]" 
                    : "bg-emerald-900/10 border border-emerald-500/20 text-emerald-100 shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                  }`}>
                  
                  {/* 2. IMAGE (Inside the bubble) */}
                  {msg.image && (
                    <div className="group relative w-full">
                      <img 
                        src={msg.image} 
                        alt="Upload" 
                        onClick={() => setViewImage(msg.image)} // Triggers the lightbox
                        className="w-full h-auto max-h-64 object-cover rounded-lg border border-white/10 cursor-zoom-in hover:brightness-110 transition-all"
                      />
                    </div>
                  )}

                  {/* 3. TEXT (Below the image) */}
                  {msg.sender === "Clara" && msg.messageId && (() => {
                    const userMsg = messages.find(
                      m => m.sender === "User" && m.messageId === msg.messageId
                    );
                    return userMsg ? (
                      <div style={{
                        fontSize: "0.7rem",
                        color: "rgba(16,185,129,0.5)",
                        marginBottom: "4px",
                        fontStyle: "italic"
                      }}>
                        ↳ re: "{userMsg.text.slice(0, 50)}{userMsg.text.length > 50 ? "…" : ""}"
                      </div>
                    ) : null;
                  })()}
                  {msg.sender === "Clara" ? (
                    <div className="prose prose-invert prose-sm max-w-none leading-relaxed
                        prose-code:bg-black/40 prose-code:text-emerald-300 prose-code:px-1 prose-code:rounded
                        prose-pre:bg-black/60 prose-pre:border prose-pre:border-white/10 prose-pre:rounded-lg
                        prose-a:text-emerald-400 prose-strong:text-white prose-headings:text-white">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                  )}
                  
                </div>
              </div>
            ))
          )}

          {/* --- THE PHANTOM BUBBLE (Live Stream) --- */}
          {streamingContent && (
            <div className="flex justify-start animate-in fade-in duration-100">
              <div className="max-w-[80%] p-4 rounded-xl flex flex-col gap-2 bg-emerald-900/10 border border-emerald-500/20 text-emerald-100 shadow-[0_0_15px_rgba(16,185,129,0.1)]">
                <div className="prose prose-invert prose-sm max-w-none leading-relaxed
                    prose-code:bg-black/40 prose-code:text-emerald-300 prose-code:px-1 prose-code:rounded
                    prose-pre:bg-black/60 prose-pre:border prose-pre:border-white/10 prose-pre:rounded-lg
                    prose-a:text-emerald-400 prose-strong:text-white prose-headings:text-white">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingContent}</ReactMarkdown>
                </div>
                <span className="inline-block w-2 h-4 bg-emerald-400 animate-pulse"></span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* --- NEW INPUT CAPSULE --- */}
        <div className="absolute bottom-0 left-0 w-full p-6 z-40 bg-gradient-to-t from-[#0a0a0a] via-[#0a0a0a]/90 to-transparent">
          
          <div 
            className={`
              relative flex items-end gap-3 p-3 rounded-2xl border transition-all duration-500 ease-out
              ${status === "thinking" 
                ? "bg-emerald-900/10 border-emerald-500/50 shadow-[0_0_30px_-5px_rgba(16,185,129,0.2)] animate-pulse" 
                : isFocused 
                  ? "bg-black/80 border-emerald-500/30 shadow-[0_0_50px_-10px_rgba(16,185,129,0.1)]" 
                  : "bg-[#111]/50 border-white/5 shadow-none backdrop-blur-sm"
              }
            `}
          >
            {/* ATTACHMENT BUTTON */}
            <button 
              onClick={() => document.getElementById('file-upload').click()}
              className={`p-3 rounded-xl transition-colors ${selectedImage ? "text-emerald-400 bg-emerald-900/20" : "text-white/40 hover:text-white hover:bg-white/5"}`}
            >
              <Paperclip size={20} />
            </button>
            <input 
              type="file" 
              id="file-upload" 
              className="hidden" 
              accept="image/*"
              onChange={handleImageUpload}
            />

            {/* TEXT INPUT AREA */}
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              onPaste={(e) => {
                const items = e.clipboardData?.items;
                if (!items) return;
                for (const item of items) {
                  if (item.type.startsWith("image/")) {
                    e.preventDefault();
                    const file = item.getAsFile();
                    const reader = new FileReader();
                    reader.onload = (ev) => setSelectedImage(ev.target.result);
                    reader.readAsDataURL(file);
                    break;
                  }
                }
              }}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Message Clara... (Ctrl+V to paste image)"
              className="w-full bg-transparent text-gray-200 placeholder-gray-600 focus:outline-none focus:ring-0 resize-none py-3 max-h-32"
              rows={1}
              style={{ minHeight: '44px' }}
            />

            {/* SEND BUTTON */}
            <button 
              onClick={sendMessage}
              disabled={!input.trim() && !selectedImage}
              className={`p-3 rounded-xl transition-all duration-300 ${
                input.trim() || selectedImage 
                  ? "bg-emerald-600 text-white shadow-[0_0_15px_rgba(16,185,129,0.4)] hover:bg-emerald-500" 
                  : "bg-[#222] text-gray-600 cursor-not-allowed"
              }`}
            >
              <Send size={20} />
            </button>
          </div>
          
          {/* IMAGE PREVIEW THUMBNAIL */}
          {selectedImage && (
            <div className="absolute -top-16 left-6 flex items-center gap-2 bg-black/80 border border-emerald-500/30 rounded-xl px-2 py-2 shadow-lg">
              <img
                src={selectedImage}
                alt="Preview"
                className="h-10 w-10 object-cover rounded-lg border border-white/10 cursor-zoom-in"
                onClick={() => setViewImage(selectedImage)}
              />
              <span className="text-xs text-emerald-400 font-mono">Image ready</span>
              <button
                onClick={() => setSelectedImage(null)}
                className="ml-1 text-white/30 hover:text-white/80 transition-colors"
              >
                <X size={14} />
              </button>
            </div>
          )}
        </div>
      </main>

      {/* --- ZONE C: THE BRAIN --- */}
      <aside className={`${isBrainOpen ? "w-80" : "w-0"} transition-all duration-300 border-l border-[var(--border-subtle)] bg-[#0f0f0f] flex flex-col`}>
        <div className="p-4 border-b border-[var(--border-subtle)] flex items-center justify-between">
          <div className="flex items-center gap-2 text-purple-400">
            <Cpu size={18} className={status === 'thinking' ? 'animate-spin-slow' : ''} />
            <span className="font-bold text-sm">NEURAL STREAM</span>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-4 scroll-smooth">
           {thoughts.map((t, i) => {
             const isLast = i === thoughts.length - 1;
             return (
               <div 
                 key={i} 
                 className={`
                   transition-all duration-500 border-l-2 pl-4 py-1 relative
                   ${isLast 
                     ? "border-emerald-500 bg-emerald-900/10 shadow-[0_0_15px_rgba(16,185,129,0.1)] opacity-100" 
                     : "border-purple-500/20 opacity-50 hover:opacity-100 hover:border-purple-500/50"
                   }
                 `}
               >
                 <span className={`block mb-1 text-[10px] tracking-widest ${isLast ? "text-emerald-400" : "text-purple-400/70"}`}>
                   [{t.time}]
                 </span>
                 <span className={`leading-relaxed whitespace-pre-wrap ${isLast ? "text-emerald-100" : "text-gray-400"}`}>
                   {t.text}
                 </span>
                 {isLast && (
                    <span className="absolute -left-[5px] top-0 w-2 h-2 bg-emerald-400 rounded-full animate-ping" />
                 )}
               </div>
             );
           })}
           <div ref={brainEndRef} className="h-4" />
        </div>
      </aside>
      {/* --- QUOTE POPUP --- */}
      {quotePopup && (
        <button
          className="fixed z-50 text-xs font-mono px-3 py-1.5 rounded-full shadow-lg
                     -translate-x-1/2 -translate-y-full
                     bg-emerald-600/90 text-white border border-emerald-400/30
                     hover:bg-emerald-500 hover:shadow-[0_0_12px_rgba(16,185,129,0.4)]
                     transition-all duration-150 animate-in fade-in zoom-in-95"
          style={{ left: quotePopup.x, top: quotePopup.y }}
          onMouseDown={(e) => {
            e.preventDefault();
            const label = quotePopup.sender === "Clara" ? "[Clara]" : "[Alkama]";
            setInput(prev => `> ${label}: ${quotePopup.text}\n\n${prev}`);
            setQuotePopup(null);
            window.getSelection()?.removeAllRanges();
            textareaRef.current?.focus();
          }}
        >
          QUOTE
        </button>
      )}

      {/* --- LIGHTBOX MODAL --- */}
      {viewImage && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setViewImage(null)} // Click anywhere to close
        >
          <img 
            src={viewImage} 
            alt="Full Screen" 
            className="max-w-full max-h-full rounded-lg shadow-2xl border border-white/10" 
          />
          <button className="absolute top-6 right-6 text-white/50 hover:text-white transition-colors">
            <X size={32} />
          </button>
        </div>
      )}

    </div>
  );
}