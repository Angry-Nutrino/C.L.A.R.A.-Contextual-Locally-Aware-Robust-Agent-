import React, { useState, useEffect, useRef } from "react";
// 1. UPDATED IMPORTS: Added Paperclip and X for the file upload UI
import { Terminal, Cpu, MessageSquare, Menu, Send, Paperclip, X } from "lucide-react";
import useClara from "./hooks/useClara";
import Typewriter from "./components/Typewriter";

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isBrainOpen, setIsBrainOpen] = useState(true);
  
  // 2. UPDATED HOOK: Getting the image tools from useClara
  const { 
    messages, thoughts, input, setInput, sendMessage, status,
    selectedImage, setSelectedImage, handleImageUpload 
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
      <aside className={`${isSidebarOpen ? "w-64" : "w-0"} transition-all duration-300 border-r border-[var(--border-subtle)] bg-[var(--bg-surface)] flex flex-col`}>
        <div className="p-4 border-b border-[var(--border-subtle)] flex items-center gap-2">
          <Terminal size={20} className="text-emerald-500" />
          <span className="font-bold tracking-wider text-emerald-500">AGENT_ZERO</span>
        </div>
        <div className="p-4 text-xs text-gray-500 mt-auto">
          Status: <span className={status === "connected" || status === "thinking" ? "text-green-500" : "text-red-500"}>{status.toUpperCase()}</span>
        </div>
      </aside>

      {/* --- ZONE B: MAIN CHAT --- */}
      <main className="flex-1 flex flex-col relative min-w-0">
        
        {/* HEADER */}
        <header className="h-14 border-b border-[var(--border-subtle)] flex items-center justify-between px-4 bg-[var(--bg-depth)]/80 backdrop-blur-md sticky top-0 z-10">
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-white/5 rounded"><Menu size={20} /></button>
          
          <span className={`text-xs font-bold px-3 py-1 rounded-full transition-all 
            ${status === 'thinking' ? 'bg-emerald-500/20 text-emerald-400 animate-pulse border border-emerald-500/50' : 'opacity-30'}`}>
            {status === 'thinking' ? 'PROCESSING...' : 'IDLE'}
          </span>
          
          <button onClick={() => setIsBrainOpen(!isBrainOpen)} className="p-2 hover:bg-white/5 rounded text-purple-400"><Cpu size={20} /></button>
        </header>

        {/* MESSAGES AREA */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth">
          {messages.length === 0 ? (
            <div className="text-center mt-20 opacity-30">
              <h1 className="text-4xl font-black mb-2">INITIALIZED</h1>
              <p>Waiting for input...</p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.sender === "User" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] p-4 rounded-xl 
                  ${msg.sender === "User" 
                    ? "bg-[#222] border border-[#333]" 
                    : "bg-emerald-900/10 border border-emerald-500/20 text-emerald-100 shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                  }`}>
                  
                  {msg.sender === "Clara" && i === messages.length - 1 ? (
                    <Typewriter text={msg.text} speed={15} /> 
                  ) : (
                    <p className="whitespace-pre-wrap">{msg.text}</p>
                  )}
                  
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        {/* --- NEW INPUT CAPSULE --- */}
        <div className="p-4 w-full max-w-3xl mx-auto">
          <div className={`
              relative transition-all duration-500 rounded-2xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] 
              shadow-lg shadow-black/50 p-2 flex flex-col gap-2
              ${status === 'thinking' ? 'animate-breathe' : ''}
          `}>
             
             {/* THUMBNAIL PREVIEW (Only shows if image is selected) */}
             {selectedImage && (
               <div className="relative w-16 h-16 ml-2 mt-2 group">
                 <img 
                   src={selectedImage} 
                   alt="Preview" 
                   className="w-full h-full object-cover rounded-lg border border-[var(--border-subtle)]" 
                 />
                 <button 
                   onClick={() => setSelectedImage(null)}
                   className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5 hover:bg-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                 >
                   <X size={12} />
                 </button>
               </div>
             )}

             <div className="flex items-end gap-2">
               {/* UPLOAD BUTTON */}
               <label className="p-3 text-gray-400 hover:text-emerald-400 cursor-pointer transition-colors rounded-xl hover:bg-white/5">
                  <Paperclip size={20} />
                  <input 
                    type="file" 
                    accept="image/*" 
                    className="hidden" 
                    onChange={handleImageUpload}
                  />
               </label>

               {/* AUTO-EXPANDING TEXTAREA */}
               <textarea 
                 ref={textareaRef}
                 value={input}
                 onChange={(e) => setInput(e.target.value)}
                 onKeyDown={handleKeyDown}
                 placeholder={status === 'thinking' ? "Agent is thinking..." : "Message Clara..."}
                 disabled={status === 'thinking'}
                 rows={1}
                 className="w-full bg-transparent border-none focus:ring-0 resize-none py-3 max-h-48 overflow-y-auto text-gray-200 placeholder-gray-500/50"
               />
               
               {/* SEND BUTTON */}
               <button 
                 onClick={sendMessage} 
                 disabled={!input.trim() && !selectedImage}
                 className={`p-3 rounded-xl transition-all ${
                    input.trim() || selectedImage 
                      ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_0_10px_rgba(16,185,129,0.4)]" 
                      : "bg-white/5 text-gray-600 cursor-not-allowed"
                 }`}
               >
                  <Send size={18} />
               </button>
             </div>
          </div>
          <div className="text-center mt-2 text-[10px] text-gray-600 font-mono">
             Clara v2.0 | Press Shift+Enter for new line
          </div>
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
                 <span className={`leading-relaxed ${isLast ? "text-emerald-100" : "text-gray-400"}`}>
                   {isLast ? (
                     <Typewriter text={t.text} speed={5} /> 
                   ) : (
                     t.text
                   )}
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

    </div>
  );
}