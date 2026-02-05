import React, { useState, useRef, useEffect } from 'react';
import { useClara } from './hooks/useClara';
import { Terminal, Cpu, Send, Image as ImageIcon, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function App() {
  const { isConnected, logs, sendMessage } = useClara();
  
  // --- STATE MANAGEMENT ---
  const [input, setInput] = useState("");
  const [selectedImage, setSelectedImage] = useState(null); // Stores the Base64 image
  
  // --- REFS ---
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null); // The invisible file picker

  // Auto-scroll to bottom when logs change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // --- HANDLERS ---
  
  // 1. When you click the Icon -> Click the hidden input
  const handleImageClick = () => {
    fileInputRef.current.click();
  };

  // 2. When a file is chosen -> Convert to Base64
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // 3. Clear the image (the little 'X' button)
  const clearImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // 4. Send Message + Image
  const handleSend = () => {
    if (!input.trim() && !selectedImage) return;

    // Create the JSON payload
    const payload = JSON.stringify({
      text: input,
      image: selectedImage 
    });

    sendMessage(payload);
    
    // Reset UI
    setInput("");
    clearImage();
  };

  return (
    <div className="h-screen w-screen bg-[#050505] text-green-500 font-mono p-4 flex flex-col items-center scanline relative overflow-hidden">
      
      {/* BACKGROUND GRID */}
      <div className="absolute inset-0 grid grid-cols-[repeat(20,minmax(0,1fr))] opacity-10 pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="border-r border-green-900 h-full"></div>
        ))}
      </div>

      {/* HEADER */}
      <div className="z-50 w-full max-w-4xl flex justify-between items-center border-b border-green-800 pb-4 mb-4 bg-[#050505]/90 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-full border ${isConnected ? "border-green-500 shadow-[0_0_10px_#00ff41]" : "border-red-500"}`}>
            <Cpu size={24} className={isConnected ? "text-green-400 animate-pulse" : "text-red-500"} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-[0.2em] text-white">CLARA <span className="text-xs text-green-600">v2.0</span></h1>
            <p className="text-[10px] text-green-700 uppercase">Neural Link: {isConnected ? "STABLE" : "OFFLINE"}</p>
          </div>
        </div>
        <div className="text-right hidden sm:block">
          <p className="text-xs text-green-800">MEM: 24GB // GPU: ACTIVE</p>
          <p className="text-xs text-green-800">LATENCY: 12ms</p>
        </div>
      </div>

      {/* CHAT AREA */}
      <div className="z-10 flex-1 w-full max-w-4xl overflow-y-auto mb-4 pr-2 space-y-4">
        {logs.map((log, index) => {
          if (!log) return null;

          const isUser = log.startsWith(">> User:");
          const isSystem = log.startsWith("System:");
          
          // Clean the prefix
          let content = log;
          if (log.startsWith(">> User:")) content = log.replace(">> User:", "");
          if (log.startsWith(">> Clara:")) content = log.replace(">> Clara:", "");
          if (log.startsWith("System:")) content = log.replace("System:", "");
          
          // Hide JSON structure from chat
          if (content.includes('{"text":')) {
             try {
                const parsed = JSON.parse(content);
                content = parsed.text + (parsed.image ? " [IMAGE UPLOADED]" : "");
             } catch (e) { content = "Invalid Data"; }
          }

          if (isSystem) return <div key={index} className="text-xs text-gray-500 text-center border-b border-gray-900 leading-[0.1em] my-4"><span className="bg-[#050505] px-2">{content}</span></div>;

          return (
            <div key={index} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[80%] p-4 rounded-lg border ${isUser ? "border-green-800 bg-green-900/10" : "border-gray-800 bg-gray-900/20"}`}>
                <div className="flex items-center gap-2 mb-2 border-b border-white/5 pb-1">
                  {isUser ? <Terminal size={14} /> : <Cpu size={14} />}
                  <span className="text-xs font-bold opacity-50">{isUser ? "COMMAND" : "RESPONSE"}</span>
                </div>
                <div className="prose prose-invert prose-p:my-1 prose-pre:bg-black/50 prose-pre:border prose-pre:border-green-900/50 text-sm opacity-90">
                   <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                </div>
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* IMAGE PREVIEW AREA (The "Badge" that appears when you pick a file) */}
      {selectedImage && (
        <div className="z-50 w-full max-w-4xl mb-2 flex justify-start">
            <div className="relative group border border-green-600 rounded overflow-hidden">
                <img src={selectedImage} alt="Upload Preview" className="h-20 w-auto opacity-80" />
                <button onClick={clearImage} className="absolute top-0 right-0 bg-red-900/80 text-white p-1 hover:bg-red-600">
                    <X size={12} />
                </button>
            </div>
        </div>
      )}

      {/* INPUT AREA */}
      <div className="z-50 w-full max-w-4xl bg-black/80 border border-green-800 rounded-lg p-2 flex items-center gap-2 shadow-[0_0_20px_rgba(0,255,0,0.1)]">
        
        {/* HIDDEN WORKER */}
        <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden" 
            accept="image/*"
        />

        {/* IMAGE BUTTON */}
        <button 
            onClick={handleImageClick}
            className={`p-2 rounded transition-colors ${selectedImage ? "text-green-400 bg-green-900/40" : "text-green-600 hover:bg-green-900/30"}`}
            title="Upload Image"
        >
          <ImageIcon size={20} />
        </button>

        {/* TEXT INPUT */}
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Enter command..."
          className="flex-1 bg-transparent border-none outline-none text-green-100 placeholder-green-900 font-mono"
        />
        
        {/* SEND BUTTON */}
        <button 
          onClick={handleSend}
          className="p-2 bg-green-900/50 hover:bg-green-700 text-green-100 rounded transition-all"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}

export default App;