import { useState, useEffect, useRef } from 'react';

export default function useClara() {
  const [messages, setMessages] = useState([]);      // The Main Chat (Center)
  const [thoughts, setThoughts] = useState([]);      // The Neural Stream (Right)
  const [input, setInput] = useState("");            // The User's Text box
  const [status, setStatus] = useState("disconnected"); 
  const [selectedImage, setSelectedImage] = useState(null);
  
  // 1. THE HOLDING CELL: Catches the live stream
  const [streamingContent, setStreamingContent] = useState(""); 
  
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = new WebSocket("ws://localhost:8001/ws");

    socketRef.current.onopen = () => {
      console.log("✅ WebSocket Connected");
      setStatus("connected");
      addThought("System", "Neural Link Established.");
    };

    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // STATE A: Internal Monologue -> Right Panel (THE STACK FIX)
      if (data.type === "thought") {
        setThoughts(prev => {
          const newThoughts = [...prev];
          const lastThought = newThoughts.length > 0 ? newThoughts[newThoughts.length - 1] : null;

          // If it's the same turn ID from Clara, overwrite the text to create the stream effect
          if (lastThought && lastThought.source === "Clara" && lastThought.turn_id === data.turn_id) {
            lastThought.text = data.content;
          } else {
            // New turn ID detected: push a brand new block to the stack
            newThoughts.push({ 
              source: "Clara", 
              text: data.content, 
              time: new Date().toLocaleTimeString(),
              turn_id: data.turn_id // Save the ID for the next comparison
            });
          }
          return newThoughts;
        });
        setStatus("thinking"); 
      } 
      
      // (Optional) Keep status updates separate so they render as distinct system logs
      if (data.type === "status") {
        addThought("System", data.content);
        setStatus("thinking");
      }
      
      // STATE B: Live Streaming -> Temporary Buffer
      if (data.type === "stream") {
        setStreamingContent(prev => prev + data.content); 
        setStatus("typing"); 
      }
      
      // STATE C: The Final Lock-in -> Main Chat
      if (data.type === "final_answer") {
        addMessage("Clara", data.content); 
        setStreamingContent("");           
        setStatus("idle");     
      }
    };

    return () => socketRef.current?.close();
  }, []);

  const addMessage = (sender, text, image = null) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages(prev => [...prev, { sender, text, image, time: timestamp }]);
  };

  const addThought = (source, text) => {
    setThoughts(prev => [...prev, { source, text, time: new Date().toLocaleTimeString() }]);
  };

  const sendMessage = () => {
    if (!input.trim() && !selectedImage) return;
    
    addMessage("User", input, selectedImage);
    
    const payload = JSON.stringify({
      text: input,
      image: selectedImage
    });
    
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(payload);
    }
    
    setInput("");
    setSelectedImage(null);
    setStatus("thinking");
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result); 
      };
      reader.readAsDataURL(file);
    }
  };

  // 2. EXPORT THE BUFFER: Expose it to the UI
  return { 
    messages, thoughts, input, setInput, sendMessage, status,
    selectedImage, setSelectedImage, handleImageUpload,
    streamingContent 
  };
}