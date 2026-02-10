import { useState, useEffect, useRef } from 'react';

export default function useClara() {
  const [messages, setMessages] = useState([]);      // The Main Chat (Center)
  const [thoughts, setThoughts] = useState([]);      // The Neural Stream (Right)
  const [input, setInput] = useState("");            // The User's Text box
  const [status, setStatus] = useState("disconnected"); // 'connected', 'thinking', 'idle'
  const [selectedImage, setSelectedImage] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    // 1. Connect to Python on Port 8001
    socketRef.current = new WebSocket("ws://localhost:8001/ws");

    socketRef.current.onopen = () => {
      console.log("✅ WebSocket Connected");
      setStatus("connected");
      addThought("System", "Neural Link Established.");
    };

    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // 2. ROUTING LOGIC
      // If it's a "Thought" -> Go to Right Panel
      if (data.type === "thought" || data.type === "status") {
        addThought("Clara", data.content);
        setStatus("thinking"); // Make the UI pulse
      } 
      
      // If it's a "Final Answer" -> Go to Center Chat
      if (data.type === "final_answer") {
        addMessage("Clara", data.content);
        setStatus("idle");     // Stop pulsing
      }
    };

    return () => socketRef.current?.close();
  }, []);

  // Helper to add to Chat (Center)
  const addMessage = (sender, text, image = null) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    // Store the image in the message object
    setMessages(prev => [...prev, { sender, text, image, time: timestamp }]);
  };

  // Helper to add to Brain (Right)
  const addThought = (source, text) => {
    setThoughts(prev => [...prev, { source, text, time: new Date().toLocaleTimeString() }]);
  };

  // The "Send" Action
  const sendMessage = () => {
    if (!input.trim() && !selectedImage) return;
    
    // Show user message (WITH IMAGE DATA now)
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
  // 4. NEW HELPER: Handle File Selection
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result); // Save as Data URL
      };
      reader.readAsDataURL(file);
    }
  };


  return { 
    messages, thoughts, input, setInput, sendMessage, status,
    selectedImage, setSelectedImage, handleImageUpload 
  };
}