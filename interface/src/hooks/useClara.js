import { useState, useEffect, useRef, useCallback } from 'react';

export default function useClara() {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('clara_messages');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [thoughts, setThoughts] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("disconnected");
  const [selectedImage, setSelectedImage] = useState(null);
  const [streamingContent, setStreamingContent] = useState("");

  const socketRef = useRef(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef(null);
  const isMountedRef = useRef(true);

  // Persist messages to localStorage on every update
  useEffect(() => {
    try {
      localStorage.setItem('clara_messages', JSON.stringify(messages));
    } catch {}
  }, [messages]);

  const addMessage = (sender, text, image = null) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages(prev => [...prev, { sender, text, image, time: timestamp }]);
  };

  const addThought = (source, text) => {
    setThoughts(prev => [...prev, { source, text, time: new Date().toLocaleTimeString() }]);
  };

  const clearHistory = () => {
    setMessages([]);
    localStorage.removeItem('clara_messages');
  };

  const connect = useCallback(() => {
    if (!isMountedRef.current) return;

    const ws = new WebSocket("ws://localhost:8001/ws");
    socketRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) return;
      retryCountRef.current = 0;
      setStatus("connected");
      addThought("System", "Neural Link Established.");
    };

    ws.onmessage = (event) => {
      if (!isMountedRef.current) return;
      const data = JSON.parse(event.data);

      if (data.type === "thought") {
        setThoughts(prev => {
          const newThoughts = [...prev];
          const last = newThoughts[newThoughts.length - 1];
          if (last && last.source === "Clara" && last.turn_id === data.turn_id) {
            last.text = data.content;
          } else {
            newThoughts.push({
              source: "Clara",
              text: data.content,
              time: new Date().toLocaleTimeString(),
              turn_id: data.turn_id
            });
          }
          return newThoughts;
        });
        setStatus("thinking");
      }

      if (data.type === "status") {
        addThought("System", data.content);
        setStatus("thinking");
      }

      if (data.type === "stream") {
        setStreamingContent(prev => prev + data.content);
        setStatus("typing");
      }

      if (data.type === "final_answer") {
        addMessage("Clara", data.content);
        setStreamingContent("");
        setStatus("idle");
      }
    };

    ws.onerror = () => {
      // onclose will fire right after — handle retry there
    };

    ws.onclose = () => {
      if (!isMountedRef.current) return;
      setStatus("disconnected");

      // Exponential backoff: 1s, 2s, 4s, 8s, 16s, cap at 30s
      const delay = Math.min(1000 * Math.pow(2, retryCountRef.current), 30000);
      retryCountRef.current += 1;

      addThought("System", `Connection lost. Retrying in ${Math.round(delay / 1000)}s... (attempt ${retryCountRef.current})`);

      retryTimerRef.current = setTimeout(() => {
        if (isMountedRef.current) connect();
      }, delay);
    };
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    connect();

    return () => {
      isMountedRef.current = false;
      clearTimeout(retryTimerRef.current);
      socketRef.current?.close();
    };
  }, [connect]);

  const sendMessage = () => {
    if (!input.trim() && !selectedImage) return;

    addMessage("User", input, selectedImage);

    const payload = JSON.stringify({ text: input, image: selectedImage });

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
      reader.onloadend = () => setSelectedImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  return {
    messages, thoughts, input, setInput, sendMessage, status,
    selectedImage, setSelectedImage, handleImageUpload,
    streamingContent, clearHistory
  };
}
