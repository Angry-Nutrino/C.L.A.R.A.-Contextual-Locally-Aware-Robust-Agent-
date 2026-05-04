import { useState, useEffect, useRef, useCallback } from 'react';

export default function useClara() {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('clara_messages');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [thoughts, setThoughts] = useState([]);
  const [tasks, setTasks] = useState([]);   // live task board
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("disconnected");
  const [selectedImage, setSelectedImage] = useState(null);
  const [streamingContent, setStreamingContent] = useState("");
  const [lastTokenUsage, setLastTokenUsage] = useState(null);
  const [voiceActive, setVoiceActive] = useState(false);
  const [claraIsSpeaking, setClaraIsSpeaking] = useState(false);
  const voiceActiveRef = useRef(false);
  const claraIsSpeakingRef = useRef(false);

  const socketRef = useRef(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef(null);
  const isMountedRef = useRef(true);
  const pendingRef = useRef(new Map());
  // key: message_id → value: true (presence set for in-flight messages)

  // Persist messages to localStorage on every update
  useEffect(() => {
    try {
      localStorage.setItem('clara_messages', JSON.stringify(messages));
    } catch {}
  }, [messages]);

  const addMessage = (sender, text, image = null, messageId = null) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (sender === "User" && messageId) {
      pendingRef.current.set(messageId, true);
    }
    setMessages(prev => [...prev, { sender, text, image, time: timestamp, messageId }]);
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

      if (data.type === "task_event") {
        const { task_id, goal, state, priority, source } = data;
        setTasks(prev => {
          const existing = prev.findIndex(t => t.task_id === task_id);
          const entry = { task_id, goal, state, priority, source };
          if (existing >= 0) {
            const updated = [...prev];
            updated[existing] = entry;
            // prune completed/failed after 2s
            if (state === "completed" || state === "failed") {
              setTimeout(() => {
                setTasks(p => p.filter(t => t.task_id !== task_id));
              }, 2000);
            }
            return updated;
          }
          return [...prev, entry];
        });
        return;
      }

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

      if (data.type === "token_usage") {
        setLastTokenUsage(data.extra);
      }

      if (data.type === "user_transcript") {
        addMessage("User", data.content, null, data.message_id);
        pendingRef.current.set(data.message_id, true);
        setStatus("thinking");
      }

      if (data.type === "speaking_start") {
        claraIsSpeakingRef.current = true;
        setClaraIsSpeaking(true);
      }

      if (data.type === "speaking_stop") {
        claraIsSpeakingRef.current = false;
        setClaraIsSpeaking(false);
      }

      if (data.type === "final_answer") {
        const msgId = data.message_id || null;
        addMessage("Clara", data.content, null, msgId);
        if (msgId) pendingRef.current.delete(msgId);
        setStreamingContent("");
        if (pendingRef.current.size === 0) {
          setStatus("idle");
        }
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

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code !== "F4") return;
      e.preventDefault();
      if (claraIsSpeakingRef.current) {
        socketRef.current?.send(JSON.stringify({ type: "voice_interrupt" }));
        return;
      }
      if (!voiceActiveRef.current) {
        voiceActiveRef.current = true;
        setVoiceActive(true);
        socketRef.current?.send(JSON.stringify({ type: "voice_start" }));
      }
    };
    const handleKeyUp = (e) => {
      if (e.code !== "F4" || !voiceActiveRef.current) return;
      voiceActiveRef.current = false;
      setVoiceActive(false);
      const messageId = crypto.randomUUID();
      socketRef.current?.send(JSON.stringify({ type: "voice_stop", message_id: messageId }));
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  const sendMessage = () => {
    if (!input.trim() && !selectedImage) return;

    const messageId = crypto.randomUUID();
    addMessage("User", input, selectedImage, messageId);

    const payload = JSON.stringify({
      text: input,
      image: selectedImage,
      message_id: messageId,
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
      reader.onloadend = () => setSelectedImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  return {
    messages, thoughts, tasks, input, setInput, sendMessage, status,
    selectedImage, setSelectedImage, handleImageUpload,
    streamingContent, clearHistory, lastTokenUsage,
    voiceActive, claraIsSpeaking,
  };
}
