import { useState, useEffect, useRef, useCallback } from 'react';

export default function useClara() {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('clara_messages');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [queryCards, setQueryCards] = useState([]);
  const [systemLogs, setSystemLogs] = useState([]);
  const [tasks, setTasks]     = useState([]);
  const [input, setInput]     = useState("");
  const [status, setStatus]   = useState("disconnected");
  const [selectedImage, setSelectedImage] = useState(null);
  const [streamingContent, setStreamingContent] = useState("");
  const [lastTokenUsage, setLastTokenUsage] = useState(null);
  const [voiceActive, setVoiceActive]       = useState(false);
  const [claraIsSpeaking, setClaraIsSpeaking] = useState(false);
  const voiceActiveRef      = useRef(false);
  const claraIsSpeakingRef  = useRef(false);
  const socketRef           = useRef(null);
  const retryCountRef       = useRef(0);
  const retryTimerRef       = useRef(null);
  const isMountedRef        = useRef(true);
  const pendingRef          = useRef(new Map());
  const taskIdToMsgRef      = useRef(new Map()); // task_id → message_id

  useEffect(() => {
    try { localStorage.setItem('clara_messages', JSON.stringify(messages)); } catch {}
  }, [messages]);

  const addMessage = (sender, text, image = null, messageId = null) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (sender === "User" && messageId) pendingRef.current.set(messageId, true);
    setMessages(prev => [...prev, { sender, text, image, time: timestamp, messageId }]);
  };

  const addSystemLog = (text) => {
    setSystemLogs(prev => [...prev.slice(-4), {
      text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }]);
  };

  const clearHistory = () => {
    setMessages([]);
    localStorage.removeItem('clara_messages');
  };

  // Create a fresh card object (module-level pure function)
  const makeCard = (messageId, query) => ({
    messageId,
    taskId: null,
    query,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    thoughts: [],
    isComplete: false,
    isCancelled: false,
    isFailed: false,
    isExpanded: true,
    manuallyExpanded: false,
  });

  // Collapse all non-pinned cards and prepend a new one
  const openCard = (card) => {
    setQueryCards(prev => [
      card,
      ...prev.map(c => c.manuallyExpanded ? c : { ...c, isExpanded: false }),
    ]);
  };

  const connect = useCallback(() => {
    if (!isMountedRef.current) return;
    const ws = new WebSocket("ws://localhost:8001/ws");
    socketRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) return;
      retryCountRef.current = 0;
      setStatus("connected");
      addSystemLog("Neural Link Established.");
    };

    ws.onmessage = (event) => {
      if (!isMountedRef.current) return;
      const data = JSON.parse(event.data);
      const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      // ── task_event ──────────────────────────────────────────────────────────
      if (data.type === "task_event") {
        const { task_id, goal, state, priority, source, message_id } = data;

        if (message_id && task_id) {
          taskIdToMsgRef.current.set(task_id, message_id);
          setQueryCards(prev => prev.map(c =>
            c.messageId === message_id && !c.taskId ? { ...c, taskId: task_id } : c
          ));
        }

        if (state === "failed") {
          const mid = message_id || taskIdToMsgRef.current.get(task_id);
          if (mid) {
            setQueryCards(prev => prev.map(c => c.messageId === mid ? { ...c, isFailed: true } : c));
            setTimeout(() => {
              setQueryCards(prev => prev.map(c =>
                c.messageId === mid && !c.manuallyExpanded ? { ...c, isExpanded: false } : c
              ));
            }, 2000);
          }
        }

        setTasks(prev => {
          const existing = prev.findIndex(t => t.task_id === task_id);
          const entry = { task_id, goal, state, priority, source };
          if (existing >= 0) {
            const updated = [...prev];
            updated[existing] = entry;
            if (state === "completed" || state === "failed") {
              setTimeout(() => setTasks(p => p.filter(t => t.task_id !== task_id)), 2000);
            }
            return updated;
          }
          return [...prev, entry];
        });
        return;
      }

      // ── task_cancelled ──────────────────────────────────────────────────────
      if (data.type === "task_cancelled") {
        if (data.success) {
          setTasks(p => p.filter(t => t.task_id !== data.task_id));
          const mid = taskIdToMsgRef.current.get(data.task_id);
          if (mid) {
            setQueryCards(prev => prev.map(c => c.messageId === mid ? { ...c, isCancelled: true } : c));
            setTimeout(() => {
              setQueryCards(prev => prev.map(c =>
                c.messageId === mid && !c.manuallyExpanded ? { ...c, isExpanded: false } : c
              ));
            }, 2000);
          }
        }
        return;
      }

      // ── thought ─────────────────────────────────────────────────────────────
      if (data.type === "thought") {
        if (data.message_id) {
          setQueryCards(prev => {
            const idx = prev.findIndex(c => c.messageId === data.message_id);
            if (idx < 0) return prev;
            const updated = [...prev];
            const card = updated[idx];
            const last = card.thoughts[card.thoughts.length - 1];
            let newThoughts;
            if (last && last.source === "Clara" && last.turn_id === data.turn_id) {
              newThoughts = [...card.thoughts.slice(0, -1), { ...last, text: data.content }];
            } else {
              newThoughts = [...card.thoughts, { source: "Clara", text: data.content, time: ts, turn_id: data.turn_id }];
            }
            updated[idx] = { ...card, thoughts: newThoughts };
            return updated;
          });
        }
        setStatus("thinking");
        return;
      }

      // ── status ──────────────────────────────────────────────────────────────
      if (data.type === "status") {
        if (data.message_id) {
          setQueryCards(prev => {
            const idx = prev.findIndex(c => c.messageId === data.message_id);
            if (idx < 0) return prev;
            const updated = [...prev];
            updated[idx] = {
              ...updated[idx],
              thoughts: [...updated[idx].thoughts, { source: "System", text: data.content, time: ts }],
            };
            return updated;
          });
        }
        setStatus("thinking");
        return;
      }

      // ── stream ───────────────────────────────────────────────────────────────
      if (data.type === "stream") {
        setStreamingContent(prev => prev + data.content);
        setStatus("typing");
        return;
      }

      // ── token_usage ──────────────────────────────────────────────────────────
      if (data.type === "token_usage") {
        setLastTokenUsage(data.extra);
        return;
      }

      // ── user_transcript (voice) ───────────────────────────────────────────────
      if (data.type === "user_transcript") {
        addMessage("User", data.content, null, data.message_id);
        pendingRef.current.set(data.message_id, true);
        setQueryCards(prev => [
          makeCard(data.message_id, data.content),
          ...prev.map(c => c.manuallyExpanded ? c : { ...c, isExpanded: false }),
        ]);
        setStatus("thinking");
        return;
      }

      // ── speaking ─────────────────────────────────────────────────────────────
      if (data.type === "speaking_start") {
        claraIsSpeakingRef.current = true;
        setClaraIsSpeaking(true);
        return;
      }
      if (data.type === "speaking_stop") {
        claraIsSpeakingRef.current = false;
        setClaraIsSpeaking(false);
        return;
      }

      // ── final_answer ──────────────────────────────────────────────────────────
      if (data.type === "final_answer") {
        const msgId = data.message_id || null;
        addMessage("Clara", data.content, null, msgId);
        if (msgId) {
          pendingRef.current.delete(msgId);
          setQueryCards(prev => prev.map(c => c.messageId === msgId ? { ...c, isComplete: true } : c));
          setTimeout(() => {
            setQueryCards(prev => prev.map(c =>
              c.messageId === msgId && !c.manuallyExpanded ? { ...c, isExpanded: false } : c
            ));
          }, 1500);
        }
        setStreamingContent("");
        if (pendingRef.current.size === 0) setStatus("idle");
        return;
      }
    };

    ws.onerror = () => {};

    ws.onclose = () => {
      if (!isMountedRef.current) return;
      setStatus("disconnected");
      const delay = Math.min(1000 * Math.pow(2, retryCountRef.current), 30000);
      retryCountRef.current += 1;
      addSystemLog(`Connection lost. Retrying in ${Math.round(delay / 1000)}s… (attempt ${retryCountRef.current})`);
      retryTimerRef.current = setTimeout(() => { if (isMountedRef.current) connect(); }, delay);
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

  // Push-to-talk
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

  const cancelTask = useCallback((taskId) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "cancel_task", task_id: taskId }));
    }
  }, []);

  const toggleCard = useCallback((messageId) => {
    setQueryCards(prev => prev.map(c => {
      if (c.messageId !== messageId) return c;
      const newExpanded = !c.isExpanded;
      return { ...c, isExpanded: newExpanded, manuallyExpanded: newExpanded };
    }));
  }, []);

  const sendMessage = () => {
    if (!input.trim() && !selectedImage) return;
    const messageId = crypto.randomUUID();
    addMessage("User", input, selectedImage, messageId);
    openCard(makeCard(messageId, input));
    socketRef.current?.readyState === WebSocket.OPEN &&
      socketRef.current.send(JSON.stringify({ text: input, image: selectedImage, message_id: messageId }));
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
    messages, queryCards, systemLogs, tasks,
    input, setInput, sendMessage, cancelTask, toggleCard, status,
    selectedImage, setSelectedImage, handleImageUpload,
    streamingContent, clearHistory, lastTokenUsage,
    voiceActive, claraIsSpeaking,
  };
}
