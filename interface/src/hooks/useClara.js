// interface/src/hooks/useClara.js
import { useState, useEffect, useRef } from 'react';

export function useClara() {
  const [isConnected, setIsConnected] = useState(false);
  const [logs, setLogs] = useState([]); // Stores "Thinking..." messages
  const [response, setResponse] = useState(""); // Stores the final answer
  
  // This ref holds the actual socket connection
  const socketRef = useRef(null);

  useEffect(() => {
    // 1. Dial the Phone (Connect to Python)
    // Note: We use port 8001 because that's where you moved api.py
    const ws = new WebSocket('ws://localhost:8001/ws');
    socketRef.current = ws;

    // 2. When Connected
    ws.onopen = () => {
      console.log("✅ React: Connected to Brain");
      setIsConnected(true);
      addLog("System: Neural Link Established.");
    };

    // 3. When a Message Arrives (From Python)
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'log') {
        // It's a "Thinking" message
        addLog(data.content);
      } 
      else if (data.type === 'response') {
        // It's the Final Answer
        setResponse(data.content);
        addLog(`>> Clara: ${data.content}`);
      }
    };

    // 4. When Disconnected
    ws.onclose = () => {
      console.log("❌ React: Disconnected");
      setIsConnected(false);
      addLog("System: Neural Link Lost.");
    };

    // Cleanup: Hang up the phone when we leave the page
    return () => {
      ws.close();
    };
  }, []);

  // Helper to add logs to our list
  const addLog = (message) => {
    setLogs((prev) => [...prev, message]);
  };

  // Function to Speak (Send data to Python)
  const sendMessage = (text) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      addLog(`>> User: ${text}`);
      socketRef.current.send(text);
    } else {
      console.error("Cannot send: Disconnected");
    }
  };

  return { isConnected, logs, response, sendMessage };
}