import React, { useState, useEffect } from 'react';

const Typewriter = ({ text, speed = 10, onComplete }) => {
  const [displayedText, setDisplayedText] = useState("");
  
  useEffect(() => {
    setDisplayedText(""); // Reset when text changes
    let i = 0;
    
    const intervalId = setInterval(() => {
      // 🛡️ SAFE MODE: Slice the string instead of appending
      // This guarantees we never skip a letter, even if React lags.
      setDisplayedText(text.slice(0, i + 1));
      
      i++;
      
      if (i === text.length) {
        clearInterval(intervalId);
        if (onComplete) onComplete();
      }
    }, speed);

    return () => clearInterval(intervalId);
  }, [text, speed, onComplete]);

  return (
    <span>
      {displayedText}
      <span className="cursor-blink"></span>
    </span>
  );
};

export default Typewriter;