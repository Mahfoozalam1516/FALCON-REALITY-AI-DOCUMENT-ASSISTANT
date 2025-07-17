import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Environment Variables
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const APP_TITLE = import.meta.env.VITE_APP_TITLE || 'AI Document Assistant';
const APP_SUBTITLE = import.meta.env.VITE_APP_SUBTITLE || 'FALCON REALITY ASSESSMENT';
const MAX_FILE_SIZE_MB = parseInt(import.meta.env.VITE_MAX_FILE_SIZE_MB || '10');
const MAX_FILES_PER_REQUEST = parseInt(import.meta.env.VITE_MAX_FILES_PER_REQUEST || '5');
const ALLOWED_FILE_TYPES = import.meta.env.VITE_ALLOWED_FILE_TYPES || '.pdf';
const TYPING_DELAY_MS = parseInt(import.meta.env.VITE_TYPING_DELAY_MS || '800');
const ANIMATION_DURATION_MS = parseInt(import.meta.env.VITE_ANIMATION_DURATION_MS || '600');
const ENABLE_ANIMATIONS = import.meta.env.VITE_ENABLE_ANIMATIONS !== 'false';
const DEBUG = import.meta.env.VITE_DEBUG === 'true';

// Debug logging function
const debugLog = (...args) => {
  if (DEBUG) {
    console.log('[RAG Chat Debug]:', ...args);
  }
};

// Configuration display for development
if (DEBUG) {
  console.log('RAG Chatbot Configuration:', {
    API_BASE_URL,
    APP_TITLE,
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_REQUEST,
    TYPING_DELAY_MS,
    ENABLE_ANIMATIONS
  });
}

// Markdown-like text formatter component
const FormattedText = ({ text }) => {
  const formatText = (text) => {
    if (!text) return '';

    // Split text into paragraphs
    let formatted = text.split('\n\n').map(paragraph => {
      if (!paragraph.trim()) return '';
      
      // Handle different formatting patterns
      let formattedParagraph = paragraph
        // Bold text (**text** or __text__)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.*?)__/g, '<strong>$1</strong>')
        
        // Italic text (*text* or _text_)
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/_(.*?)_/g, '<em>$1</em>')
        
        // Headers (# ## ###)
        .replace(/^### (.*$)/gm, '<h3 class="formatted-h3">$1</h3>')
        .replace(/^## (.*$)/gm, '<h2 class="formatted-h2">$1</h2>')
        .replace(/^# (.*$)/gm, '<h1 class="formatted-h1">$1</h1>')
        
        // Code blocks (```code```)
        .replace(/```(.*?)```/gs, '<pre class="code-block"><code>$1</code></pre>')
        
        // Inline code (`code`)
        .replace(/`(.*?)`/g, '<code class="inline-code">$1</code>')
        
        // Lists (- item or * item or 1. item)
        .replace(/^[\s]*[-*]\s+(.*)$/gm, '<li class="bullet-item">$1</li>')
        .replace(/^[\s]*\d+\.\s+(.*)$/gm, '<li class="numbered-item">$1</li>')
        
        // Links [text](url)
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="formatted-link">$1</a>')
        
        // Line breaks
        .replace(/\n/g, '<br/>');

      return formattedParagraph;
    });

    // Wrap consecutive list items in ul/ol
    let result = formatted.join('</p><p class="formatted-paragraph">');
    
    // Wrap list items in proper lists
    result = result
      .replace(/(<li class="bullet-item">.*?<\/li>)(?:\s*<\/p><p class="formatted-paragraph">)?(?=<li class="bullet-item">|$)/gs, 
        (match) => `<ul class="formatted-list">${match.replace(/<\/p><p class="formatted-paragraph">/g, '')}</ul>`)
      .replace(/(<li class="numbered-item">.*?<\/li>)(?:\s*<\/p><p class="formatted-paragraph">)?(?=<li class="numbered-item">|$)/gs, 
        (match) => `<ol class="formatted-list">${match.replace(/<\/p><p class="formatted-paragraph">/g, '')}</ol>`);
    
    // Clean up and wrap in paragraphs
    result = `<p class="formatted-paragraph">${result}</p>`;
    
    // Clean up empty paragraphs and fix formatting
    result = result
      .replace(/<p class="formatted-paragraph"><\/p>/g, '')
      .replace(/<p class="formatted-paragraph">(<h[1-6])/g, '$1')
      .replace(/(<\/h[1-6]>)<\/p>/g, '$1')
      .replace(/<p class="formatted-paragraph">(<ul|<ol)/g, '$1')
      .replace(/(<\/ul>|<\/ol>)<\/p>/g, '$1')
      .replace(/<p class="formatted-paragraph">(<pre)/g, '$1')
      .replace(/(<\/pre>)<\/p>/g, '$1');

    return result;
  };

  return (
    <div 
      className="formatted-content"
      dangerouslySetInnerHTML={{ __html: formatText(text) }}
    />
  );
};

// Typing animation component
const TypingAnimation = () => (
  <div className="typing-container">
    <div className="typing-indicator">
      <span></span>
      <span></span>
      <span></span>
    </div>
    <span className="typing-text">AI is thinking...</span>
  </div>
);

// Copy to clipboard component
const CopyButton = ({ text }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <button className="copy-button" onClick={handleCopy} title="Copy to clipboard">
      {copied ? '✅' : '📋'}
    </button>
  );
};

// Main App Component
function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const uploadRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-resize textarea
  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  };

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setIsConnected(true);
        debugLog('Backend connection established');
      }
    } catch (error) {
      setIsConnected(false);
      debugLog('Backend connection failed:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputMessage]);

  // File validation function
  const validateFiles = (files) => {
    const validFiles = [];
    const errors = [];
    
    if (files.length > MAX_FILES_PER_REQUEST) {
      errors.push(`Maximum ${MAX_FILES_PER_REQUEST} files allowed per upload.`);
      return { validFiles: files.slice(0, MAX_FILES_PER_REQUEST), errors };
    }
    
    files.forEach(file => {
      // Check file type
      if (!file.name.toLowerCase().endsWith(ALLOWED_FILE_TYPES.toLowerCase())) {
        errors.push(`${file.name}: Only ${ALLOWED_FILE_TYPES.toUpperCase()} files are supported.`);
        return;
      }
      
      // Check file size
      if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        errors.push(`${file.name}: File too large. Maximum size: ${MAX_FILE_SIZE_MB}MB`);
        return;
      }
      
      validFiles.push(file);
    });
    
    return { validFiles, errors };
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    
    debugLog('Files dropped:', files.map(f => f.name));
    
    const { validFiles, errors } = validateFiles(files);
    
    if (errors.length > 0) {
      setUploadStatus({
        type: 'warning',
        message: `Some files were rejected: ${errors.join(' ')}`
      });
    }

    if (validFiles.length > 0) {
      handleMultipleFiles(validFiles);
    }
  };

  // Handle multiple file upload
  const handleMultipleFiles = async (files) => {
    if (files.length === 0) return;

    debugLog('Processing files:', files.map(f => f.name));

    const { validFiles, errors } = validateFiles(Array.from(files));
    
    if (errors.length > 0) {
      setUploadStatus({
        type: 'error',
        message: `File validation errors: ${errors.join(' ')}`
      });
      
      if (validFiles.length === 0) return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    
    // Add all valid files to FormData
    for (let file of validFiles) {
      formData.append('files', file);
    }

    try {
      debugLog('Uploading to:', `${API_BASE_URL}/upload`);
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      debugLog('Upload response:', data);
      
      // Update uploaded files list
      const successfulFiles = data.files_processed.filter(f => f.status === 'success');
      const failedFiles = data.files_processed.filter(f => f.status === 'failed');
      
      setUploadedFiles(prev => [...prev, ...successfulFiles]);
      setTotalDocuments(prev => prev + successfulFiles.length);
      
      // Create status message
      let statusMessage = `Successfully processed ${successfulFiles.length} document(s) with ${data.total_chunks_processed} chunks.`;
      
      if (failedFiles.length > 0) {
        statusMessage += ` ${failedFiles.length} file(s) failed to process.`;
      }
      
      setUploadStatus({
        type: successfulFiles.length > 0 ? 'success' : 'error',
        message: statusMessage,
        details: data.files_processed
      });

      // Add system message to chat
      if (successfulFiles.length > 0) {
        const fileNames = successfulFiles.map(f => f.filename).join(', ');
        const systemMessage = {
          type: 'system',
          content: `📄 Successfully uploaded ${successfulFiles.length} document(s): ${fileNames}. You can now ask questions about these documents!`,
          timestamp: new Date().toISOString(),
          id: Date.now()
        };
        setMessages(prev => [...prev, systemMessage]);
      }

    } catch (error) {
      console.error('Error uploading files:', error);
      setUploadStatus({
        type: 'error',
        message: 'Failed to upload files. Please check your connection and try again.'
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Handle chat message
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !isConnected) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
      id: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          session_id: sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Update session ID if received
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      // Simulate typing delay for better UX with smooth reveal
      setTimeout(() => {
        const botMessage = {
          type: 'bot',
          content: data.response,
          timestamp: data.timestamp,
          sources_found: data.sources_found,
          id: Date.now() + 1,
          isNew: true
        };

        setMessages(prev => [...prev, botMessage]);
        setIsTyping(false);
        
        debugLog('Bot response:', data.response.substring(0, 100) + '...');
        
        // Remove the isNew flag after animation
        if (ENABLE_ANIMATIONS) {
          setTimeout(() => {
            setMessages(prev => prev.map(msg => 
              msg.id === botMessage.id ? { ...msg, isNew: false } : msg
            ));
          }, ANIMATION_DURATION_MS);
        }
      }, TYPING_DELAY_MS);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your request. Please check your connection and try again.',
        timestamp: new Date().toISOString(),
        id: Date.now() + 1
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsTyping(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;
    
    debugLog('Files selected:', files.map(f => f.name));
    
    const { validFiles, errors } = validateFiles(files);
    
    if (errors.length > 0) {
      setUploadStatus({
        type: 'error',
        message: `File validation errors: ${errors.join(' ')}`
      });
      
      if (validFiles.length === 0) return;
    }

    await handleMultipleFiles(validFiles);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    debugLog('Chat cleared');
  };

  const clearAllDocuments = async () => {
    try {
      debugLog('Clearing all documents...');
      
      const response = await fetch(`${API_BASE_URL}/clear`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        setUploadedFiles([]);
        setTotalDocuments(0);
        setMessages([]);
        setSessionId(null);
        setUploadStatus({
          type: 'success',
          message: 'All documents cleared successfully.'
        });
        
        debugLog('Documents cleared successfully');
      }
    } catch (error) {
      console.error('Error clearing documents:', error);
      setUploadStatus({
        type: 'error',
        message: 'Failed to clear documents.'
      });
    }
  };

  const suggestedQuestions = [
    "What are these documents about?",
    "Summarize the key points from all documents",
    "What are the main topics covered?",
    "Compare the different documents",
    "Extract important facts and figures",
    "What conclusions can be drawn?"
  ];

  return (
    <div className="app">
      {/* Animated Background */}
      <div className="background-animation">
        <div className="floating-shapes">
          <div className="shape shape-1"></div>
          <div className="shape shape-2"></div>
          <div className="shape shape-3"></div>
          <div className="shape shape-4"></div>
          <div className="shape shape-5"></div>
        </div>
      </div>

      {/* Connection Status */}
      {!isConnected && (
        <div className="connection-banner">
          <span>⚠️ Cannot connect to backend. Please ensure the server is running.</span>
          <button onClick={checkBackendConnection} className="retry-button">
            Retry Connection
          </button>
        </div>
      )}

      <header className="app-header">
        <div className="header-content">
          <div className="logo-container">
            <div className="logo-icon">🤖</div>
            <div className="logo-text">
              <h1>{APP_TITLE}</h1>
              <p className="subtitle">{APP_SUBTITLE}</p>
            </div>
          </div>
          <div className="header-stats">
            <div className="stat-item">
              <span className="stat-number">{messages.filter(m => m.type === 'user').length}</span>
              <span className="stat-label">Questions</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{totalDocuments}</span>
              <span className="stat-label">Documents</span>
            </div>
            <div className="stat-item">
              <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                <span className="status-dot"></span>
                <span className="status-text">{isConnected ? 'Connected' : 'Offline'}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="main-content">
        {/* Enhanced Upload Section */}
        <div className="upload-section">
          <div 
            className={`upload-container ${isDragging ? 'dragging' : ''} ${isUploading ? 'uploading' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            ref={uploadRef}
          >
            <div className="upload-area">
              <div className="upload-icon">
                {isUploading ? (
                  <div className="upload-spinner">
                    <div className="spinner"></div>
                  </div>
                ) : (
                  <div className="file-icon">📄</div>
                )}
              </div>
              <div className="upload-text">
                <h3>{isUploading ? 'Processing Documents...' : 'Upload Your PDF Documents'}</h3>
                <p>{isDragging ? 'Drop your files here' : `Drag & drop up to ${MAX_FILES_PER_REQUEST} files or click to browse`}</p>
                <p className="upload-limits">Max file size: {MAX_FILE_SIZE_MB}MB per file</p>
              </div>
              <div className="upload-buttons">
                <label htmlFor="file-upload" className="upload-button">
                  {isUploading ? (
                    <>
                      <span className="button-spinner"></span>
                      Processing...
                    </>
                  ) : (
                    <>
                      <span className="button-icon">📁</span>
                      Choose Files
                    </>
                  )}
                </label>
                {totalDocuments > 0 && (
                  <button onClick={clearAllDocuments} className="clear-docs-button">
                    <span className="button-icon">🗑️</span>
                    Clear All Documents
                  </button>
                )}
              </div>
              <input
                id="file-upload"
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept={ALLOWED_FILE_TYPES}
                multiple
                disabled={isUploading}
                style={{ display: 'none' }}
              />
            </div>
          </div>

          {uploadStatus && (
            <div className={`upload-status ${uploadStatus.type} fade-in`}>
              <div className="status-icon">
                {uploadStatus.type === 'success' ? '✅' : uploadStatus.type === 'warning' ? '⚠️' : '❌'}
              </div>
              <div className="status-content">
                <span>{uploadStatus.message}</span>
                {uploadStatus.details && (
                  <div className="upload-details">
                    {uploadStatus.details.map((file, index) => (
                      <div key={index} className={`file-status ${file.status}`}>
                        <span className="filename">{file.filename}</span>
                        <span className="file-info">
                          {file.status === 'success' 
                            ? `${file.chunks} chunks` 
                            : file.reason
                          }
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Uploaded Files List */}
          {uploadedFiles.length > 0 && (
            <div className="uploaded-files">
              <h4>📚 Uploaded Documents ({uploadedFiles.length})</h4>
              <div className="files-grid">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="file-card">
                    <div className="file-info">
                      <span className="file-name">{file.filename}</span>
                      <span className="file-stats">{file.chunks} chunks</span>
                    </div>
                    <div className="file-status success">✅</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Enhanced Chat Section */}
        <div className="chat-container">
          <div className="chat-header">
            <div className="chat-title">
              <div className="chat-icon">💬</div>
              <h3>Chat with your documents</h3>
              <div className={`status-indicator ${totalDocuments > 0 && isConnected ? 'online' : 'offline'}`}></div>
            </div>
            <div className="chat-actions">
              <button onClick={clearChat} className="clear-button" title="Clear Chat">
                <span className="button-icon">🗑️</span>
                Clear Chat
              </button>
            </div>
          </div>

          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-chat">
                <h3>Welcome to {APP_TITLE}!</h3>
                <p>Upload PDF documents and start having intelligent conversations about their content.</p>
                {totalDocuments === 0 ? (
                  <div className="getting-started">
                    <h4>Getting Started:</h4>
                    <ol>
                      <li>Upload one or more PDF documents above</li>
                      <li>Wait for processing to complete</li>
                      <li>Ask questions about your documents</li>
                    </ol>
                  </div>
                ) : (
                  <div className="example-questions">
                    <h4>Try asking:</h4>
                    <div className="question-chips">
                      {suggestedQuestions.map((question, index) => (
                        <span 
                          key={index}
                          className="chip" 
                          onClick={() => setInputMessage(question)}
                        >
                          "{question}"
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={message.id || index} className={`message ${message.type} message-enter ${message.isNew ? 'new-message' : ''}`}>
                  <div className="message-avatar">
                    {message.type === 'user' && <div className="avatar user-avatar">👤</div>}
                    {message.type === 'bot' && <div className="avatar bot-avatar">🤖</div>}
                    {message.type === 'system' && <div className="avatar system-avatar">📢</div>}
                  </div>
                  <div className="message-bubble">
                    <div className="message-content">
                      {message.type === 'bot' ? (
                        <>
                          <FormattedText text={message.content} />
                          <div className="message-actions">
                            <CopyButton text={message.content} />
                          </div>
                        </>
                      ) : (
                        <div className="user-message-text">{message.content}</div>
                      )}
                      {message.type === 'bot' && message.sources_found > 0 && (
                        <div className="sources-info">
                          <span className="sources-badge">
                            📊 {message.sources_found} relevant sources found
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="message-timestamp">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))
            )}
            
            {(isLoading || isTyping) && (
              <div className="message bot message-enter">
                <div className="message-avatar">
                  <div className="avatar bot-avatar">🤖</div>
                </div>
                <div className="message-bubble">
                  <div className="message-content">
                    <TypingAnimation />
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  !isConnected 
                    ? "Connection lost. Please check your backend server..."
                    : totalDocuments > 0 
                      ? "Ask me anything about your documents..." 
                      : "Upload documents first to start chatting..."
                }
                disabled={isLoading || totalDocuments === 0 || !isConnected}
                rows="1"
                className="message-input"
              />
              <button 
                onClick={handleSendMessage}
                disabled={isLoading || !inputMessage.trim() || totalDocuments === 0 || !isConnected}
                className="send-button"
                title="Send message"
              >
                <span className="send-icon" >➡️</span>
              </button>
            </div>
            {!isConnected ? (
              <div className="input-hint error">
                <span>⚠️ Backend server is not available</span>
              </div>
            ) : totalDocuments === 0 ? (
              <div className="input-hint">
                <span>💡 Upload PDF documents to start chatting</span>
              </div>
            ) : (
              <div className="input-hint">
                <span>💬 Ask questions about your uploaded documents</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;