import React, { useState, useEffect, useRef } from 'react';
import './App.css';


const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! Upload your documents (PDF, TXT, MD) on the left to start asking questions with RAG.'
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loadingAnswer, setLoadingAnswer] = useState(false);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loadingAnswer]);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setUploadStatus({ type: 'loading', message: `Processing & chunking ${file.name}...` });

    try {
      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus({
          type: 'success',
          message: `Uploaded! ${result.document.total_chunks} chunks extracted.`
        });
        fetchDocuments();
      } else {
        setUploadStatus({
          type: 'error',
          message: result.detail || 'Upload failed.'
        });
      }
    } catch (err) {
      setUploadStatus({
        type: 'error',
        message: 'Could not connect to backend server.'
      });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputQuery.trim() || loadingAnswer) return;

    const userMessage = { role: 'user', content: inputQuery.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInputQuery('');
    setLoadingAnswer(true);

    try {
      // Placeholder for query endpoint, simulates bot response or calls backend
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `I received your question regarding your uploaded documents: "${userMessage.content}". (Embedding and retrieval query will return relevant chunks once query endpoint is linked).`
          }
        ]);
        setLoadingAnswer(false);
      }, 1000);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, an error occurred while processing your request.' }
      ]);
      setLoadingAnswer(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar for Upload and Document list */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>📄 Document Hub</h2>
          <p>Upload documents for RAG Knowledge base</p>
        </div>

        <div
          className="upload-card"
          onClick={() => !uploading && fileInputRef.current?.click()}
        >
          <div className="upload-icon">📁</div>
          <p style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>
            Click to upload PDF, TXT or MD
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
            disabled={uploading}
          />
          <button className="upload-btn" disabled={uploading}>
            {uploading ? 'Processing...' : 'Choose File'}
          </button>
        </div>

        {uploadStatus && (
          <div className={`upload-status status-${uploadStatus.type}`}>
            {uploadStatus.message}
          </div>
        )}

        <div className="doc-list-section">
          <h3>Indexed Documents ({documents.length})</h3>
          {documents.length === 0 ? (
            <p style={{ fontSize: '0.8rem', color: '#64748b' }}>No documents uploaded yet.</p>
          ) : (
            documents.map((doc) => (
              <div key={doc.document_id} className="doc-item">
                <div className="doc-title" title={doc.filename}>
                  {doc.filename}
                </div>
                <div className="doc-meta">
                  <span>{doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'}</span>
                  <span>{doc.total_chunks} chunks</span>
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="chat-main">
        <header className="chat-header">
          <h1>💬 RAG Chatbot</h1>
          <span className="badge">FastAPI & MongoDB Motor</span>
        </header>

        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message-bubble message-${msg.role}`}
            >
              {msg.content}
            </div>
          ))}

          {loadingAnswer && (
            <div className="message-bubble message-assistant">
              <em>Thinking and retrieving context...</em>
            </div>
          )}

          <div ref={chatBottomRef} />
        </div>

        <div className="chat-input-container">
          <form className="chat-form" onSubmit={handleSendMessage}>
            <input
              type="text"
              className="chat-input"
              placeholder="Ask anything about your uploaded documents..."
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              disabled={loadingAnswer}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={loadingAnswer || !inputQuery.trim()}
            >
              Send
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;

