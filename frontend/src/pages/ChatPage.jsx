import React, { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  createChatSession,
  sendMessage,
  listMessages,
  getBatch,
  getDocumentTree,
} from "../api/client.js";
import TreeView from "../components/TreeView.jsx";

export default function ChatPage() {
  const { batchId } = useParams();
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [batch, setBatch] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [tree, setTree] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    createChatSession(batchId)
      .then((res) => {
        setSessionId(res.data.id);
        return listMessages(res.data.id);
      })
      .then((res) => {
        if (res) setMessages(res.data);
      });

    getBatch(batchId).then((res) => {
      setBatch(res.data);
      if (res.data.documents.length > 0) {
        setSelectedDocId(res.data.documents[0].id);
      }
    });
  }, [batchId]);

  // Tree sidebar shows one document at a time - a batch can hold many
  // documents, so we default to the first one and let the user switch.
  useEffect(() => {
    if (!selectedDocId) return;
    getDocumentTree(selectedDocId)
      .then((res) => setTree(res.data))
      .catch(() => setTree([]));
  }, [selectedDocId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId) return;
    const userText = input;
    setInput("");
    setMessages((prev) => [
      ...prev,
      {
        id: `temp-${Date.now()}`,
        role: "user",
        content: userText,
        cited_sections: [],
      },
    ]);
    setSending(true);
    try {
      const res = await sendMessage(sessionId, userText);
      setMessages((prev) => [...prev, res.data]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          content: "Error: could not get a response.",
          cited_sections: [],
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="chat-layout">
      <div className="chat-window">
        <h2>Chat{batch ? ` - ${batch.name}` : ""}</h2>
        <div className="chat-messages">
          {messages.map((m) => (
            <div key={m.id} className={`chat-bubble-wrap ${m.role}`}>
              <div className={`chat-bubble ${m.role}`}>{m.content}</div>
              {m.cited_sections && m.cited_sections.length > 0 && (
                <div className="citation-row">
                  {m.cited_sections.map((c) => (
                    <span key={c.id} className="citation-pill">
                      {c.filename ? `${c.filename} - ` : ""}
                      {c.title}
                      {(c.page_start || c.page_end) &&
                        ` (p${c.page_start}${c.page_end && c.page_end !== c.page_start ? `-${c.page_end}` : ""})`}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <div className="chat-input-row">
          <input
            type="text"
            placeholder="Ask a question about these documents..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button onClick={handleSend} disabled={sending || !input.trim()}>
            {sending ? "..." : "Send"}
          </button>
        </div>
      </div>

      <aside className="tree-sidebar">
        <h3>Document structure</h3>
        {batch && batch.documents.length > 1 && (
          <select
            value={selectedDocId || ""}
            onChange={(e) => setSelectedDocId(e.target.value)}
            style={{ marginBottom: 12, width: "100%" }}
          >
            {batch.documents.map((d) => (
              <option key={d.id} value={d.id}>
                {d.filename} ({d.status})
              </option>
            ))}
          </select>
        )}
        <TreeView tree={tree} />
      </aside>
    </div>
  );
}