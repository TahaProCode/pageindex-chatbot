import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listDocuments } from "../api/client.js";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDocs = () =>
      listDocuments().then((res) => setDocuments(res.data));

    fetchDocs().finally(() => setLoading(false));

    // Poll every 5s so statuses (uploaded -> parsing -> indexing -> ready)
    // update automatically during a long batch-indexing run, without the
    // user needing to manually refresh the page.
    const interval = setInterval(fetchDocs, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <p style={{ color: "var(--text-dim)" }}>Loading...</p>;

  return (
    <div className="card">
      <h2>Documents</h2>
      {documents.length === 0 && (
        <p style={{ color: "var(--text-dim)" }}>No documents uploaded yet.</p>
      )}
      {documents.map((doc) => (
        <div className="doc-list-item" key={doc.id}>
          <div>
            <div>{doc.filename}</div>
            <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
              {new Date(doc.created_at).toLocaleString()}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="status-badge">{doc.status}</span>
            <Link
              to={`/chat/${doc.id}`}
              className="nav-link"
              style={{ color: "var(--accent)" }}
            >
              Chat →
            </Link>
          </div>
        </div>
      ))}
    </div>
  );
}
