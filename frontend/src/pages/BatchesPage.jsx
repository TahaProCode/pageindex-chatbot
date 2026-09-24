import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listBatches } from "../api/client.js";

export default function BatchesPage() {
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBatches = () =>
      listBatches().then((res) => setBatches(res.data));

    fetchBatches().finally(() => setLoading(false));

    // Poll every 5s so document statuses (uploaded -> parsing -> indexing
    // -> ready) update automatically during a long batch-indexing run,
    // without the user needing to manually refresh the page.
    const interval = setInterval(fetchBatches, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <p style={{ color: "var(--text-dim)" }}>Loading...</p>;

  return (
    <div className="card">
      <h2>Batches</h2>
      {batches.length === 0 && (
        <p style={{ color: "var(--text-dim)" }}>No batches uploaded yet.</p>
      )}
      {batches.map((batch) => {
        const allReady =
          batch.documents.length > 0 &&
          batch.documents.every((d) => d.status === "ready");

        return (
          <div className="doc-list-item" key={batch.id}>
            <div>
              <div>{batch.name}</div>
              <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
                {new Date(batch.created_at).toLocaleString()} ·{" "}
                {batch.documents.length} document(s)
              </div>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                {batch.documents.map((d) => (
                  <span
                    key={d.id}
                    className="status-badge"
                    style={{ marginRight: 6 }}
                  >
                    {d.filename}: {d.status}
                  </span>
                ))}
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              {allReady ? (
                <Link
                  to={`/chat/${batch.id}`}
                  className="nav-link"
                  style={{ color: "var(--accent)" }}
                >
                  Chat →
                </Link>
              ) : (
                <span style={{ color: "var(--text-dim)", fontSize: 13 }}>
                  Indexing...
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
