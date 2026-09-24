import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadBatch } from "../api/client.js";

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files || []));
    setError(null);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      await uploadBatch(files);
      navigate("/batches");
    } catch (err) {
      setError(
        err?.response?.data?.detail || "Upload failed. Please try again.",
      );
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="card">
      <h2>Upload documents</h2>
      <p style={{ color: "var(--text-dim)" }}>
        Select one or more PDFs. Files selected together become one batch -
        they'll be indexed one at a time, then you can chat across all of them
        at once.
      </p>

      <input
        type="file"
        accept="application/pdf"
        multiple
        onChange={handleFileChange}
        disabled={uploading}
      />

      {files.length > 0 && (
        <ul style={{ marginTop: 12 }}>
          {files.map((f, i) => (
            <li key={i} style={{ fontSize: 14 }}>
              {f.name}
            </li>
          ))}
        </ul>
      )}

      {error && (
        <p style={{ color: "var(--danger, #e5484d)", marginTop: 12 }}>
          {error}
        </p>
      )}

      <div style={{ marginTop: 16 }}>
        <button
          onClick={handleUpload}
          disabled={uploading || files.length === 0}
        >
          {uploading
            ? "Uploading..."
            : `Upload ${files.length || ""} document${files.length === 1 ? "" : "s"}`}
        </button>
      </div>
    </div>
  );
}
