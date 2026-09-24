import axios from "axios";

// Vite dev server proxies /api/* to the FastAPI backend (see vite.config.js)
const client = axios.create({
  baseURL: "/api",
});

// --- Batches (upload happens here now, as a group of files) ---

export const uploadBatch = (files) => {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  return client.post("/batches/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const listBatches = () => client.get("/batches");

export const getBatch = (id) => client.get(`/batches/${id}`);

// --- Documents (read-only - listing / tree lookup for a single document
// inside a batch, used by the tree sidebar) ---

export const listDocuments = () => client.get("/documents");

export const getDocument = (id) => client.get(`/documents/${id}`);

export const getDocumentTree = (id) => client.get(`/documents/${id}/tree`);

// --- Chat (now scoped to a batch, not a single document) ---

export const createChatSession = (batchId) =>
  client.post(`/chat/sessions/${batchId}`);

export const sendMessage = (sessionId, content) =>
  client.post(`/chat/sessions/${sessionId}/messages`, { content });

export const listMessages = (sessionId) =>
  client.get(`/chat/sessions/${sessionId}/messages`);

export default client;