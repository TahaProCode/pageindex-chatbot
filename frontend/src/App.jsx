// App.jsx
import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import UploadPage from "./pages/UploadPage.jsx";
import BatchesPage from "./pages/BatchesPage.jsx";
import ChatPage from "./pages/ChatPage.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<UploadPage />} />
        <Route path="/batches" element={<BatchesPage />} />
        <Route path="/chat/:batchId" element={<ChatPage />} />
      </Route>
    </Routes>
  );
}
