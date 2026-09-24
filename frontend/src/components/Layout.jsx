import React from "react";
import { NavLink, Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>PageIndex Chatbot</h1>
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
        >
          Upload
        </NavLink>
        <NavLink
          to="/batches"
          className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
        >
          Batches
        </NavLink>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
