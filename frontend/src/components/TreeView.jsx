import React, { useState } from "react";

function TreeNode({ node, depth }) {
  const [expanded, setExpanded] = useState(depth < 1);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div style={{ marginLeft: depth > 0 ? 14 : 0 }}>
      <div
        className="tree-node-row"
        onClick={() => hasChildren && setExpanded((e) => !e)}
        style={{ cursor: hasChildren ? "pointer" : "default" }}
      >
        {hasChildren && (
          <span className="tree-toggle">{expanded ? "▾" : "▸"}</span>
        )}
        {!hasChildren && (
          <span className="tree-toggle tree-toggle-leaf">•</span>
        )}
        <span className="tree-title">{node.title}</span>
        {(node.page_start || node.page_end) && (
          <span className="tree-pages">
            p{node.page_start}
            {node.page_end && node.page_end !== node.page_start
              ? `-${node.page_end}`
              : ""}
          </span>
        )}
      </div>
      {node.summary && <div className="tree-summary">{node.summary}</div>}
      {hasChildren && expanded && (
        <div>
          {node.children.map((child) => (
            <TreeNode key={child.id} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function TreeView({ tree }) {
  if (!tree || tree.length === 0) {
    return (
      <p style={{ color: "var(--text-dim)", fontSize: 13 }}>
        No structure available yet.
      </p>
    );
  }

  return (
    <div className="tree-view">
      {tree.map((node) => (
        <TreeNode key={node.id} node={node} depth={0} />
      ))}
    </div>
  );
}
