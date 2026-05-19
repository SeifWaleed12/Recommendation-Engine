import React, { useState } from "react";
import { api } from "../api.js";
import ProductCard from "./ProductCard.jsx";

export default function UserExplorer() {
  const [userId, setUserId] = useState("");
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  async function load() {
    setError("");
    try {
      const res = await api.get(`/api/v1/recommendations/${encodeURIComponent(userId)}`, { params: { n: 10 } });
      setItems(res.data.map((row) => ({ ...row, title: row.item_id, price: row.score || 0, category: row.retrieval_source || "Recommendation" })));
    } catch {
      setError("Could not load recommendations.");
    }
  }

  return (
    <main className="mx-auto max-w-7xl space-y-5 px-4 py-6">
      <div className="rounded-lg bg-white p-5 shadow-sm">
        <h1 className="text-2xl font-black">Recommendations</h1>
        <div className="mt-4 flex gap-2">
          <input value={userId} onChange={(e) => setUserId(e.target.value)} className="rounded-md border px-3 py-2 font-mono" placeholder="user_id" />
          <button onClick={load} className="rounded-md bg-slate-900 px-4 py-2 font-semibold text-white">Load</button>
        </div>
        {error && <p className="mt-3 text-rose-700">{error}</p>}
      </div>
      <div className="flex gap-4 overflow-x-auto">
        {items.map((item) => <ProductCard key={item.item_id} item={item} />)}
      </div>
    </main>
  );
}
