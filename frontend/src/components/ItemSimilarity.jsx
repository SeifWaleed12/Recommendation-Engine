import React, { useState } from "react";
import { api } from "../api.js";

export default function ItemSimilarity() {
  const [itemId, setItemId] = useState("");
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  async function load() {
    setError("");
    try {
      const res = await api.get(`/api/v1/items/${encodeURIComponent(itemId)}/similar`, { params: { n: 10 } });
      setItems(res.data);
    } catch {
      setError("Could not load similar items.");
    }
  }

  return (
    <main className="mx-auto max-w-7xl space-y-5 px-4 py-6">
      <div className="rounded-lg bg-white p-5 shadow-sm">
        <h1 className="text-2xl font-black">Similar Items</h1>
        <div className="mt-4 flex gap-2">
          <input value={itemId} onChange={(e) => setItemId(e.target.value)} className="rounded-md border px-3 py-2 font-mono" placeholder="item_id" />
          <button onClick={load} className="rounded-md bg-slate-900 px-4 py-2 font-semibold text-white">Load</button>
        </div>
        {error && <p className="mt-3 text-rose-700">{error}</p>}
      </div>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {items.map((item) => <pre key={item.item_id || item.id} className="overflow-auto rounded-lg bg-white p-4 text-xs shadow-sm">{JSON.stringify(item, null, 2)}</pre>)}
      </div>
    </main>
  );
}
