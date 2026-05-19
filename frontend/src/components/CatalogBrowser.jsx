import React, { useState } from "react";
import { api } from "../api.js";

export default function CatalogBrowser() {
  const [category, setCategory] = useState("Electronics");
  const [items, setItems] = useState([]);

  async function load() {
    const res = await api.get("/api/v1/items", { params: { category, limit: 20 } });
    setItems(res.data);
  }

  return (
    <section className="mb-6 rounded-lg bg-white p-5 shadow-sm">
      <h2 className="text-xl font-black">Catalog Browser</h2>
      <div className="mt-4 flex gap-2">
        <input value={category} onChange={(e) => setCategory(e.target.value)} className="rounded-md border px-3 py-2" />
        <button onClick={load} className="rounded-md bg-slate-900 px-4 py-2 font-semibold text-white">Browse</button>
      </div>
      <div className="mt-4 grid gap-2 md:grid-cols-2">
        {items.map((item) => <div key={item.external_id || item.item_id} className="rounded border p-3 text-sm">{item.title || item.external_id}</div>)}
      </div>
    </section>
  );
}
