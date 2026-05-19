import React, { useState } from "react";
import { api } from "../api.js";

export default function EventSimulator() {
  const [form, setForm] = useState({ user_id: "", item_id: "", event_type: "view", session_id: "frontend_session" });
  const [status, setStatus] = useState("");

  async function submit() {
    setStatus("");
    try {
      await api.post("/api/v1/events", form);
      setStatus("Event recorded.");
    } catch {
      setStatus("Event failed.");
    }
  }

  return (
    <section className="rounded-lg bg-white p-5 shadow-sm">
      <h2 className="text-xl font-black">Event Simulator</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-4">
        {["user_id", "item_id", "session_id"].map((field) => (
          <input key={field} value={form[field]} onChange={(e) => setForm({ ...form, [field]: e.target.value })} className="rounded-md border px-3 py-2" placeholder={field} />
        ))}
        <select value={form.event_type} onChange={(e) => setForm({ ...form, event_type: e.target.value })} className="rounded-md border px-3 py-2">
          <option value="view">view</option>
          <option value="add_to_cart">add_to_cart</option>
          <option value="purchase">purchase</option>
        </select>
      </div>
      <button onClick={submit} className="mt-4 rounded-md bg-slate-900 px-4 py-2 font-semibold text-white">Send Event</button>
      {status && <p className="mt-3 text-sm text-slate-600">{status}</p>}
    </section>
  );
}
