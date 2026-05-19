import React, { useState } from "react";
import AutoDemo from "./components/AutoDemo.jsx";
import CatalogBrowser from "./components/CatalogBrowser.jsx";
import EventSimulator from "./components/EventSimulator.jsx";
import ItemSimilarity from "./components/ItemSimilarity.jsx";
import MetricsDashboard from "./components/MetricsDashboard.jsx";
import Storefront from "./components/Storefront.jsx";
import UserExplorer from "./components/UserExplorer.jsx";

const tabs = [
  ["store", "Store"],
  ["demo", "Demo"],
  ["recommendations", "Recommendations"],
  ["similar", "Similar Items"],
  ["benchmarks", "Benchmarks"],
  ["simulator", "Simulator"]
];

export default function App() {
  const [tab, setTab] = useState("store");
  return (
    <div className="min-h-screen">
      <nav className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl gap-2 overflow-x-auto px-4 py-3">
          {tabs.map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium ${
                tab === id ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </nav>
      {tab === "store" && <Storefront />}
      {tab === "demo" && <AutoDemo />}
      {tab === "recommendations" && <UserExplorer />}
      {tab === "similar" && <ItemSimilarity />}
      {tab === "benchmarks" && <MetricsDashboard />}
      {tab === "simulator" && (
        <main className="mx-auto max-w-7xl px-4 py-6">
          <CatalogBrowser />
          <EventSimulator />
        </main>
      )}
    </div>
  );
}
