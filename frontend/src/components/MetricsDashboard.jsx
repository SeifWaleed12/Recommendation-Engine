import React, { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const colors = {
  "Our Hybrid System (Optimized)": "#2563eb",
  "Popularity Baseline": "#f97316",
  "RetailGPT / MTL-SA (LLM SOTA)": "#9333ea",
  "Pure SASRec (Standard Transformer)": "#0d9488",
  "GRU4Rec (Standard RNN)": "#eab308",
  "Pure SASRec (No Fallback)": "#dc2626",
  "GRU4Rec / NARM": "#a8a29e",
  "SR-GNN / STAMP (Graph SOTA)": "#ec4899"
};

const COHORTS = [
  { id: "RR_WARM", label: "RetailRocket (Warm)" },
  { id: "RR_COLD", label: "RetailRocket (Cold)" }
];

export default function MetricsDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [activeCohort, setActiveCohort] = useState("RR_WARM");

  useEffect(() => {
    fetch("/reports/benchmark_4way.json")
      .then((res) => {
        if (!res.ok) throw new Error("Report not found");
        return res.json();
      })
      .then(setData)
      .catch(() => setError("Benchmark 4-way report not found. Run python -m evaluation.benchmark"));
  }, []);

  const chartData = useMemo(() => {
    if (!data || !data[activeCohort]) return [];
    
    const metricsToDisplay = [
      ["HR@10", "hr_10"],
      ["HR@50", "hr_50"],
      ["NDCG@50", "ndcg_50"],
      ["Recall@50", "recall_50"],
      ["Cand Rec@50", "candidate_recall_50"],
      ["Cand Rec@500", "candidate_recall_500"]
    ];

    const systemsArray = data[activeCohort];

    return metricsToDisplay.map(([label, key]) => {
      const row = { metric: label };
      systemsArray.forEach((system) => {
        row[system.name] = system[key];
      });
      return row;
    });
  }, [data, activeCohort]);

  const insight = useMemo(() => {
    if (!data || !data[activeCohort]) return "";
    const systemsArray = data[activeCohort];
    const systemsMap = Object.fromEntries(systemsArray.map((row) => [row.name, row]));
    
    const our = systemsMap["Our Hybrid System"];
    const pop = systemsMap["Popularity Baseline"];
    const cf = systemsMap["User-Based CF Baseline"];
    
    if (!our) return "Run benchmarks to calculate lift against baselines.";
    
    let text = `Our Hybrid System achieved a ${Math.round(our.hr_10 * 100)}% Hit Rate @ 10 on the ${activeCohort} cohort.`;
    
    if (pop && pop.hr_10 > 0) {
      const popLift = ((our.hr_10 - pop.hr_10) / pop.hr_10) * 100;
      text += ` This is a ${popLift.toFixed(0)}% lift over the Popularity Baseline.`;
    } else if (pop && pop.hr_10 === 0 && our.hr_10 > 0) {
      text += ` The Popularity Baseline failed completely (0% HR@10) on this cohort.`;
    }

    return text;
  }, [data, activeCohort]);

  return (
    <main className="mx-auto max-w-7xl space-y-6 px-4 py-6">
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h1 className="text-2xl font-black mb-4">
          Benchmarks
        </h1>
        
        {/* Tab Selector */}
        <div className="flex space-x-2 border-b border-slate-200 mb-6 pb-2">
          {COHORTS.map(cohort => (
            <button
              key={cohort.id}
              onClick={() => setActiveCohort(cohort.id)}
              className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${
                activeCohort === cohort.id 
                  ? "bg-blue-50 text-blue-700 border-b-2 border-blue-600" 
                  : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
              }`}
            >
              {cohort.label}
            </button>
          ))}
        </div>

        {error && <p className="mt-3 text-rose-700">{error}</p>}
        
        {data && data[activeCohort] && (
          <>
            <div className="mt-5 h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {/* Remove hardcoded reference lines since these datasets are completely different */}
                  {data[activeCohort].map((system) => (
                    <Bar 
                      key={system.name} 
                      dataKey={system.name} 
                      fill={colors[system.name] || "#64748b"} 
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div className="rounded-lg bg-blue-50 p-4 mt-6 font-semibold text-blue-900 shadow-inner">
              {insight}
            </div>
            
            {/* Display Coverage Metric separately since it's not on the same scale (0-1 usually very small) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              {data[activeCohort].map(system => (
                <div key={`${system.name}-cov`} className="rounded-lg border border-slate-100 p-3 flex justify-between bg-slate-50 text-sm">
                  <span className="text-slate-600 font-medium truncate pr-2">{system.name} Coverage:</span>
                  <span className="font-bold text-slate-800">{(system.coverage * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
