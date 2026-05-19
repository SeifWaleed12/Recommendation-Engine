import { Play } from "lucide-react";
import React, { useMemo, useState } from "react";
import { api } from "../api.js";
import { categoryColor } from "./ProductCard.jsx";

const messages = [
  "Creating cold user...",
  "Loading warm user profile...",
  "Running recommendation pipeline...",
  "Capturing step results..."
];

const actionColors = {
  view: "bg-blue-100 text-blue-700",
  add_to_cart: "bg-yellow-100 text-yellow-800",
  purchase: "bg-emerald-100 text-emerald-700",
  get_recs: "bg-purple-100 text-purple-700"
};

function diffItems(steps, index) {
  const current = steps[index];
  if (current?.action !== "get_recs") return { added: new Set(), removed: [] };
  const currentIds = new Set((current.recommendations_after || []).map((item) => item.item_id));
  const previous = [...steps.slice(0, index)].reverse().find((step) => step.action === "get_recs");
  const previousItems = previous?.recommendations_after || [];
  const previousIds = new Set(previousItems.map((item) => item.item_id));
  return {
    added: new Set([...currentIds].filter((item) => !previousIds.has(item))),
    removed: previousItems.filter((item) => !currentIds.has(item))
  };
}

function MiniGrid({ step, steps, index }) {
  if (step?.action !== "get_recs") return null;
  const { added, removed } = diffItems(steps, index);
  return (
    <div className="mt-4 grid grid-cols-2 gap-3 xl:grid-cols-3">
      {(step.recommendations_after || []).slice(0, 6).map((item) => (
        <div key={item.item_id} className={`relative rounded-lg border bg-white p-2 ${added.has(item.item_id) ? "border-emerald-500" : "border-slate-200"}`}>
          {added.has(item.item_id) && <span className="absolute right-2 top-2 rounded-full bg-emerald-500 px-2 py-0.5 text-[10px] font-bold text-white">New</span>}
          <div className={`mb-2 h-16 rounded bg-gradient-to-br ${categoryColor(item.category)}`} />
          <div className="line-clamp-1 text-xs font-semibold">{item.title}</div>
          <div className="mt-1 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600">{item.retrieval_source || "rec"}</div>
        </div>
      ))}
      {removed.slice(0, 2).map((item) => (
        <div key={`removed-${item.item_id}`} className="rounded-lg border border-slate-200 bg-slate-100 p-2 opacity-70">
          <div className="mb-2 h-16 rounded bg-slate-300" />
          <div className="line-clamp-1 text-xs font-semibold line-through">{item.title}</div>
          <div className="mt-1 rounded-full bg-slate-300 px-2 py-0.5 text-[10px] text-slate-700">Removed</div>
        </div>
      ))}
    </div>
  );
}

function DemoColumn({ title, data, error, stepIndex }) {
  const step = data?.steps?.[stepIndex];
  const progress = data?.personalization_progression?.[stepIndex] ?? 0;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-xl font-black">{title}</h2>
      {error && <div className="mt-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}</div>}
      {data && (
        <>
          <div className="mt-3 flex items-center justify-between gap-3">
            <code className="truncate text-xs text-slate-500">{data.user_id}</code>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold">{data.total_interactions} interactions</span>
          </div>
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs font-medium text-slate-500">
              <span>Personalization Score</span>
              <span>{Math.round(progress * 100)}%</span>
            </div>
            <div className="h-3 rounded-full bg-slate-100">
              <div className="h-3 rounded-full bg-emerald-500 transition-all duration-500" style={{ width: `${progress * 100}%` }} />
            </div>
          </div>
          {step && (
            <div className="mt-6">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="font-bold">Step {step.step_number}: {step.action} {step.item_title || ""}</h3>
                <span className={`rounded-full px-3 py-1 text-xs font-bold ${actionColors[step.action] || "bg-slate-100"}`}>{step.action.toUpperCase()}</span>
              </div>
              <p className="mt-3 text-sm text-slate-600">{step.explanation}</p>
              <MiniGrid step={step} steps={data.steps} index={stepIndex} />
            </div>
          )}
          {stepIndex >= data.steps.length - 1 && (
            <div className="mt-5 rounded-lg bg-slate-50 p-4 text-sm text-slate-700">{data.final_profile_summary}</div>
          )}
        </>
      )}
    </section>
  );
}

export default function AutoDemo() {
  const [cold, setCold] = useState(null);
  const [warm, setWarm] = useState(null);
  const [coldError, setColdError] = useState("");
  const [warmError, setWarmError] = useState("");
  const [loading, setLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  const maxSteps = Math.max(cold?.steps?.length || 0, warm?.steps?.length || 0);
  const loadingText = useMemo(() => messages[Math.floor(Date.now() / 1500) % messages.length], [loading]);

  async function runDemo() {
    setLoading(true);
    setColdError("");
    setWarmError("");
    setStepIndex(0);
    const coldReq = api.post("/api/v1/demo/run", { user_type: "cold_user" });
    const warmReq = api.post("/api/v1/demo/run", { user_type: "warm_user" });
    const [coldRes, warmRes] = await Promise.allSettled([coldReq, warmReq]);
    if (coldRes.status === "fulfilled") setCold(coldRes.value.data); else setColdError("Cold user demo failed.");
    if (warmRes.status === "fulfilled") setWarm(warmRes.value.data); else setWarmError("Warm user demo failed.");
    setLoading(false);
  }

  return (
    <main className="mx-auto max-w-7xl space-y-6 px-4 py-6">
      <div className="flex flex-col items-center gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <button onClick={runDemo} disabled={loading} className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-5 py-3 font-bold text-white disabled:opacity-60">
          <Play className="h-5 w-5" /> Run Demo
        </button>
        {loading && <div className="animate-pulse text-sm font-medium text-slate-600">{loadingText}</div>}
        {maxSteps > 0 && (
          <label className="w-full max-w-md text-center text-sm font-semibold">
            Step {stepIndex + 1} of {maxSteps}
            <input type="range" min="0" max={maxSteps - 1} value={stepIndex} onChange={(e) => setStepIndex(Number(e.target.value))} className="mt-2 w-full" />
          </label>
        )}
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <DemoColumn title="Cold User" data={cold} error={coldError} stepIndex={stepIndex} />
        <DemoColumn title="Warm User" data={warm} error={warmError} stepIndex={stepIndex} />
      </div>
      {cold && warm && stepIndex >= maxSteps - 1 && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-5 text-center font-semibold text-blue-900">
          The cold user needed multiple interactions before recommendations became personalized. The warm user's recommendations were personalized from the first request.
        </div>
      )}
    </main>
  );
}
