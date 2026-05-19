import React from "react";
import { ShoppingCart } from "lucide-react";

export function categoryColor(category = "default") {
  let hash = 0;
  for (const char of category) hash = char.charCodeAt(0) + ((hash << 5) - hash);
  const colors = [
    "from-blue-500 to-cyan-400",
    "from-emerald-500 to-lime-400",
    "from-rose-500 to-orange-400",
    "from-violet-500 to-fuchsia-400",
    "from-amber-500 to-yellow-300",
    "from-slate-700 to-slate-500"
  ];
  return colors[Math.abs(hash) % colors.length];
}

export default function ProductCard({ item, onClick, badge }) {
  return (
    <button
      onClick={() => onClick?.(item)}
      className="w-56 shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-white text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
    >
      <div className={`flex h-36 items-center justify-center bg-gradient-to-br ${categoryColor(item.category)}`}>
        <ShoppingCart className="h-10 w-10 text-white/90" />
      </div>
      <div className="space-y-2 p-3">
        <div className="line-clamp-2 min-h-10 text-sm font-semibold text-slate-900">{item.title}</div>
        <div className="flex items-center justify-between">
          <span className="font-bold text-emerald-700">${Number(item.price || 0).toFixed(2)}</span>
          {(badge || item.retrieval_source) && (
            <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-medium text-slate-700">
              {badge || item.retrieval_source}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}
