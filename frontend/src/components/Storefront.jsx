import { Search, ShoppingCart } from "lucide-react";
import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import ProductCard, { categoryColor } from "./ProductCard.jsx";
import ProductDetail from "./ProductDetail.jsx";

const categories = ["Electronics", "Computers", "Phones", "Audio", "Cameras", "Gaming", "Accessories"];

function SkeletonRow() {
  return (
    <div className="flex gap-4 overflow-hidden">
      {Array.from({ length: 5 }).map((_, idx) => (
        <div key={idx} className="h-56 w-56 shrink-0 animate-pulse rounded-lg bg-slate-200" />
      ))}
    </div>
  );
}

function ProductRow({ title, items = [], onOpen, badge }) {
  if (!items.length) return null;
  return (
    <section className="space-y-3">
      <h2 className="text-xl font-bold text-slate-950">{title}</h2>
      <div className="flex gap-4 overflow-x-auto pb-3">
        {items.map((item) => (
          <ProductCard key={item.item_id} item={item} badge={badge} onClick={onOpen} />
        ))}
      </div>
    </section>
  );
}

export default function Storefront() {
  const [userId, setUserId] = useState("");
  const [draftUserId, setDraftUserId] = useState("");
  const [home, setHome] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedItem, setSelectedItem] = useState(null);

  useEffect(() => {
    let timer;
    api.get("/api/v1/demo/users").then((res) => {
      const warm = res.data.warm_user_id || "guest";
      const cold = res.data.cold_user_id || "guest";
      
      setUserId(warm);
      setDraftUserId(warm);

      let isWarm = true;
      timer = setInterval(() => {
        isWarm = !isWarm;
        const nextUser = isWarm ? warm : cold;
        setUserId(nextUser);
        setDraftUserId(nextUser);
      }, 4000);
    }).catch(() => {
      setUserId("guest");
      setDraftUserId("guest");
    });

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!userId) return;
    setLoading(true);
    setError("");
    api.get(`/api/v1/storefront/home/${encodeURIComponent(userId)}`)
      .then((res) => setHome(res.data))
      .catch(() => setError("Couldn't load recommendations. Try again."))
      .finally(() => setLoading(false));
  }, [userId]);

  if (selectedItem) {
    return <ProductDetail itemId={selectedItem.item_id} userId={userId} onBack={() => setSelectedItem(null)} />;
  }

  return (
    <main className="bg-slate-50">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-4">
          <div className="text-2xl font-black tracking-tight">RecShop</div>
          <div className="relative flex-1">
            <Search className="absolute left-3 top-2.5 h-5 w-5 text-slate-400" />
            <input className="w-full rounded-lg border border-slate-300 py-2 pl-10 pr-3" placeholder="Search products" />
          </div>
          <ShoppingCart className="h-6 w-6 text-slate-700" />
        </div>
        <div className="mx-auto flex max-w-7xl gap-2 overflow-x-auto px-4 pb-4">
          {categories.map((cat) => (
            <button key={cat} className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium hover:bg-slate-100">
              {cat}
            </button>
          ))}
        </div>
      </header>

      <div className="mx-auto max-w-7xl space-y-8 px-4 py-6">
        <section className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-600">Viewing as:</span>
            <input value={draftUserId} onChange={(e) => setDraftUserId(e.target.value)} className="rounded-md border px-3 py-1 font-mono text-sm" />
            <button onClick={() => setUserId(draftUserId)} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white">Apply</button>
          </div>
          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${home?.is_personalized ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-600"}`}>
            {home?.is_personalized ? "Personalized" : "Browsing as Guest"}
          </span>
        </section>

        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-700">
            {error} <button onClick={() => setUserId(userId)} className="ml-2 font-semibold underline">Retry</button>
          </div>
        )}

        {loading ? <SkeletonRow /> : (
          <>
            <section className="grid gap-4 md:grid-cols-3">
              {(home?.hero_items || []).slice(0, 3).map((item) => (
                <button key={item.item_id} onClick={() => setSelectedItem(item)} className={`rounded-xl bg-gradient-to-br ${categoryColor(item.category)} p-6 text-left text-white shadow-sm`}>
                  <div className="mb-3 text-sm font-semibold opacity-85">{item.category}</div>
                  <h1 className="line-clamp-2 text-2xl font-black">{item.title}</h1>
                  <p className="mt-3 line-clamp-2 text-sm opacity-90">{item.description?.slice(0, 80)}</p>
                  <div className="mt-5 text-xl font-bold">${Number(item.price || 0).toFixed(2)}</div>
                </button>
              ))}
            </section>
            <ProductRow title="Recommended for You" items={home?.for_you} onOpen={setSelectedItem} />
            <ProductRow title={home?.because_you_viewed?.[0]?.badge || "Because You Viewed"} items={home?.because_you_viewed} onOpen={setSelectedItem} />
            <ProductRow title="Trending Now" items={home?.trending_now} onOpen={setSelectedItem} badge="Trending" />
            {(home?.category_rows || []).map((row) => (
              <ProductRow key={row.category} title={row.category} items={row.items} onOpen={setSelectedItem} />
            ))}
          </>
        )}
      </div>
    </main>
  );
}
