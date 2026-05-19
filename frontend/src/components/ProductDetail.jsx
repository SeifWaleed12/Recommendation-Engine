import { Heart, ShoppingCart } from "lucide-react";
import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { categoryColor } from "./ProductCard.jsx";

function SideItem({ item, onOpen }) {
  const score = Math.round(Number(item.recommendation_score || 0.86) * 100);
  return (
    <button onClick={() => onOpen(item.item_id)} className="flex gap-3 rounded-lg border border-slate-200 bg-white p-3 text-left hover:shadow-sm">
      <div className={`h-16 w-16 shrink-0 rounded-md bg-gradient-to-br ${categoryColor(item.category)}`} />
      <div className="min-w-0 flex-1">
        <div className="line-clamp-1 text-sm font-semibold">{item.title}</div>
        <div className="mt-1 text-sm font-bold text-emerald-700">${Number(item.price || 0).toFixed(2)}</div>
        <div className="text-xs text-slate-500">{score}% match</div>
      </div>
    </button>
  );
}

function Section({ title, items, onOpen, limit }) {
  return (
    <section className="space-y-2">
      <h3 className="font-bold text-slate-900">{title}</h3>
      {(items || []).slice(0, limit).map((item) => <SideItem key={item.item_id} item={item} onOpen={onOpen} />)}
    </section>
  );
}

export default function ProductDetail({ itemId, userId, onBack }) {
  const [currentItemId, setCurrentItemId] = useState(itemId);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setCurrentItemId(itemId);
  }, [itemId]);

  useEffect(() => {
    setLoading(true);
    setError("");
    const params = userId ? { user_id: userId } : {};
    api.get(`/api/v1/storefront/product/${encodeURIComponent(currentItemId)}`, { params })
      .then((res) => setData(res.data))
      .catch(() => setError("Couldn't load product. Try again."))
      .finally(() => setLoading(false));
  }, [currentItemId, userId]);

  if (loading) return <main className="mx-auto max-w-7xl px-4 py-8">Loading product...</main>;
  if (error) return <main className="mx-auto max-w-7xl px-4 py-8 text-rose-700">{error}</main>;

  const product = data.product;
  return (
    <main className="mx-auto max-w-7xl px-4 py-6">
      <button onClick={onBack} className="mb-4 text-sm font-semibold text-slate-600 hover:text-slate-950">Back to store</button>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(320px,2fr)]">
        <section className="space-y-5">
          <div className={`flex aspect-[16/10] items-center justify-center rounded-xl bg-gradient-to-br ${categoryColor(product.category)}`} />
          <div>
            <h1 className="text-3xl font-black text-slate-950">{product.title}</h1>
            <p className="mt-2 text-sm text-slate-500">{product.brand} • {product.category}</p>
            <p className="mt-4 text-3xl font-black text-emerald-700">${Number(product.price || 0).toFixed(2)}</p>
          </div>
          <p className="text-slate-700">{product.description}</p>
          <div className="flex gap-3">
            <button className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-5 py-3 font-semibold text-white">
              <ShoppingCart className="h-5 w-5" /> Add to Cart
            </button>
            <button className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-5 py-3 font-semibold">
              <Heart className="h-5 w-5" /> Add to Wishlist
            </button>
          </div>
        </section>
        <aside className="space-y-6">
          <Section title="You May Also Like" items={data.you_may_also_like} limit={4} onOpen={setCurrentItemId} />
          <Section title="Frequently Bought Together" items={data.frequently_bought_together} limit={3} onOpen={setCurrentItemId} />
          <Section title={`More in ${product.category}`} items={data.category_picks} limit={4} onOpen={setCurrentItemId} />
        </aside>
      </div>
    </main>
  );
}
