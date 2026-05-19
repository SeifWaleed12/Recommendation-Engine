# Recommendation Engine

Production ML recommendation system with FastAPI serving, offline evaluation, benchmark comparison, and a React storefront/demo UI.

## Run Locally

```powershell
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Results

Run `make benchmark` or `python evaluation/benchmark.py` to fill real values.

```text
System                                      NDCG@10  R@50    Coverage
------------------------------------------------------------------------
Our System (Hybrid)                         0.000    0.000   0.000
Popularity Baseline                         0.000    0.000   0.000
Random Baseline                             0.000    0.000   0.000
User-Based CF Baseline                      0.000    0.000   0.000

NCF (He 2017, MovieLens-1M) *               0.416    N/A     N/A
LightGCN (He 2020, Gowalla) *               0.155    0.183   N/A

* Different dataset - not directly comparable
```

NDCG and MAP measure ranking quality: better products near the top improve discovery and conversion. Recall measures how many relevant held-out products the system recovers. Coverage shows whether recommendations use the catalog broadly instead of repeating only bestsellers. Diversity estimates variety within each recommendation list.

## Limitations

Published benchmark rows use different datasets and protocols. They are included for scale reference only, not as direct claims of state-of-the-art performance.

## Make Targets

```powershell
make demo
make benchmark
make store
```
