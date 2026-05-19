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

The system is evaluated on the RetailRocket dataset. The offline metrics are computed separately for warm users (those with historical interactions) and cold users (new/unseen users).

### Warm Users Benchmark (`RR_WARM`)

| System | HR@10 | NDCG@50 | Recall@50 | Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Our Hybrid System (Optimized)** | **0.1947** | **0.1520** | **0.3113** | **0.0033** |
| Popularity Baseline | 0.0332 | 0.1027 | 0.2136 | 0.0001 |
| RetailGPT / MTL-SA (LLM SOTA)* | 0.6210 | 0.4740 | 0.8100 | 0.0150 |
| Pure SASRec (Standard Transformer)* | 0.4100 | 0.2010 | 0.2850 | 0.0050 |
| GRU4Rec (Standard RNN)* | 0.3900 | 0.1800 | 0.2600 | 0.0040 |

### Cold Users Benchmark (`RR_COLD`)

| System | HR@10 | NDCG@50 | Recall@50 | Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Our Hybrid System (Optimized)** | **0.2000** | **0.1644** | **0.3030** | **0.0014** |
| Popularity Baseline | 0.0095 | 0.0873 | 0.2110 | 0.0001 |
| Pure SASRec (No Fallback)* | 0.1260 | 0.0500 | 0.2000 | 0.0050 |
| GRU4Rec / NARM* | 0.1700 | 0.0800 | 0.2750 | 0.0040 |
| SR-GNN / STAMP (Graph SOTA)* | 0.3250 | 0.2500 | 0.4750 | 0.0100 |

*\* Reference baselines obtained under different dataset splits or protocols; shown for scale context.*

## Limitations


Published benchmark rows use different datasets and protocols. They are included for scale reference only, not as direct claims of state-of-the-art performance.

## Make Targets

```powershell
make demo
make benchmark
make store
```
