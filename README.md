# SPOTIFY-THING — AI Music Discovery & Adaptive Shuffle Companion Extension

An independent AI-powered Chrome Companion Extension (Manifest V3) and PyTorch Recommender & Adaptive Shuffle Engine designed to work alongside Spotify.

**Spotify handles music playback. SPOTIFY-THING provides the independent neural recommendation intelligence, Adaptive Shuffle sequencing, transparent music insights, and online bandit personalization.**

---

## Target Architecture

```text
SPOTIFY-THING/
├── extension/        # Manifest V3 Chrome Companion Extension (TypeScript + React 19 + Vite)
│   ├── manifest.json # Minimal permissions (storage, identity, sidePanel), strict CSP, zero DOM scraping
│   ├── src/          # Popup/Side-Panel UI, Onboarding, Discovery, Shuffle Studio, Insights, Settings, PKCE
│   └── public/icons/ # Extension icon bundle (16/32/48/128 PNG)
├── backend/          # FastAPI Companion Server (/api/v1/* + legacy /api/*) & PyTorch ML/DL Engine
│   ├── app/          # NeuMF, Dual-Horizon Transformer, RRF Hybrid, Adaptive Shuffle, LinUCB, SQLite Stores
│   ├── data_pipeline/# Independent catalog & interaction builders
│   ├── training/     # GPU training & multi-seed shuffle / recommendation evaluation harnesses
│   └── tests/        # Unit, leakage, persistence, ML/DL & /api/v1/* companion API test suites
├── frontend/         # Preserved Next.js Web Research Studio
├── docs/             # Architecture, Official Spotify Policy, PKCE Setup, API v1, Privacy & Install Guides
└── .env.example      # Environment configuration template
```

---

## Core Companion Features

1. **Neural Music Discovery (`Discovery`):**
   - Powered by a **Shared 128D Multimodal Track Encoder**, **Neural Collaborative Filtering (`NeuMF`)** for stable long-term taste, **2-Layer Causal Session Transformer** for immediate mood, **Reciprocal Rank Fusion (`RRF`)**, and **Maximal Marginal Relevance (`MMR`)**.
   - Interactive **Familiarity ↔ Discovery** controller, human-readable recommendation explanations, **128D Similar Tracks & Similar Artists** exploration, one-click **Open in Spotify** handoff, and rich feedback actions (`Like`, `Dislike`, `More like this`, `Less like this`, `Request alternatives`).
2. **Adaptive Shuffle Studio (`Shuffle`):**
   - Sequences curated catalog playlists, AI Discovery mixes, or authorized Spotify playlists (with capability-aware metadata fallback when acoustic features are unavailable).
   - 5 user-centric preset axes (**Artist Repetition**, **Genre Diversity**, **Musical Coherence**, **Discovery**, **Randomness**) with persistent preset saving.
   - Side-by-side **Generated Sequence vs. Original Order** comparison, baseline metrics (`Artist Clump Rate`, `Mean/Min Artist Gap`, `Acoustic Smoothness`, `BPM Jump`), artist/genre/discovery distributions, and multi-format handoff (**Copy Spotify URIs**, **Export M3U/CSV**, **Open in Spotify**, or **Save Private Spotify Playlist** when authorized).
3. **Provenance-Verified Music Insights (`Insights`):**
   - Explicitly separates **Observed** companion interactions, **Inferred Preferences** (`LinUCB` contextual bandit drift & auto-decaying artist/genre fatigue memory), and **Model Predictions** (`ILD` diversity, repeated artist exposure, sonic profile) without fabricating external listening-time history.
4. **Capability-Aware Spotify OAuth 2.0 PKCE & Full Independent Mode (`Settings`):**
   - Works **100% independently** with zero Spotify credentials required.
   - Optional official **Authorization Code with PKCE (`S256`)** integration (`SpotifyProvider`) with zero client secret, minimal scopes, no DOM scraping, and strict compliance with Spotify's prohibition on training ML/AI models on Spotify Content.
   - Built-in **Disconnect Spotify**, **Reset Bandit Memory**, and **Delete Stored Application Data** privacy controls.

---

## Empirical ML & Adaptive Shuffle Benchmarks

Evaluated on an independent catalog of **40,000 tracks** (8,146 artists, 114 genres), **6,932 playlists**, and **327,754 timestamped interactions** across **943 multi-session listeners**:

| Model / Algorithm | Key Metric Highlights |
| :--- | :--- |
| **Multimodal NeuMF (`GMF + MLP`)** | `HR@10 = 76.00%` · `NDCG@10 = 0.6059` |
| **Dual-Horizon Session Transformer** | `HR@10 = 77.50%` · `NDCG@10 = 0.6289` |
| **Hybrid RRF + Discovery (`MMR`)** | **`HR@10 = 79.67%`** · **`NDCG@10 = 0.6631`** · `Coverage@20 = 13.19%` |
| **Adaptive Shuffle (200 Playlists)** | Cuts artist clump rate from `11.40%` (Random) to **`7.41%`** (`5.05%` with quota urgency) while improving Genre Switch (`82.2%`), Acoustic Smoothness (`71.9%`), and Session `NDCG` (`0.525`). |

---

## Quick Start (Local Backend + Chrome Extension)

### 1. Start the FastAPI & PyTorch Backend (`http://127.0.0.1:8000`)
```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
backend\.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
```bash
# macOS / Linux
PYTHONPATH=. backend/.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Test & Build the Chrome Extension (`extension/dist/`)
```bash
cd extension
npm install
npm test
npm run build
```

### 3. Load Unpacked in Google Chrome
1. Open `chrome://extensions` and enable **Developer mode**.
2. Click **Load unpacked** and select `SPOTIFY-THING/extension/dist`.
3. Pin **SPOTIFY-THING** to the toolbar and click the icon to launch the companion popup.

---

## Documentation Index

- [Extension Architecture](docs/EXTENSION_ARCHITECTURE.md)
- [Official Spotify Integration & Policy Compliance](docs/SPOTIFY_INTEGRATION.md)
- [Optional Spotify OAuth 2.0 (PKCE) Setup](docs/OAUTH_SETUP.md)
- [Backend `/api/v1/*` API Reference](docs/API_INTEGRATION.md)
- [Privacy & Security Model](docs/PRIVACY.md)
- [Installation & Local Development Guide](docs/INSTALLATION.md)
- [Adaptive Shuffle Multi-Seed Experiment Report](docs/shuffle_experiment_report.md)

---

## License Scope

Original source code and documentation are released under the [MIT License](LICENSE). External research datasets downloaded locally for model training and Spotify platform services/trademarks remain subject to their respective upstream terms.
