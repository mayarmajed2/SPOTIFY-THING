# Dataset License & Rights Audit (`docs/DATASET_LICENSE_AUDIT.md`)

**Review Date:** 2026-11-27  
**Repository Code License:** MIT License ([LICENSE](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/LICENSE)) — **applies strictly to original source code and documentation only.**

---

## 1. Classification Matrix

| Dataset / Asset | Upstream Source | License / Governing Terms | Research Training? | Public Redistribution in Git? | Commercial Use Cleared? | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Last.fm 1K** (`matthewfranglen/lastfm-1k`) | Derived from Oscar Celma / MTG-UPF Last.fm 1K | Non-commercial academic/research terms | Yes (research only) | **No** (excluded via `.gitignore`) | **No** | Anonymized user scrobbles (2005–2009); no commercial clearance. |
| **Spotify Million Playlist Dataset** (`jaxliu/Spotify_Million_Playlist_Dataset_Challenge`) | Originally Spotify RecSys Challenge 2018 / AIcrowd | Non-commercial academic/research RecSys Challenge license | Research challenge scope only | **No** (excluded via `.gitignore`) | **No** | Spotify-owned playlist metadata; strictly non-commercial. |
| **Spotify Tracks Audio Features** (`maharshipandya/spotify-tracks-dataset`) | Community-collected Spotify Web API snapshot on Hugging Face / Kaggle | Wrapper labeled BSD on HF, **but underlying fields originate from Spotify Web API** | Restricted under Spotify Developer Terms | **No** (excluded via `.gitignore`) | **No** | Spotify Developer Terms Section IV prohibit training ML/AI models on Spotify Content. See `docs/SPOTIFY_COMPLIANCE.md`. |
| **Spotify Embed `<iframe>`** | `https://open.spotify.com/embed/track/{id}` | Spotify Developer Terms & Embed Policy | N/A (UI playback only) | N/A (runtime iframe) | Subject to Spotify Terms | Used purely for optional client-side preview playback; no audio scraping. |
| **SPOTIFY-THING Source Code & Docs** | Original repository code | **MIT License** | Yes | Yes | Yes (code only) | Architecture, PyTorch modules, Adaptive Shuffle, FastAPI, Next.js UI. |

---

## 2. Key Compliance Controls Enforced in Repository
1. **Zero Restricted Data in Git:** `data/raw/`, `data/processed/`, `data/processed_v2/`, and binary model checkpoints (`.pt`, `.npy`, `.pkl`, `.db`) are excluded via `.gitignore`.
2. **Modular, Data-Source-Agnostic Engine:** The recommendation models (`MultimodalTrackEncoder`, `NeuMF`, `DualHorizonSessionTransformer`) and `AdaptiveShuffleEngine` operate on generic integer IDs and 12D continuous audio vectors — they can be trained on **commercially cleared or open-license catalogs** (e.g., **FMA — Free Music Archive**, **MTG-Jamendo**, or proprietary partner catalogs) without depending on Spotify Web API data.
