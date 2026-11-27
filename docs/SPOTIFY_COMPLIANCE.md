# Spotify Developer Platform Compliance Assessment (`docs/SPOTIFY_COMPLIANCE.md`)

**Review Date:** 2026-11-27  
**Official References Reviewed:**
- Spotify Developer Terms of Service: https://developer.spotify.com/terms
- Spotify Developer Policy: https://developer.spotify.com/policy
- Spotify Web API Changes Announcement (Nov 27, 2024): https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api

---

## 1. Critical Platform Policy Constraints

1. **Prohibition on ML/AI Training with Spotify Content (Developer Terms, Section IV — Restrictions):**
   - Spotify explicitly forbids developers from using the Spotify Platform or any Spotify Content (including Web API metadata and audio attributes) to train or ingest into machine learning or AI models.
   - **Spotify Premium does NOT grant API or ML training rights.**
2. **Deprecation of Core Audio & Discovery Endpoints (Effective Nov 27, 2024):**
   - For all new and Development-Mode applications, Spotify restricted access (`403 Forbidden`) to:
     - `GET /v1/audio-features`
     - `GET /v1/audio-analysis`
     - `GET /v1/recommendations`
     - `GET /v1/artists/{id}/related-artists`
   - Therefore, any independent commercial product **cannot** depend on live Spotify Web API audio features.
3. **Spotify Embed Policy:**
   - The official `https://open.spotify.com/embed/track/{id}` iframe in `PlayerDock.tsx` is permitted for interactive preview playback subject to Spotify branding/embed terms, and must never be used to scrape or capture audio streams.

---

## 2. Architectural Path to Full Commercial & Proposal Readiness

Because SPOTIFY-THING has a long-term ambition of being presented as a **technical proposal to Spotify** (or operating independently on open catalogs):

1. **For Independent / Open-Source Deployment:** Swap the third-party Spotify audio CSV for **open-licensed audio feature extraction** computed directly via DSP/open models (e.g., `librosa` / `Essentia` on **MTG-Jamendo** or **Free Music Archive (FMA)** tracks) and open interaction benchmarks (e.g., **Yandex Yambda** or **ListenBrainz** CC0 scrobbles).
2. **For an Internal Spotify Technical Proposal:** Present the **source architecture, zero-leakage evaluation framework, Dual-Horizon Transformer, LinUCB online feedback loop, and Constrained Probabilistic Adaptive Shuffle** as a **bring-your-own-catalog engine** designed to run strictly inside Spotify's first-party infrastructure on Spotify's own authorized internal datasets.
