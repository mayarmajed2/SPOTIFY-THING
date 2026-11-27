# FastAPI Companion Backend `/api/v1/*` Reference

All Companion Extension routes are mounted under `/api/v1` in [v1.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/app/api/v1.py), while legacy `/api/*` endpoints in [main.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/app/main.py) remain intact for the Next.js web application.

## Authentication & Headers

- Public endpoints: `GET /api/v1/health`, `POST /api/v1/auth/session`
- Authenticated endpoints require either:
  - `X-Extension-Token: st_ext_<random>`
  - `Authorization: Bearer st_ext_<random>`
- Raw session tokens are **never** stored in SQLite; only `SHA-256` digests are stored in `extension_users.token_hash`.

---

## Endpoint Summary

| Method & Path | Purpose |
| :--- | :--- |
| `GET /api/v1/health` | Reports API version, device (`cuda`/`cpu`), catalog dimensions, and loaded PyTorch model flags. |
| `POST /api/v1/auth/session` | Creates an isolated extension user profile (`feedback_slot_idx >= 1,000,000`) and returns `session_token` once. |
| `GET /api/v1/auth/session` | Validates current session token and returns user preferences & Spotify capability state. |
| `POST /api/v1/auth/logout` | Revokes session token and clears server-side state for that session. |
| `GET /api/v1/users/me` | Returns user profile, preferences, Spotify connection state, and bandit feedback summary. |
| `DELETE /api/v1/users/me` | Permanently deletes user session, preferences, Spotify status metadata, and SQLite bandit/fatigue history. |
| `GET /api/v1/preferences` | Retrieves onboarding status, discovery/diversity settings, shuffle presets, and available catalog genres. |
| `PUT /api/v1/preferences` | Updates user preferences and recalibrates warm-start taste anchor when genres/artists change. |
| `POST /api/v1/discovery` | Runs `NeuMF + Dual-Horizon Transformer + RRF Item-CF + LinUCB + MMR` to return tracks, discoveries, artists, genres, and explanations. |
| `POST /api/v1/recommendations` | Returns ranked track recommendations and familiarity/discovery split. |
| `GET /api/v1/discovery/similar-tracks/{item_idx}` | Finds nearest-neighbor tracks in shared 128D multimodal embedding space. |
| `GET /api/v1/discovery/similar-artists?artist=...` | Finds nearest-neighbor artists using 128D L2-normalized artist prototype embeddings. |
| `GET /api/v1/discovery/catalog-playlists` | Lists multi-artist curated playlists for Independent Mode Adaptive Shuffle. |
| `GET /api/v1/discovery/search?q=...` | Searches the 40,000-track independent catalog by title, artist, or genre. |
| `POST /api/v1/shuffle` | Generates an Adaptive Shuffle sequence (supporting catalog playlists, discovery mixes, and out-of-catalog fallback tracks) with baseline comparisons, distributions, and URI/M3U/CSV exports. |
| `PUT /api/v1/shuffle/preferences` | Persists preferred 5-axis Adaptive Shuffle Studio presets. |
| `POST /api/v1/feedback` | Records `like`, `dislike`, `not_interested`, `more_like_this`, `less_like_this`, `artist_fatigue`, `genre_fatigue`, or implicit playback feedback into `LinUCB` + `Fatigue Memory`. |
| `GET /api/v1/feedback` | Returns feedback summary and recent recorded events. |
| `POST /api/v1/feedback/reset` | Resets `LinUCB` matrices, fatigue counters, and dismissed tracks for the current user. |
| `GET /api/v1/insights` | Returns strictly provenance-labeled `observed`, `inferred_preferences`, and `model_predictions` analytics. |
| `GET / PUT /api/v1/spotify/status` | Reads/updates non-sensitive Spotify capability flags (never OAuth tokens). |
| `POST /api/v1/spotify/disconnect` | Resets stored Spotify connection state to Independent Mode. |
| `POST /api/v1/spotify/resolve-tracks` | Transiently maps external track metadata to catalog indices for read-only inference. |
