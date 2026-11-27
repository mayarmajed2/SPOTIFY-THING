# Official Spotify Platform Integration & Policy Compliance

## 1. Guiding Compliance Principles

SPOTIFY-THING strictly adheres to the **Spotify Developer Terms** and **Spotify Developer Policy**:

1. **Zero AI/ML Training on Spotify Content:** Spotify Developer Policy explicitly prohibits using the Spotify Platform or Spotify Content to train or ingest into machine learning or AI models. SPOTIFY-THING's neural models (`MultimodalTrackEncoder`, `NeuMF`, `DualHorizonSessionTransformer`) are pre-trained solely on independent datasets. Any authorized playlist metadata retrieved via OAuth is used transiently for sequence ordering and UI display and is **never** persisted into training datasets or used to update model weights.
2. **Zero Scraping or Private Endpoints:** The extension contains no content scripts on `*.spotify.com`, never reads the Spotify Web Player DOM, never intercepts browser network calls, and never calls undocumented internal endpoints.
3. **Full Independent Mode:** Every core capability (Discovery, Adaptive Shuffle Studio, Music Insights, Feedback Personalization, URI/M3U/CSV export, and `https://open.spotify.com/track/{id}` playback handoff) works without Spotify authentication.

---

## 2. Current Spotify Web API Capabilities & Restrictions (2025–2026)

| Capability | Endpoint / Mechanism | Status & Handling in `SpotifyProvider` |
| :--- | :--- | :--- |
| **OAuth Login** | `Authorization Code + PKCE (S256)` | Supported via `chrome.identity.launchWebAuthFlow`. Zero client secret. |
| **Current User Profile** | `GET /v1/me` (`user-read-private`) | Supported for allowlisted users (Dev Mode) or Extended Quota apps. Catches `403` gracefully. |
| **List User Playlists** | `GET /v1/me/playlists` (`playlist-read-private`) | Filtered strictly to playlists **owned or collaborated on** by the authenticated user. |
| **Read Playlist Items** | `GET /v1/playlists/{id}/tracks` | Restricted by Spotify to user-owned or collaborative playlists (`403` otherwise). |
| **Create Private Playlist** | `POST /v1/me/playlists` + `POST /v1/playlists/{id}/items` | Uses current `/me/playlists` endpoint (replacing removed `/users/{id}/playlists`) and `/items` (with `/tracks` fallback). Requested on-demand via `playlist-modify-private`. |
| **Playback Context** | `GET /v1/me/player` (`user-read-playback-state`) | Optional opt-in read-only check; handles `204 No Content` and `403` cleanly. |
| **Audio Features / Analysis** | `GET /v1/audio-features`, `/v1/audio-analysis` | **Restricted (`403`)** for Dev Mode / new apps — **never called** by SPOTIFY-THING. |
| **Spotify Recommendations** | `GET /v1/recommendations`, `/related-artists` | **Restricted (`403`)** — **never called**; all recommendations come from our independent PyTorch engine. |
| **Native Shuffle / Queue Reorder** | N/A | **Not supported** by official Spotify Web API. Clearly disclosed in UI; sequences are exported via URIs, M3U, CSV, deep links, or optional private playlist creation. |

---

## 3. Development Mode Operational Notes

Under current Spotify Developer Dashboard rules for **Development Mode** applications:
- The **app owner** must hold an active **Spotify Premium** subscription.
- Up to **5 test users** must be explicitly added to the app's **User Management** allowlist in the Spotify Developer Dashboard; un-allowlisted accounts receive `403 Forbidden` from `/v1/me`.
- Whenever `SpotifyProvider` encounters `401`, `403`, or `429`, it surfaces a clear human-readable explanation and keeps **Independent Mode** active without data loss or crashes.
