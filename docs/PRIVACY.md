# Privacy & Security Model

## 1. Minimal Browser Permissions

SPOTIFY-THING's Manifest V3 configuration ([manifest.json](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/extension/manifest.json)) requests only three Chrome API permissions:
- `storage`: Stores your local companion session token, preferred API URL, and (if connected) PKCE token bundle locally in Chrome.
- `identity`: Launches the official `chrome.identity.launchWebAuthFlow` window only when you explicitly initiate Spotify PKCE login.
- `sidePanel`: Allows opening the companion interface in Chrome's side panel.

**Permissions Intentionally Excluded:**
- No `tabs`, `scripting`, `activeTab`, `cookies`, or `webRequest` permissions.
- Zero `content_scripts` — SPOTIFY-THING never reads, modifies, or scrapes `open.spotify.com` or any other website you visit.

---

## 2. Token & Secret Protection

- **No Client Secret:** Public OAuth 2.0 Authorization Code with PKCE (`S256`) is used exclusively.
- **Spotify OAuth Tokens Stay in Chrome:** Spotify `access_token` and `refresh_token` values remain inside `chrome.storage.session` / `chrome.storage.local` in the browser and are **never** transmitted to the SPOTIFY-THING FastAPI backend.
- **Hashed Backend Session Tokens:** Your SPOTIFY-THING backend session token (`st_ext_...`) is hashed with `SHA-256` before storage in SQLite (`extension_state.db`), so raw bearer tokens are never written to disk.
- **Log & Error Redaction:** Both the extension API client and FastAPI exception handlers strip `Bearer ...` and `st_ext_...` patterns from error messages.

---

## 3. Data Separation & Non-Training Guarantee

SPOTIFY-THING maintains strict boundaries across data categories:
1. **Observed Companion Activity:** Only explicit feedback clicks (`Like`, `Dislike`, `More/Less like this`, `Not interested`, temporary fatigue) and preferences you submit inside SPOTIFY-THING are stored in your isolated user profile. We never fabricate or scrape external listening-time history.
2. **Authorized Spotify Metadata:** If you connect Spotify to sequence an owned playlist, track titles/IDs are resolved transiently in memory to compute the shuffle order and are **never** saved to training datasets or used to train ML/AI models.
3. **User Isolation:** Each extension session maps to a dedicated `feedback_slot_idx` in SQLite WAL storage, preventing cross-user leakage.

---

## 4. User Data Deletion & Disconnect Controls

At any time in **SPOTIFY-THING → Settings**:
- **Disconnect Spotify:** Immediately deletes locally cached Spotify PKCE tokens and resets the profile to Independent Mode.
- **Reset Bandit Feedback & Fatigue Memory:** Wipes your `LinUCB` adaptation state, liked/disliked item sets, and temporary artist/genre fatigue counters (`POST /api/v1/feedback/reset`).
- **Delete Stored Preferences & Application Data:** Permanently deletes your user record, hashed session token, preferences, and feedback state from the backend SQLite databases (`DELETE /api/v1/users/me`) and clears local Chrome Extension storage.
