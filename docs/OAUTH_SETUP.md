# Optional Spotify OAuth 2.0 (PKCE) Setup Guide

SPOTIFY-THING works out-of-the-box in **Independent Companion Mode** with zero API keys. Follow this guide only if you wish to connect your own Spotify Developer application to import owned playlists or export Adaptive Shuffle sequences as private playlists.

---

## Step 1 — Load the Extension and Copy Your Redirect URI

1. Build the extension (`cd extension && npm run build`) and load `extension/dist` unpacked in `chrome://extensions`.
2. Open the **SPOTIFY-THING** popup, complete onboarding in **Independent Mode**, and open the **Settings** tab.
3. Under **Official Spotify OAuth 2.0 (PKCE)**, click **Copy** next to the generated Redirect URI:
   ```text
   https://<your-32-char-extension-id>.chromiumapp.org/spotify-callback
   ```

---

## Step 2 — Create a Spotify Developer App

1. Visit the official [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and sign in (note: Spotify requires an active **Spotify Premium** subscription on the developer account for Development Mode apps).
2. Click **Create app**:
   - **App name:** `SPOTIFY-THING Local Companion`
   - **Redirect URIs:** Paste the exact `https://<extension-id>.chromiumapp.org/spotify-callback` URL from Step 1 and click **Add**.
   - **APIs used:** Select **Web API**.
3. Save the app and open **User Management** in the dashboard to confirm your Spotify account email is listed (Development Mode allows up to 5 allowlisted users).
4. Copy the public **Client ID** from the app's Settings page.
   > **Security Rule:** Do **NOT** copy or paste your `Client Secret` anywhere. SPOTIFY-THING uses RFC 7636 Authorization Code with PKCE (`S256`), which authenticates securely without a client secret.

---

## Step 3 — Connect Inside SPOTIFY-THING

1. Open **SPOTIFY-THING → Settings**.
2. Paste your **Client ID** into **Spotify Developer Client ID**.
3. Review the requested minimal scopes:
   - `user-read-private` & `playlist-read-private` (read your profile and owned/collaborative playlists).
   - `user-read-playback-state` *(optional)* — read active player context.
   - `playlist-modify-private` *(optional)* — save generated Adaptive Shuffle sequences as new private playlists.
4. Click **Connect Spotify (PKCE)** and approve the official Spotify authorization prompt.
5. To revoke access at any time, click **Disconnect** in **Settings** (which wipes local PKCE tokens immediately) and optionally remove the app on your [Spotify Account Apps page](https://www.spotify.com/account/apps/).
