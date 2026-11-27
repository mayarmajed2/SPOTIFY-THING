/**
 * Manifest V3 Background Service Worker for SPOTIFY-THING Companion Extension.
 *
 * Responsibilities:
 * - Initializes default companion configuration on extension install without invasive permissions.
 * - Handles background token refresh checks and backend health pings via message passing.
 * - Opens Side Panel or Companion Tab cleanly when requested by popup UI.
 * - Never injects scripts into Spotify pages or intercepts Spotify network traffic.
 */

import { getValidSpotifyAccessToken } from "../services/spotify-auth";
import { extensionStorage } from "../services/storage";

const DEFAULT_API_URL = "http://127.0.0.1:8000";

if (typeof chrome !== "undefined" && chrome?.runtime?.onInstalled) {
  chrome.runtime.onInstalled.addListener(async (details) => {
    const savedApiUrl = await extensionStorage.getLocal<string>("st_companion_api_base_url");
    if (!savedApiUrl) {
      await extensionStorage.setLocal("st_companion_api_base_url", DEFAULT_API_URL);
    }
    if (details.reason === "install" && chrome.sidePanel?.setPanelBehavior) {
      try {
        await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false });
      } catch {
        // Ignore if browser version does not support setPanelBehavior
      }
    }
  });
}

export async function handleServiceWorkerMessage(
  message: { type?: string; payload?: Record<string, unknown> },
): Promise<Record<string, unknown>> {
  switch (message?.type) {
    case "PING_BACKEND_HEALTH": {
      const baseUrl =
        (await extensionStorage.getLocal<string>("st_companion_api_base_url")) || DEFAULT_API_URL;
      try {
        const resp = await fetch(`${baseUrl.replace(/\/+$/, "")}/api/v1/health`);
        if (!resp.ok) {
          return { ok: false, status: resp.status };
        }
        const data = (await resp.json()) as Record<string, unknown>;
        return { ok: true, health: data };
      } catch {
        return { ok: false, error: "BACKEND_UNREACHABLE" };
      }
    }

    case "CHECK_SPOTIFY_TOKEN_VALIDITY": {
      const bundle = await getValidSpotifyAccessToken();
      return {
        authenticated: Boolean(bundle),
        expiresAt: bundle?.expires_at ?? null,
      };
    }

    case "OPEN_COMPANION_TAB": {
      if (typeof chrome !== "undefined" && chrome?.tabs?.create && chrome?.runtime?.getURL) {
        const url = chrome.runtime.getURL("popup.html?view=expanded");
        await chrome.tabs.create({ url });
        return { opened: true, url };
      }
      return { opened: false };
    }

    default:
      return { ok: false, error: "UNKNOWN_MESSAGE_TYPE" };
  }
}

if (typeof chrome !== "undefined" && chrome?.runtime?.onMessage) {
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    handleServiceWorkerMessage(message)
      .then((res) => sendResponse(res))
      .catch(() => sendResponse({ ok: false, error: "WORKER_ERROR" }));
    return true;
  });
}
