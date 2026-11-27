/**
 * Chrome Extension Storage Abstraction.
 * Uses `chrome.storage.local` / `chrome.storage.session` in MV3 Extension runtime,
 * and falls back cleanly to Web Storage in Vitest / dev preview environments.
 */

function hasChromeLocalStorage(): boolean {
  return typeof chrome !== "undefined" && Boolean(chrome?.storage?.local);
}

function hasChromeSessionStorage(): boolean {
  return typeof chrome !== "undefined" && Boolean(chrome?.storage?.session);
}

export const extensionStorage = {
  async getLocal<T>(key: string): Promise<T | null> {
    if (hasChromeLocalStorage()) {
      const res = await chrome.storage.local.get(key);
      return (res[key] as T) ?? null;
    }
    try {
      const raw = globalThis.localStorage?.getItem(key);
      return raw ? (JSON.parse(raw) as T) : null;
    } catch {
      return null;
    }
  },

  async setLocal<T>(key: string, value: T): Promise<void> {
    if (hasChromeLocalStorage()) {
      await chrome.storage.local.set({ [key]: value });
      return;
    }
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  },

  async removeLocal(key: string): Promise<void> {
    if (hasChromeLocalStorage()) {
      await chrome.storage.local.remove(key);
      return;
    }
    globalThis.localStorage?.removeItem(key);
  },

  async getSession<T>(key: string): Promise<T | null> {
    if (hasChromeSessionStorage()) {
      const res = await chrome.storage.session.get(key);
      return (res[key] as T) ?? null;
    }
    try {
      const raw = globalThis.sessionStorage?.getItem(key);
      return raw ? (JSON.parse(raw) as T) : null;
    } catch {
      return null;
    }
  },

  async setSession<T>(key: string, value: T): Promise<void> {
    if (hasChromeSessionStorage()) {
      await chrome.storage.session.set({ [key]: value });
      return;
    }
    globalThis.sessionStorage?.setItem(key, JSON.stringify(value));
  },

  async removeSession(key: string): Promise<void> {
    if (hasChromeSessionStorage()) {
      await chrome.storage.session.remove(key);
      return;
    }
    globalThis.sessionStorage?.removeItem(key);
  },

  async clearAllAppStorage(): Promise<void> {
    if (hasChromeLocalStorage()) {
      await chrome.storage.local.clear();
    } else {
      globalThis.localStorage?.clear();
    }
    if (hasChromeSessionStorage()) {
      await chrome.storage.session.clear();
    } else {
      globalThis.sessionStorage?.clear();
    }
  },
};
