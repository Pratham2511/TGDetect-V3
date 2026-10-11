"use client";

import { createContext, useContext, useEffect, useSyncExternalStore, ReactNode } from "react";
import { DEFAULT_THEME, THEME_STORAGE_KEY, resolveTheme, type Theme } from "./theme";

export type { Theme } from "./theme";
const key = THEME_STORAGE_KEY;
const event = "tgdetect-theme-change";
let sessionTheme: Theme | null = null;

function getSnapshot(): Theme {
  if (sessionTheme) return sessionTheme;
  const query = new URLSearchParams(window.location.search).get("theme");
  if (query === "light" || query === "dark") return query;
  try { return resolveTheme(localStorage.getItem(key)); }
  catch { return DEFAULT_THEME; }
}
function subscribe(callback: () => void) {
  const onStorage = () => { sessionTheme = null; callback(); };
  window.addEventListener("storage", onStorage);
  window.addEventListener(event, callback);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(event, callback);
  };
}
function setTheme(theme: Theme) {
  sessionTheme = theme;
  try { localStorage.setItem(key, theme); } catch { /* Retain the choice for this session. */ }
  window.dispatchEvent(new Event(event));
}
const ThemeContext = createContext({ theme: DEFAULT_THEME, setTheme, toggleTheme: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useSyncExternalStore(subscribe, getSnapshot, () => DEFAULT_THEME);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.classList.toggle("light", theme === "light");
  }, [theme]);
  return <ThemeContext.Provider value={{ theme, setTheme, toggleTheme: () => setTheme(theme === "dark" ? "light" : "dark") }}>{children}</ThemeContext.Provider>;
}
export const useTheme = () => useContext(ThemeContext);
