export type Theme = "dark" | "light";

export const DEFAULT_THEME: Theme = "dark";
export const THEME_STORAGE_KEY = "tgdetect-theme";

export function resolveTheme(value: unknown): Theme {
  return value === "light" || value === "dark" ? value : DEFAULT_THEME;
}
