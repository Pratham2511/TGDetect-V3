"use client";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "@/lib/theme-context";
import styles from "./console.module.css";

export function ThemeControl() {
  const { theme, setTheme } = useTheme();
  return (
    <div className={styles.themeControl} role="group" aria-label="Color theme">
      <button type="button" aria-label="Solarized light theme" aria-pressed={theme === "light"} onClick={() => setTheme("light")}><Sun size={15} /><span>Light</span></button>
      <button type="button" aria-label="Black dark theme" aria-pressed={theme === "dark"} onClick={() => setTheme("dark")}><Moon size={15} /><span>Dark</span></button>
    </div>
  );
}
