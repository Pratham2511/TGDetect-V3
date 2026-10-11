import type { Metadata } from "next";
import localFont from "next/font/local";

import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/lib/theme-context";
import { DEFAULT_THEME, THEME_STORAGE_KEY } from "@/lib/theme";

const instrumentSans = localFont({
  src: [
    { path: "./fonts/InstrumentSans.ttf", weight: "400 700", style: "normal" },
    { path: "./fonts/InstrumentSans-Italic.ttf", weight: "400 700", style: "italic" },
  ],
  variable: "--font-instrument",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TGDetect · Investigation workspace",
  description:
    "Temporal graph investigation with traceable alert evidence, checkpoint verification and candidate attack-chain reconstruction.",
  keywords: [
    "TGDetect",
    "Temporal Graph",
    "Graph Neural Network",
    "GAT",
    "Node memory",
    "TGNN",
    "Threat Detection",
    "Attack Chain",
    "MITRE ATT&CK",
  ],
  authors: [{ name: "TGDetect" }],
  icons: { icon: "/logo.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${instrumentSans.variable} ${DEFAULT_THEME}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `(function(){var t=${JSON.stringify(DEFAULT_THEME)};try{var q=new URLSearchParams(location.search).get('theme');var s=q==='dark'||q==='light'?q:localStorage.getItem(${JSON.stringify(THEME_STORAGE_KEY)});if(s==='dark'||s==='light')t=s}catch(e){}document.documentElement.classList.toggle('dark',t==='dark');document.documentElement.classList.toggle('light',t==='light')})();` }} />
      </head>
      <body
        className={`antialiased bg-background text-foreground`}
      >
        <ThemeProvider>{children}</ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
