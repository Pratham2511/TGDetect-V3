import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  devIndicators: false,
  turbopack: { root: process.cwd() },
  experimental: { proxyClientMaxBodySize: "50mb" },
  async rewrites() {
    return {
      beforeFiles: [{ source: "/", destination: "/landing/index.html" }],
      afterFiles: [
        {
          source: "/api/:path*",
          destination: `${process.env.TGDETECT_API_ORIGIN || "http://127.0.0.1:8000"}/api/:path*`,
        },
      ],
      fallback: [],
    };
  },
};
export default nextConfig;
