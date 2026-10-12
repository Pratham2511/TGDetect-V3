# TGDetect landing page

The approved Trace Engine landing page is served at `/` by the same Next.js process as the dashboard at `/dashboard`. Every workspace CTA uses `/dashboard`, so it works with the current host and port. The dashboard logo returns to `/`.

## Build and edit

From `frontend/`, run `npm ci`, then `npm run dev` or `npm run build && npm start`. Both development and production builds generate `public/landing/` automatically; no second server or package installation is needed. Generated files are excluded from Git.

Edit `index.html`, `styles.css`, and `src/`. Run `npm run build:landing` after changing landing files while the development server is running. Static assets live in `assets/`; Three.js, Anime.js, and Meshopt are bundled from the pinned frontend lockfile. There are no runtime CDN dependencies.

Next.js rewrites `/` to the generated HTML without changing the address. Serving it as a separate HTML document keeps its styles and animation lifecycle independent of the React dashboard. Navigating to the dashboard releases the landing scene and loads the existing application. The same-origin `/api/` proxy and `/legacy` route are retained.

With both services running, `npm run check:integration` verifies the landing response and assets, complete glTF delivery, same-origin workspace CTAs, dashboard route, and live API proxy. Set `TGDETECT_APP_URL` to check a different local origin.

## Model and compatibility

`assets/trace-engine.glb` is the actual approved Blender MCP export: 12 rigid part groups, 66 meshes, and the authored assembly clip. It is approximately 2.2 MB with embedded machining textures and Meshopt compression. No geometry decimation, scale animation, or browser-generated replacement model is used. The editable native Blender scene remains in the separate local design workspace.

Five chapters have pauses before and after each part movement. Downward scrolling assembles the machine; upward scrolling reverses the same transforms. Motion off, system reduced motion, data-saving preferences, and unavailable hardware graphics use genuine Blender stills. Rendering pauses outside the visible section and when the tab is hidden, with capped pixel density and an adaptive 30/24 FPS limit.

The bundle targets Chrome/Edge 110+, Firefox 115+, and Safari 16.4+. Chromium desktop and mobile layouts were directly checked, including the user's Intel UHD 620 integrated graphics. Other browser engines have not been directly tested.

See [third-party notices](THIRD-PARTY-NOTICES.md) and the local Instrument Sans OFL in `assets/`.
