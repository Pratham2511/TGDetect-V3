import assert from 'node:assert/strict';

const origin = new URL(process.env.TGDETECT_APP_URL || 'http://127.0.0.1:3000');
async function request(path) {
  const response = await fetch(new URL(path, origin), { signal: AbortSignal.timeout(15000) });
  assert.equal(response.status, 200, `${path} must return HTTP 200`);
  return response;
}

const landing = await (await request('/')).text();
assert.match(landing, /Threats leave/);
assert.doesNotMatch(landing, /\/_next\/static/, 'Landing must not load the dashboard runtime');
const workspaceLinks = [...landing.matchAll(/<a\b[^>]*href="([^"]+)"[^>]*data-workspace-link/g)];
assert.ok(workspaceLinks.length >= 4, 'Expected the landing workspace entry points');
for (const [, href] of workspaceLinks) {
  assert.equal(new URL(href, origin).pathname, '/dashboard');
  assert.equal(new URL(href, origin).origin, origin.origin, 'Workspace entry must use the current host');
}
const assets = new Set([...landing.matchAll(/(?:src|href)="(\/landing\/[^"#]+)"/g)].map((match) => match[1]));
await Promise.all([...assets].map((path) => request(path)));
const model = await (await request('/landing/assets/trace-engine.glb')).arrayBuffer();
assert.equal(new DataView(model).getUint32(0, true), 0x46546c67, 'Model must be a binary glTF asset');
assert.equal(new DataView(model).getUint32(8, true), model.byteLength, 'Model must not be truncated');
const dashboard = await (await request('/dashboard')).text();
assert.match(dashboard, /TGDetect/);
assert.match(dashboard, /\/_next\/static/, 'Dashboard must render the Next.js application');
assert.doesNotMatch(dashboard, /Threats leave/, 'Dashboard must not resolve to the landing rewrite');
const proxy = await request('/api/cadets/summary');
assert.match(proxy.headers.get('content-type') || '', /application\/json/);
const summary = await proxy.json();
assert.ok(Object.keys(summary).length > 0, 'The backend summary must reach the dashboard origin');
console.log(`Passed landing, ${assets.size} static assets, complete glTF, ${workspaceLinks.length} same-origin CTAs, dashboard, and live API proxy at ${origin.origin}.`);
