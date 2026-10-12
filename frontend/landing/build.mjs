import { build } from 'esbuild';
import { cp, mkdir, rename, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = dirname(fileURLToPath(import.meta.url));
const output = resolve(source, '../public/landing');
const staging = resolve(source, '../public/.landing-build');

// Build into a separate directory before replacing the generated files.
await rm(staging, { recursive: true, force: true });
await mkdir(staging, { recursive: true });
try {
  await Promise.all([
    cp(resolve(source, 'index.html'), resolve(staging, 'index.html')),
    cp(resolve(source, 'styles.css'), resolve(staging, 'styles.css')),
    cp(resolve(source, 'assets'), resolve(staging, 'assets'), { recursive: true }),
  ]);
  await build({
    absWorkingDir: source,
    entryPoints: ['src/app.js'],
    outdir: resolve(staging, 'dist'),
    bundle: true,
    splitting: true,
    format: 'esm',
    minify: true,
    target: ['chrome110', 'firefox115', 'safari16.4', 'edge110'],
    legalComments: 'linked',
  });
  await rm(output, { recursive: true, force: true });
  await rename(staging, output);
  console.log('Built the landing page at public/landing; workspace links point to /dashboard.');
} catch (error) {
  await rm(staging, { recursive: true, force: true });
  throw error;
}
