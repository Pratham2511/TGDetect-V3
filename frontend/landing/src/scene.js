import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { MeshoptDecoder } from 'meshoptimizer';

export async function createScene(holder, visual) {
  const renderer = new THREE.WebGLRenderer({ alpha: false, antialias: true, powerPreference: 'low-power' });
  renderer.setClearColor(0x000000, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.transmissionResolutionScale = .5;
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, innerWidth < 761 ? 1.25 : 1.5));
  holder.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('aria-hidden', 'true');
  const gl = renderer.getContext();
  const graphicsInfo = gl.getExtension('WEBGL_debug_renderer_info');
  visual.dataset.graphics = graphicsInfo ? String(gl.getParameter(graphicsInfo.UNMASKED_RENDERER_WEBGL)) : 'WebGL2';
  if (/swiftshader|llvmpipe|softpipe|software rasterizer/i.test(visual.dataset.graphics)) {
    renderer.dispose(); renderer.domElement.remove();
    throw new Error('Hardware graphics are unavailable; using the Blender still preview.');
  }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, .1, 500);
  const presentation = new THREE.Group();
  scene.add(presentation);
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  let gltf, hdr, env;
  try {
    [gltf, hdr] = await Promise.all([loader.loadAsync('/landing/assets/trace-engine.glb'), new HDRLoader().loadAsync('/landing/assets/studio.hdr')]);
    const pmrem = new THREE.PMREMGenerator(renderer);
    env = pmrem.fromEquirectangular(hdr);
    scene.environment = env.texture;
    scene.environmentIntensity = .8;
    hdr.dispose(); pmrem.dispose();
  } catch (error) { renderer.dispose(); renderer.domElement.remove(); hdr?.dispose(); throw error; }
  presentation.add(gltf.scene);
  gltf.scene.traverse((object) => {
    if (!object.isMesh) return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      if (!material.name.includes('Inspection glass')) return;
      // The thin inspection pane needs reflections, not a second full-scene refraction pass.
      // Alpha glass avoids refraction sampling artifacts on integrated graphics.
      material.transmission = 0;
      material.transparent = true;
      material.opacity = .12;
      material.depthWrite = false;
      material.roughness = .06;
      material.clearcoat = 1;
      material.clearcoatRoughness = .06;
      material.needsUpdate = true;
    });
  });
  const mixer = new THREE.AnimationMixer(gltf.scene);
  gltf.animations.forEach((clip) => { const action = mixer.clipAction(clip); action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true; action.play(); });
  const light = new THREE.DirectionalLight(0xfff5ec, 1.2);
  light.position.set(6, 7, 7); scene.add(light);
  const fill = new THREE.DirectionalLight(0xe6eeff, .6);
  fill.position.set(-5, 1, 4); scene.add(fill);
  const direction = new THREE.Vector3(9.5, 6.2, -12).normalize();
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right).normalize();
  const bounds = new THREE.Box3();
  const center = new THREE.Vector3();
  const point = new THREE.Vector3();
  let width = 1, height = 1, frame = 340, framing = 0, motion = true, visible = true, dirty = true, dead = false, contextLost = false;
  let raf = 0, lastTime = 0, idleTime = 0, averageCost = 0, slowSamples = 0, renders = 0;
  let fps = 30;

  function resize() {
    width = holder.clientWidth; height = holder.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false); dirty = true; schedule();
  }
  function fit() {
    presentation.updateMatrixWorld(true);
    bounds.setFromObject(presentation);
    bounds.getCenter(center);
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < 8; i++) {
      point.set(i & 1 ? bounds.max.x : bounds.min.x, i & 2 ? bounds.max.y : bounds.min.y, i & 4 ? bounds.max.z : bounds.min.z).sub(center);
      const x = point.dot(right), y = point.dot(up);
      xMin = Math.min(xMin, x); xMax = Math.max(xMax, x); yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
    }
    const mobile = width < 761;
    const heroRect = mobile ? [.03, .48, .94, .45] : [.51, .06, .49, .87];
    const storyRect = mobile ? [.03, .43, .94, .40] : [.025, .29, .95, .58];
    const rect = heroRect.map((n, i) => THREE.MathUtils.lerp(n, storyRect[i], framing));
    const aspect = width / height;
    const span = Math.max((yMax - yMin) / rect[3], (xMax - xMin) / rect[2] / aspect) * 1.08;
    const halfHeight = span / 2, halfWidth = halfHeight * aspect;
    const ndcX = (rect[0] + rect[2] / 2) * 2 - 1;
    const ndcY = 1 - (rect[1] + rect[3] / 2) * 2;
    const aim = center.clone().addScaledVector(right, -ndcX * halfWidth).addScaledVector(up, -ndcY * halfHeight);
    const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    camera.position.copy(aim).addScaledVector(direction, distance);
    camera.lookAt(aim);
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
  }
  function draw(time) {
    raf = 0;
    if (dead || contextLost || !visible || document.hidden || !motion) return;
    if (time - lastTime < 1000 / fps) { schedule(); return; }
    const start = performance.now();
    idleTime += Math.min((time - lastTime) / 1000, .05);
    lastTime = time;
    mixer.setTime(frame / 30);
    // Whole-object motion only: the authored parts remain rigid and retain their thickness.
    presentation.rotation.x = Math.sin(idleTime * .25) * .018;
    presentation.rotation.y = Math.sin(idleTime * .2) * .035;
    fit();
    renderer.render(scene, camera);
    visual.dataset.scene = 'ready';
    visual.dataset.renderFrame = frame.toFixed(2);
    visual.dataset.triangles = String(renderer.info.render.triangles);
    visual.dataset.drawCalls = String(renderer.info.render.calls);
    visual.dataset.renders = String(++renders);
    averageCost = averageCost * .95 + (performance.now() - start) * .05;
    if (averageCost > 24) slowSamples++; else slowSamples = Math.max(0, slowSamples - 1);
    if (slowSamples > 50) { fps = 24; renderer.setPixelRatio(1); slowSamples = 0; }
    visual.dataset.fpsLimit = String(fps);
    dirty = false;
    schedule();
  }
  function schedule() { if (!raf && !dead && !contextLost && visible && !document.hidden && motion) raf = requestAnimationFrame(draw); }
  function cancel() { if (raf) cancelAnimationFrame(raf); raf = 0; }
  // Observe the containing section: the negatively margined sticky canvas may extend
  // outside it, even when an opaque product section completely covers the canvas.
  const headerHeight = document.querySelector('.header').offsetHeight;
  const visibility = new IntersectionObserver(([entry]) => {
    // A narrow sliver behind the header is not enough to warrant GPU rendering.
    visible = entry.isIntersecting && entry.intersectionRatio >= .01;
    if (visible) schedule(); else cancel();
  }, {rootMargin:`-${headerHeight}px 0px 0px 0px`, threshold: .01});
  visibility.observe(document.querySelector('#experience'));
  const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(holder);
  const onVisibility = () => { if (document.hidden) cancel(); else { lastTime = 0; schedule(); } };
  document.addEventListener('visibilitychange', onVisibility);
  const onLost = (event) => { event.preventDefault(); contextLost = true; cancel(); visual.dataset.scene = 'poster'; };
  const onRestored = () => { contextLost = false; dirty = true; schedule(); };
  renderer.domElement.addEventListener('webglcontextlost', onLost);
  renderer.domElement.addEventListener('webglcontextrestored', onRestored);
  resize(); schedule();
  return {
    setPose(nextFrame, nextFraming) { frame = nextFrame; framing = nextFraming; dirty = true; schedule(); },
    setMotion(enabled) { motion = enabled; if (motion) { lastTime = 0; schedule(); } else cancel(); },
    dispose() {
      dead = true; cancel(); visibility.disconnect(); resizeObserver.disconnect(); document.removeEventListener('visibilitychange', onVisibility);
      renderer.domElement.removeEventListener('webglcontextlost', onLost); renderer.domElement.removeEventListener('webglcontextrestored', onRestored);
      mixer.stopAllAction(); mixer.uncacheRoot(gltf.scene);
      const textures = new Set(), materials = new Set(), geometries = new Set();
      gltf.scene.traverse((obj) => { if (!obj.isMesh) return; geometries.add(obj.geometry); for (const mat of Array.isArray(obj.material) ? obj.material : [obj.material]) materials.add(mat); });
      materials.forEach((mat) => { Object.values(mat).forEach((value) => { if (value?.isTexture) textures.add(value); }); mat.dispose(); });
      textures.forEach((texture) => texture.dispose()); geometries.forEach((geometry) => geometry.dispose()); env.dispose(); renderer.dispose(); renderer.domElement.remove();
    }
  };
}
