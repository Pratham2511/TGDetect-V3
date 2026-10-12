import { onScroll, animate, createTimer } from 'animejs';

const experience = document.querySelector('#experience');
const hero = document.querySelector('#hero');
const assembly = document.querySelector('#assembly');
const visual = document.querySelector('.visual');
const poster = document.querySelector('#poster');
const motionButton = document.querySelector('#motion');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
let motion = !reduced.matches && !navigator.connection?.saveData;
let scene, observer, layout, lastProgress = 0, activeChapter = -1, loading = false;
let disposed = false;
const chapters = [...document.querySelectorAll('[data-chapter]')];
const jumps = [...document.querySelectorAll('[data-jump]')];
const clamp = (n) => Math.max(0, Math.min(1, n));
const smooth = (n) => { const t = clamp(n); return t * t * (3 - 2 * t); };
const chapterFrames = [1, 100, 177, 221, 294, 340];

function measure() {
  const header = document.querySelector('.header').offsetHeight;
  layout = { header, hero: hero.offsetHeight, travel: experience.offsetHeight - innerHeight + header, story: assembly.offsetHeight - innerHeight + header };
}
function sample(progress) {
  lastProgress = progress;
  if (!layout) measure();
  const distance = progress * layout.travel;
  const storyProgress = clamp((distance - layout.hero) / layout.story);
  const chapter = Math.min(4, Math.floor(storyProgress * 5));
  if (chapter !== activeChapter) {
    activeChapter = chapter;
    chapters.forEach((el, i) => { el.hidden = i !== chapter; });
    jumps.forEach((el, i) => { if (i === chapter) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current'); });
    document.querySelector('#stage-count').textContent = `0${chapter + 1} / 05`;
  }
  const local = storyProgress * 5 - chapter;
  // Each chapter rests before and after its movement, in either scroll direction.
  let frame = chapterFrames[chapter] + (chapterFrames[chapter + 1] - chapterFrames[chapter]) * smooth((local - .20) / .55);
  if (distance < layout.hero) frame = 340 - 339 * smooth((distance / layout.hero - .38) / .62);
  const framing = smooth(distance / layout.hero);
  visual.dataset.frame = frame.toFixed(2);
  visual.dataset.chapter = String(chapter + 1);
  visual.dataset.progress = progress.toFixed(4);
  if (!motion || visual.dataset.scene === 'poster') {
    const exploded = distance >= layout.hero && chapter < 4;
    const source = exploded ? '/landing/assets/exploded.jpg' : '/landing/assets/assembled.jpg';
    if (poster.getAttribute('src') !== source) poster.src = source;
    if (framing > .6) { poster.style.width = '100%'; poster.style.height = '45%'; poster.style.top = '40%'; poster.style.right = '0'; }
    else { poster.removeAttribute('style'); }
  }
  scene?.setPose(frame, framing);
}
async function loadScene() {
  if (loading || scene || !motion || disposed) return;
  loading = true;
  try {
    const { createScene } = await import('./scene.js');
    if (disposed) return;
    scene = await createScene(document.querySelector('#model'), visual);
    if (disposed) { scene.dispose(); return; }
    scene.setMotion(motion);
    sample(lastProgress);
  } catch (error) {
    visual.dataset.scene = 'poster';
    visual.dataset.error = error.message;
    console.warn('Interactive model unavailable; Blender stills remain visible.', error);
    motion = false;
    updateMotion();
  } finally { loading = false; }
}
function updateMotion() {
  motionButton.textContent = motion ? 'Motion on' : 'Motion off';
  motionButton.setAttribute('aria-pressed', String(motion));
  motionButton.setAttribute('aria-label', motion ? 'Motion on, pause animation' : 'Motion off, enable animation');
  document.documentElement.classList.toggle('motion-off', !motion);
  scene?.setMotion(motion);
  if (!motion) visual.dataset.scene = 'static';
  else if (scene) visual.dataset.scene = 'ready';
  else visual.dataset.scene = 'poster';
  poster.removeAttribute('style');
  sample(lastProgress);
  if (motion) loadScene();
}
motionButton.addEventListener('click', () => { motion = !motion; updateMotion(); });
reduced.addEventListener('change', () => { motion = !reduced.matches && !navigator.connection?.saveData; updateMotion(); });
measure();
// Linking a clock keeps endpoint updates active even when an anchor skips the entire range.
const scrollClock = createTimer({ duration: 1000, autoplay: false });
observer = onScroll({ target: experience, sync: true, repeat: true, enter: () => `${layout.header} top`, leave: 'bottom bottom', onUpdate: (state) => sample(state.progress) });
observer.link(scrollClock);
const resize = new ResizeObserver(() => { measure(); if (observer.target) observer.refresh(); sample(lastProgress); });
resize.observe(experience);
resize.observe(hero);
jumps.forEach((button, i) => button.addEventListener('click', () => {
  const top = experience.offsetTop - layout.header + layout.hero + layout.story * ((i + .32) / 5);
  scrollTo({ top, behavior: motion ? 'smooth' : 'instant' });
}));

const tabs = [...document.querySelectorAll('[role=tab]')];
function selectTab(tab) {
  tabs.forEach((el) => {
    const selected = el === tab;
    el.setAttribute('aria-selected', String(selected));
    el.tabIndex = selected ? 0 : -1;
    document.getElementById(el.getAttribute('aria-controls')).hidden = !selected;
  });
}
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (event) => {
    let index;
    if (event.key === 'ArrowRight') index = (i + 1) % tabs.length;
    if (event.key === 'ArrowLeft') index = (i - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') index = 0;
    if (event.key === 'End') index = tabs.length - 1;
    if (index !== undefined) { event.preventDefault(); selectTab(tabs[index]); tabs[index].focus(); }
  });
});

const technology = document.querySelector('.technology');
const reveal = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.target === technology) entry.target.classList.toggle('in-view', entry.isIntersecting && motion);
    else if (entry.isIntersecting) {
      if (motion) animate(entry.target, { translateY: [14, 0], opacity: [.65, 1], duration: 650, ease: 'outCubic' });
      reveal.unobserve(entry.target);
    }
  }
}, { threshold: .12 });
reveal.observe(technology);
document.querySelectorAll('.section-heading,.closing h2').forEach((el) => reveal.observe(el));
updateMotion();
document.fonts.ready.then(() => { measure(); if (observer.target) observer.refresh(); });
addEventListener('pagehide', (event) => {
  if (event.persisted) return;
  disposed = true;
  observer.revert(); scrollClock.cancel(); resize.disconnect(); reveal.disconnect(); scene?.dispose();
});
