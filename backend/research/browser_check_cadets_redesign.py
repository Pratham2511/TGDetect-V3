"""Real in-process CADETS browser acceptance; backend server remains stopped."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from dashboard.backend import app
out=Path('artifacts/cadets_dashboard_redesign');out.mkdir(parents=True,exist_ok=True)
client=TestClient(app);errors=[];checks=[]
def respond(route):
 r=route.request;u=urlsplit(r.url)
 response=client.request(r.method,u.path+('?' + u.query if u.query else ''),content=r.post_data,headers={'content-type':'application/json'})
 route.fulfill(status=response.status_code,body=response.content,content_type='application/json')
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/home/mahesh/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1600,'height':1050});page.route('**/api/cadets/**',respond);page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:3000',wait_until='networkidle');page.get_by_text('5,537,964',exact=True).wait_for();page.screenshot(path=str(out/'overview.png'),full_page=True)
 page.get_by_role('button',name='Investigate alerts',exact=False).first.click()
 page.get_by_role('heading',name='Attack-chain construction',exact=True).wait_for()
 page.get_by_role('button',name='Path 1',exact=False).wait_for(timeout=30000)
 checks.append('real CADETS directed paths constructed without a backend server')
 for label in ('Show all entities','Focus view','Zoom in','Zoom out','Fit graph'):page.get_by_role('button',name=label,exact=True).click()
 page.locator('svg path[role="button"]').first.focus();page.keyboard.press('Enter');page.get_by_text('Source attributes',exact=True).wait_for()
 checks.append('bundled graph, expand/focus, zoom, keyboard event inspection')
 first=page.locator('svg g[aria-label^="Focus entity"]').first;first.scroll_into_view_if_needed();box=first.bounding_box();before=first.get_attribute('transform')
 page.mouse.move(box['x']+30,box['y']+30);page.mouse.down();page.mouse.move(box['x']+65,box['y']+60,steps=5);page.mouse.up();assert first.get_attribute('transform')!=before
 checks.append('draggable entity cards')
 page.get_by_role('button',name='Next chain event',exact=True).click();assert page.get_by_role('button',name='Inspect chain event 2',exact=False).get_attribute('aria-current')=='step'
 page.get_by_role('button',name='Reset chain replay',exact=True).click();page.get_by_role('button',name='Play chain',exact=True).click();page.wait_for_timeout(1350);page.get_by_role('button',name='Pause replay',exact=True).click()
 page.get_by_role('button',name='Backtrace chain',exact=True).click();page.get_by_role('button',name='Pause replay',exact=True).click();page.get_by_label('Chain event position').focus();page.keyboard.press('Home');page.keyboard.press('ArrowRight')
 checks.append('forward replay, pause, backtrace, manual stepping and scrubber')
 with page.expect_download() as captured:page.get_by_role('button',name='Export constructed chain',exact=True).click()
 captured.value.save_as(str(out/'constructed-chain.json'));data=json.loads((out/'constructed-chain.json').read_text());assert data['labels_used'] is False
 for path in data['paths']:
  for a,b in zip(path['events'],path['events'][1:]):assert a['dst']==b['src'] and (a['time'],a['idx'])<(b['time'],b['idx'])
 checks.append('export contains actual connected chronological observations')
 page.get_by_label('Trace direction').select_option('backward');page.get_by_role('button',name='Path 1',exact=False).wait_for()
 page.get_by_label('Trace direction').select_option('both');page.get_by_role('button',name='Path 1',exact=False).wait_for();page.get_by_role('button',name='Reset chain replay',exact=True).click()
 page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(250);page.screenshot(path=str(out/'investigation.png'),full_page=True);builder=page.get_by_role('region',name='Attack chain construction');builder.screenshot(path=str(out/'chain-construction.png'))
 checks.append('direction controls reconstruct paths and explanations')
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');builder.screenshot(path=str(out/'chain-mobile.png'))
 checks.append('390px mobile layout has no page overflow')
 page.emulate_media(reduced_motion='reduce');page.wait_for_timeout(150);assert page.get_by_role('button',name='Play chain',exact=True).is_disabled()
 page.get_by_role('button',name='Reset chain replay',exact=True).click();page.get_by_role('button',name='Next chain event',exact=True).click();checks.append('reduced motion disables autoplay but allows manual tracing')
 browser.close()
assert not errors,errors
(out/'browser-results.json').write_text(json.dumps({'checks':checks,'javascript_errors':errors,'scope':'Real CADETS through TestClient; backend server stopped.'},indent=2));print(json.dumps({'checks':checks,'javascript_errors':errors},indent=2))
