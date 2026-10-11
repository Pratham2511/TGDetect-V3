"""Desktop/mobile acceptance against the real CADETS service and V3 weights."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path('artifacts/cadets_dashboard_review')
out.mkdir(parents=True,exist_ok=True)
errors = []
checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/home/mahesh/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',headless=True,args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:3000',wait_until='networkidle')
    page.get_by_text('5,537,964',exact=True).wait_for(timeout=30000)
    page.screenshot(path=str(out/'overview-desktop.png'),full_page=True)
    checks.append('all splits loaded; real held-out metrics shown')
    page.get_by_role('button',name='Investigate alerts',exact=False).first.click()
    page.get_by_text('Why this entity was flagged',exact=True).wait_for()
    page.locator('[role="region"][aria-label="Candidate temporal chain graph"] svg').wait_for()
    page.get_by_text('19.4664',exact=True).first.wait_for()
    page.screenshot(path=str(out/'investigation-desktop.png'),full_page=True)
    checks.append('ranked alerts, exact scoring explanation and bounded temporal graph')
    page.locator('svg path[role="button"]').first.focus()
    page.keyboard.press('Enter')
    page.get_by_role('region',name='Selected event').wait_for() if page.get_by_role('region',name='Selected event').count() else page.get_by_text('SELECTED OBSERVED EVENT').wait_for()
    page.get_by_text('Source attributes',exact=True).wait_for()
    checks.append('keyboard graph inspection and observed endpoint attributes')
    with page.expect_download() as captured:
        page.get_by_role('button',name='Export',exact=True).click()
    downloaded = captured.value
    downloaded.save_as(str(out/'chain.json'))
    assert json.loads((out/'chain.json').read_text())['labels_used'] is False
    checks.append('chain export excludes label-driven reconstruction')
    page.get_by_role('button',name='Event explorer',exact=True).click()
    page.get_by_role('button',name='test:0',exact=True).wait_for()
    page.get_by_role('button',name='Next page',exact=True).click()
    page.get_by_role('button',name='test:20',exact=True).wait_for()
    page.get_by_label('Filter by node ID').fill('218987')
    page.get_by_role('button',name='Apply',exact=True).click()
    page.get_by_text('#218987',exact=True).first.wait_for()
    page.get_by_label('Dataset split').select_option('train')
    page.get_by_role('button',name='Clear',exact=True).click()
    page.get_by_role('button',name='train:0',exact=True).wait_for()
    assert page.get_by_text('Not scored',exact=True).count() == 20
    checks.append('event pagination, entity search, split switching and honest missing scores')
    page.screenshot(path=str(out/'events-desktop.png'),full_page=True)
    page.get_by_role('button',name='Model & validation',exact=True).click()
    page.get_by_role('button',name='Run 1,024 test events',exact=True).click()
    page.get_by_text('Saved GPU score difference',exact=True).wait_for(timeout=120000)
    page.screenshot(path=str(out/'model-desktop.png'),full_page=True)
    checks.append('real checkpoint chronological inference in browser')
    page.set_viewport_size({'width':390,'height':844})
    for label in ('Command center','Investigate alerts','Event explorer','Model & validation'):
        page.get_by_role('button',name=label,exact=False).first.click()
        page.wait_for_timeout(250)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),label+' mobile overflow'
    page.get_by_role('button',name='Command center',exact=True).click()
    page.screenshot(path=str(out/'overview-mobile.png'),full_page=True)
    checks.append('all four views fit a 390px mobile viewport')
    page.route('**/api/cadets/summary',lambda route:route.fulfill(status=503,content_type='application/json',body=json.dumps({'detail':'Index unavailable test'})))
    page.reload(wait_until='networkidle')
    page.get_by_role('alert').get_by_text('Index unavailable test',exact=True).wait_for()
    checks.append('service failure is visibly reported')
    page.unroute('**/api/cadets/summary')
    page.set_viewport_size({'width':1440,'height':1000})
    page.goto('http://127.0.0.1:3000/legacy',wait_until='domcontentloaded')
    page.locator('#nav-overview').wait_for(timeout=60000)
    checks.append('legacy Sysmon workspace remains accessible')
    browser.close()
assert not errors,errors
(out/'browser-results.json').write_text(json.dumps({'checks':checks,'javascript_errors':errors},indent=2))
print(json.dumps({'checks':checks,'javascript_errors':errors},indent=2))
