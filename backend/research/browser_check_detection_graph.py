"""Live upload/inference and connected-entity graph browser acceptance."""
import json
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
from playwright.sync_api import sync_playwright, expect
from research.test_graph_uploads import events_from_prepared
out=Path('artifacts/cadets_detection_graph');out.mkdir(parents=True,exist_ok=True)
records=events_from_prepared(1024)
records.append(dict(time=records[-1]['time']+1,relation='CONNECT',src={'id':'browser-test-host','host':'acceptance-isolated-host','type':'subject','cmd_line':'curl'},dst={'id':'browser-test-ip','remote_ip':'192.0.2.200','type':'netflow'}))
fixture=out/'acceptance-upload.json';fixture.write_text(json.dumps(records))
checks,errors=[],[]
def graph_response(r):return '/api/graph-datasets/' in r.url and '/graph?' in r.url and r.ok
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/home/mahesh/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:3000',wait_until='networkidle')
    page.get_by_role('button',name='Detection graph',exact=True).click()
    red=page.locator('g[data-prediction="predicted_malicious"]');green=page.locator('g[data-prediction="predicted_benign"]')
    red.first.wait_for();green.first.wait_for()
    checks.append('bundled graph loads from dataset API with both classes')
    with page.expect_response(graph_response) as response:
        page.get_by_label('Detection class filter').select_option('malicious')
    graph=response.value.json();assert all(next(n for n in graph['nodes'] if n['id']==i)['alert'] for i in graph['seeds'])
    assert any(not n['alert'] for n in graph['nodes'])
    checks.append('malicious selection retains connected benign context')
    red.first.wait_for();identifier=red.first.get_attribute('data-node-id')
    with page.expect_response(graph_response) as response:
        red.first.focus();page.keyboard.press('Enter')
    assert response.value.json()['seeds']==[int(identifier)]
    page.get_by_role('complementary',name='Detection node details').wait_for()
    checks.append('clicking a red node fetches its actual connected neighborhood')
    page.get_by_role('button',name='Reset focus',exact=True).click()
    with page.expect_response(graph_response):
        page.get_by_label('Detection class filter').select_option('all')
    page.get_by_label('Graph starting entity').select_option('128.55.12.167')
    page.wait_for_load_state('networkidle');page.locator('g[data-node-id="9997"]').wait_for()
    assert '128.55.12.167' in page.locator('g[data-node-id="9997"]').text_content()
    checks.append('IP dropdown focuses recorded address rather than a fixed sample')
    page.screenshot(path=str(out/'detection-graph.png'),full_page=True)
    with page.expect_response(lambda r:'/api/graph-datasets?name=' in r.url and r.request.method=='POST') as response:
        page.get_by_label('Upload graph logs').set_input_files(str(fixture))
    assert response.value.status==202
    page.wait_for_function('document.querySelector("[aria-label=\\"Graph dataset\\"]").value!=="cadets"',timeout=60000)
    uploaded=page.get_by_label('Graph dataset',exact=True).input_value()
    page.locator('g[data-node-id]').first.wait_for()
    checks.append('file picker uploads logs, real V3 scoring completes, uploaded graph opens automatically')
    with page.expect_response(lambda r:'/entities?' in r.url and 'kind=host' in r.url and r.ok):
        page.get_by_label('Graph entity type').select_option('host')
    page.get_by_label('Graph starting entity').select_option('acceptance-isolated-host')
    page.wait_for_load_state('networkidle')
    expect(page.locator('g[data-node-id]')).to_have_count(2)
    assert '192.0.2.200' in page.get_by_role('region',name='Malicious and benign node graph').text_content()
    checks.append('uploaded host selector shows only its connected two-node component')
    page.locator('g[data-node-id]').first.click();page.get_by_role('complementary',name='Detection node details').wait_for()
    page.get_by_label('Show benchmark label separately').check();page.get_by_text('Benchmark: not supplied.',exact=False).wait_for()
    checks.append('uploaded nodes expose actual scores without invented benchmark labels')
    with page.expect_download() as captured:page.get_by_role('button',name='Export graph',exact=True).click()
    captured.value.save_as(str(out/'graph.json'));data=json.loads((out/'graph.json').read_text())
    assert data['dataset']['id']==uploaded and len(data['nodes'])==2 and data['events'][0]['split']=='upload'
    checks.append('export contains selected upload and connected observed events')
    page.screenshot(path=str(out/'uploaded-host-graph.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(out/'detection-mobile.png'),full_page=True)
    checks.append('upload controls, selectors and graph fit 390px viewport')
    page.reload(wait_until='networkidle');page.get_by_role('button',name='Detection graph',exact=True).click()
    page.locator(f'option[value="{uploaded}"]').wait_for(state='attached')
    checks.append('uploaded dataset survives page reload')
    browser.close()
assert not errors,errors
(out/'browser-results.json').write_text(json.dumps({'checks':checks,'javascript_errors':errors,'fixture':'CADETS excerpt plus explicitly constructed isolated acceptance-test component'},indent=2))
print(json.dumps({'checks':checks,'javascript_errors':errors},indent=2))
