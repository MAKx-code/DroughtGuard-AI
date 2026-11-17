"""Final verification smoke test after redesign — no Streamlit import."""
import sys
sys.path.insert(0,'.')
print('=== DroughtGuard AI Final Verification ===')

# 1. plotly scatter_map API
import plotly.express as px
import pandas as pd
df_test = pd.DataFrame({'lat':[15.3,16.2],'lon':[76.5,77.3],'district':['Koppal','Raichur'],
                        'risk_severity':['High','Severe'],'risk_score':[62,78],
                        'rainfall_deviation_pct':[-30,-45],'reservoir_storage_pct':[22,15]})
fig = px.scatter_map(df_test, lat='lat', lon='lon', color='risk_severity',
                     size='risk_score', hover_name='district',
                     map_style='carto-darkmatter', zoom=6)
print('1. px.scatter_map (new API) OK')

# 2. background_gradient with matplotlib
df2 = pd.DataFrame({'A':[60,78,45]})
styled = df2.style.background_gradient(subset=['A'], cmap='RdYlGn_r')
print('2. background_gradient (matplotlib) OK')

# 3. SVG background asset
from pathlib import Path
import base64
svg_path = Path('assets/bg_farmland.svg')
assert svg_path.exists()
data = base64.b64encode(svg_path.read_bytes()).decode()
assert len(data) > 100
print(f'3. SVG background asset OK ({len(svg_path.read_bytes())} bytes)')

# 4. All 6 page identifiers in app.py
app_src = Path('app.py').read_text(encoding='utf-8')
for p in ['Overview','Drought Risk','Agent Intelligence','Farmer Advisory','Officer','Monitoring']:
    assert p in app_src, f'Page missing: {p}'
print('4. All 6 pages present OK')

# 5. Design system in app.py
for token in ['C_FOREST','C_AMBER','C_SAGE_LT','C_OFFWHITE','SEV_COLORS','CHART_LAYOUT']:
    assert token in app_src, f'Token missing: {token}'
print('5. Design tokens OK')

# 6. No scatter_mapbox (old API)
assert 'scatter_mapbox' not in app_src
assert 'scatter_map' in app_src
print('6. scatter_map API (not scatter_mapbox) confirmed')

# 7. All functional features preserved
for feature in ['approve_alert','reject_alert','run_agent1','run_agent2','run_agent3',
                'guardrail_status','get_monitoring_data']:
    assert feature in app_src, f'Feature missing: {feature}'
print('7. All functional features preserved OK')

# 8. Synthetic data label present
assert 'SYNTHETIC DEMO' in app_src or 'Synthetic Demo' in app_src
print('8. Synthetic data label present OK')

# 9. No CHART_LAYOUT xaxis/yaxis (removed to fix duplicate kwarg)
assert 'xaxis=dict' not in app_src[app_src.find('CHART_LAYOUT = dict'):app_src.find('CHART_LAYOUT = dict')+400]
print('9. CHART_LAYOUT xaxis/yaxis conflict fixed OK')

# 10. Core modules still work
from data_service import get_dataset, get_taluk_data
from drought_engine import calculate_risk
from mcp_server import call_tool
from guardrails import check_input
df = get_dataset()
row = get_taluk_data('Raichur','Manvi')
r = calculate_risk(row)
w = call_tool('get_district_weather', district='Kalaburagi')
g = check_input('What crops should I plant during drought?', 'test')
print(f'10. Core modules OK: {len(df)} rows, risk={r["score"]}, weather={w["rainfall_received_mm"]}mm, guardrail={g.allowed}')

print()
print('=== ALL VERIFICATION CHECKS PASSED ===')
print('Dashboard at: http://localhost:8501')
