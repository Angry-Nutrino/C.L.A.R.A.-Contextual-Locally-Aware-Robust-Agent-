import json, random

def gen_id():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_', k=20))

def make_slate(text):
    return [{'type':'paragraph','children':[{'text':line}],'id':gen_id()} for line in text.split('\n')]

def make_rect(x, y, w, h, zi):
    return {
        'modifiedAt':1777463700600,'modifiedBy':'-OrNng_LdkPQFz8mOSIP',
        'type':'rectangle','userId':'tx77g76OMEQf8NS355lYBPwDGN82',
        'version':1,'x':x,'y':y,'elementStyle':0,'roughness':0,
        'width':w,'height':h,'seed':random.randint(100000,9999999),
        'zIndex':zi,'id':gen_id(),'isDeleted':False,'opacity':100,'angle':0,
        'shouldApplyRoughness':True,'groupIds':[],'lockedGroupId':None,
        'diagramId':None,'containerId':None,'figureId':None,
        'fillStyle':'solid','backgroundColor':'transparent',
        'strokeColor':'#1c1c1c','strokeWidth':1,'strokeStyle':'solid',
        'strokeSharpness':'round','colorMode':0
    }

def make_textbox(x, y, w, h, text, zi, scale=0.875):
    return {
        'modifiedAt':1777463700600,'modifiedBy':'-OrNng_LdkPQFz8mOSIP',
        'type':'textbox','userId':'tx77g76OMEQf8NS355lYBPwDGN82',
        'version':1,'x':x,'y':y,'scale':scale,'fontSize':20,
        'textAlign':'left','fontFamily':2,'mode':'normal',
        'text':text,'slate':make_slate(text),'roughness':0,
        'verticalAlign':'middle','width':w,'height':h,
        'seed':random.randint(100000,9999999),'hasFixedBounds':True,
        'zIndex':zi,'id':gen_id(),'isDeleted':False,'opacity':100,'angle':0,
        'shouldApplyRoughness':True,'groupIds':[],'lockedGroupId':None,
        'diagramId':None,'containerId':None,'figureId':None,
        'backgroundColor':'transparent','fillStyle':'solid',
        'strokeColor':'#1c1c1c','strokeWidth':1,'strokeStyle':'solid',
        'strokeSharpness':'round'
    }

def make_arrow(x, y, points, zi, stroke_style='solid'):
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    return {
        'modifiedAt':1777463700600,'modifiedBy':'-OrNng_LdkPQFz8mOSIP',
        'type':'arrow','userId':'tx77g76OMEQf8NS355lYBPwDGN82',
        'version':1,'x':x,'y':y,
        'strokeColor':'#1c1c1c','backgroundColor':'transparent',
        'fillStyle':'solid','strokeWidth':1,'strokeStyle':stroke_style,
        'roughness':0,'strokeSharpness':'round','points':points,
        'width':max(xs)-min(xs),'height':max(ys)-min(ys),
        'seed':random.randint(100000,9999999),
        'zIndex':zi,'id':gen_id(),'isDeleted':False,'opacity':100,'angle':0,
        'shouldApplyRoughness':True,'groupIds':[],'lockedGroupId':None,
        'diagramId':None,'containerId':None,'figureId':None,
        'lastCommittedPoint':None,'startArrowhead':None,'endArrowhead':'arrow',
        'arrowHeadSize':12,'textGap':None,'gaps':[],
        'startBinding':None,'endBinding':None
    }

# LOAD from current updated diagram
with open('clara_diagram_updated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
elements = data['elements']
zi = max(e.get('zIndex', 0) for e in elements) + 1

new_elems = []

# ─────────────────────────────────────────────────────────────────
# /soul  HTTP GET endpoint
# Backend rect: x=[89,328], y=[-202,324]
# Place /soul as a small box directly above the backend rect.
# Backend rect top is y=-202. Put soul box at y=-290, h=70 → bottom=-220.
# Gap of 18px above backend rect. Same x width: x=89, w=239.
# ─────────────────────────────────────────────────────────────────
soul_x, soul_y, soul_w, soul_h = 89, -290, 239, 70
new_elems.append(make_rect(soul_x, soul_y, soul_w, soul_h, zi)); zi+=1
new_elems.append(make_textbox(soul_x+10, soul_y+10, soul_w-20, soul_h-20,
    'GET /soul\ncpu% | VRAM | RAM  →  sidebar vitals',
    zi, scale=0.75)); zi+=1

# ─────────────────────────────────────────────────────────────────
# token_usage WS event label
# This fires back over /ws after every user request.
# Place it as a floating label to the right of the backend rect,
# near the top where the WS return arrow would exit toward the UI.
# Backend rect right edge: x=328. Put label at x=340, y=-100, w=180, h=50.
# ─────────────────────────────────────────────────────────────────
tok_x, tok_y, tok_w, tok_h = 340, -110, 200, 55
new_elems.append(make_textbox(tok_x, tok_y, tok_w, tok_h,
    'WS: token_usage event\nprompt | completion | cached',
    zi, scale=0.72)); zi+=1

elements.extend(new_elems)

with open('clara_diagram_updated.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, separators=(',',':'))

print(f'Added {len(new_elems)} elements (soul box x2 + token_usage label)')
print(f'Total elements: {len(elements)}')
print(f'/soul box:       x=[{soul_x},{soul_x+soul_w}]  y=[{soul_y},{soul_y+soul_h}]')
print(f'token_usage:     x=[{tok_x},{tok_x+tok_w}]  y=[{tok_y},{tok_y+tok_h}]')
print('Saved: clara_diagram_updated.json')
