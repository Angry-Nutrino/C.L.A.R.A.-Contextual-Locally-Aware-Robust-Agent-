import json, uuid, random

def gen_id():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_', k=20))

def make_slate(text):
    return [{'type':'paragraph','children':[{'text':line}],'id':gen_id()} for line in text.split('\n')]

def update_textbox(elements, eid, new_text):
    for e in elements:
        if e.get('id') == eid:
            e['text'] = new_text
            e['slate'] = make_slate(new_text)
            e['modifiedAt'] = 1777463700600
            return True
    print(f'WARNING: {eid} not found')
    return False

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

# LOAD
with open('clara_diagram.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
elements = data['elements']
max_zi = max(e.get('zIndex', 0) for e in elements)

# =====================
# TEXT UPDATES
# =====================
# Existing element positions for reference:
#   Interpreter rect:          x=[-295,-71]  y=[-577,-362]  center-top=(-183,-577)  center-bottom=(-183,-362)
#   Consolidation Memory box:  x=[-629,-396] y=[-382,-269]
#   ReAct rect:                x=[-294,-68]  y=[-96, 151]   center-top=(-181,-96)
#   Tools rect:                x=[-800,-604] y=[-11, 225]
#   Tool parser label:         x=-760, y=-124
#   Backend rect:              x=[89, 328]   y=[-202, 324]
#   Memory JSON rect:          x=[89, 328]   y=[-443,-233]  (V2hRpqZXf60dRnSLc9M4z)

updates = [
    # Gatekeeper -> Interpreter
    ('KhZEGTEzx61wwUDASaFXC',
     'INTERPRETER\n\n(grok-4-1-fast-non-reasoning)\n\n\nINTENT JSON\n\ntool  |  confidence\n\nrequires_planning'),
    # Tools: remove MOONDREAM, add VISION/DESKTOP CMD
    ('DbelK8pKmxUsOHQDBr0y_',
     '          TOOLS\n\n\n\nWEB SEARCH\n\n\nPYTHON REPL\n\n\nVISION (Grok)\n\n\nRAG / FAISS\n\n\nDESKTOP CMD'),
    # Consolidation Memory -> memorize_episode
    ('FN1JvtEtbgETCOB4los43',
     '    memorize_episode\n\n      memory.json'),
    # ReAct: fix model name + OBSERVATION -> GLINT
    ('CjDaFexOdI-wXnrT-7eyo',
     '(grok-4-1-fast-reasoning)\n\n\n    ReAct Agent\n\n\nTHOUGHT\n\n\nACTION\n\n\nGLINT'),
    # Tool parser -> Tool Executor
    ('JzzKbala1XFUSLKFYfDgo',
     'TOOL EXECUTOR\n\n  parse_actions()'),
    # /ws endpoint - current handle_message pattern
    ('adsbtiy9-jVnxWQqCZBz2',
     '/ws endpoint\n\n\nasync def handle_message(){\n\n  on_step callback\n\n}\n\n\nmessage_id assigned\n\nfire-and-forget\n\nasyncio.create_task()'),
    # Dashboard stats -> broadcast_task_event
    ('eYNS2bl45uGqgIJjoYulz',
     'broadcast_task_event()\n\ntask board state updates'),
    # BOOST TOOL CALL TRUE -> FAST PATH description
    ('EAzVH791IoPMNU4PSRrcZ',
     'FAST PATH\n\nformat_llm  ~2-4s\non failure -> DELIBERATE'),
    # LLM label -> DELIBERATE
    ('Q5aQSN9pgPFL-n-eXhglK',
     '         DELIBERATE'),
    # Fix typo C.L.A.RA. -> C.L.A.R.A.
    ('KSmp18JNu7VGvXBBJonBK',
     'C.L.A.R.A.'),
    # Memory JSON: updated structure
    ('BhFtWXS4leV9NwrdwLETt',
     'user_profile:{}\n\nepisodic_log:{\n\n  recency: last 3\n\n  semantic: top 2\n\n}\n\nvault:{\n\n  permanent facts\n\n  0.85 cosine dedup\n\n}'),
    # 5. The ReAct Loop -> 5. DELIBERATE ReAct Loop
    ('NMt6x7MqYvDankZR1ANR6',
     '5.\n\nDELIBERATE\nReAct Loop'),
    # Context description - accurate to current get_smart_context
    ('xv4cm7r-6LElZWO8gxOJ_',
     'get_smart_context()\nget_archive_context()\ntool_registry.search()\n-> [DISCOVERED_TOOLS]'),
    # Saving summary -> memorize_episode
    ('Uy9SPT4o1dlFEjdufP1Dk',
     'memorize_episode()\nasyncio.to_thread\n(non-blocking)'),
    # Step 2 label - now shows EventQueue flow
    ('ZdO2EUoGFZ10XDOzAbwLw',
     'WebSocket input\n-> message_id assigned\n-> EventQueue'),
    # Backend block label
    ('FUkB7gaBUEVKR75jcsgPX',
     'THE BACKEND\n\n\nFAST API\n(WebSocket)'),
    # Context grab -> always-on (no longer conditional)
    ('HQ3inRuR-ldOrexY9N3Jn',
     'CONTEXT INJECTION\n(always-on)'),
    # Fix thought stream typo
    ('yUqUfQKQpPFk4kEV0RYUF',
     '5.1*  Thought stream through WebSockets to Stream bar'),
]

for eid, text in updates:
    update_textbox(elements, eid, text)

print(f'Text updates applied: {len(updates)}')

# =====================
# NEW ELEMENTS — CLEAN LAYOUT
# =====================
# Coordinate reference:
#   Interpreter rect top-center:    x=-183, y=-577
#   Interpreter rect bottom-center: x=-183, y=-362
#   Consolidation Memory right edge: x=-396
#   ReAct rect top-center:          x=-181, y=-96
#   Backend rect left edge:         x=89
#   Tools rect top:                 y=-11
#   Canvas currently goes left to x=-800

# Layout strategy:
#   - LEFT COLUMN (x=-1200 to -950): Background Scheduler + EnvironmentWatcher
#     These are autonomous trigger sources, placed far left out of the pipeline flow
#   - CENTER-LEFT COLUMN (x=-620 to -380): EventQueue + OrchestratorLoop
#     Stacked vertically above Interpreter. Clear of existing elements.
#     Interpreter is at y=-577, so these go at y=-900 and y=-780 (plenty of space above)
#   - CENTER (between Interpreter bottom y=-362 and ReAct top y=-96):
#     Router sits here. Gap = 266px. Router at y=-330, h=100, bottom at y=-230.
#     x position: -295 to -71 is where Interpreter is.
#     Consolidation Memory is at x=[-629,-396]. Router at x=-290 (just right of consMem right edge -396).
#     Actually Interpreter is x=[-295,-71], ReAct is x=[-294,-68].
#     Router should be same center x as Interpreter → x=-295, w=224 (same width).
#     But consolidation mem is at y=[-382,-269] and Router at y=[-330,-230] — they overlap y but
#     consolidation mem ends at x=-396 and router starts at x=-295, so NO horizontal overlap. Fine.

new_elems = []
zi = max_zi + 1

# ─────────────────────────────────────────────
# BACKGROUND SCHEDULER  (far left, top)
# y range: -900 to -700 (h=200)
# x range: -1220 to -950 (w=270)
# ─────────────────────────────────────────────
bs_x, bs_y, bs_w, bs_h = -1220, -900, 270, 200
new_elems.append(make_rect(bs_x, bs_y, bs_w, bs_h, zi)); zi+=1
new_elems.append(make_textbox(bs_x+12, bs_y+12, bs_w-24, bs_h-24,
    'BACKGROUND SCHEDULER\n\nhealth_check  2 min\nmemory_maintenance  5 min\ncontext_warmup  10 min',
    zi, scale=0.75)); zi+=1

# ─────────────────────────────────────────────
# ENVIRONMENT WATCHER  (far left, below BS)
# y range: -670 to -470 (h=200)
# x range: -1220 to -950 (w=270)
# gap between BS and EW = 30px
# ─────────────────────────────────────────────
ew_x, ew_y, ew_w, ew_h = -1220, -660, 270, 200
new_elems.append(make_rect(ew_x, ew_y, ew_w, ew_h, zi)); zi+=1
new_elems.append(make_textbox(ew_x+12, ew_y+12, ew_w-24, ew_h-24,
    'ENVIRONMENT WATCHER\n\nfile_change\nmemory_growth\ninteraction_density\nrag_rebuild',
    zi, scale=0.75)); zi+=1

# ─────────────────────────────────────────────
# EVENT QUEUE  (center-left, above Interpreter)
# Interpreter top is y=-577. Put EventQueue at y=-880, h=90 → bottom=-790.
# x center same as Interpreter center (-183). EQ w=240 → x=-303 to -63.
# But that overlaps with Interpreter x=[-295,-71]. Let's offset left:
# x=-580 to -340 (w=240). Center of EQ = -460. Clear of Interpreter.
# Actually let's align EQ+Orch with each other. Both at x=-580, w=240.
# ─────────────────────────────────────────────
eq_x, eq_y, eq_w, eq_h = -580, -900, 240, 90
new_elems.append(make_rect(eq_x, eq_y, eq_w, eq_h, zi)); zi+=1
new_elems.append(make_textbox(eq_x+10, eq_y+10, eq_w-20, eq_h-20,
    'EVENT QUEUE\nasync priority queue',
    zi, scale=0.8)); zi+=1

# ─────────────────────────────────────────────
# ORCHESTRATOR LOOP  (center-left, below EventQueue)
# EQ bottom = -810. Gap 20px. Orch at y=-790, h=90 → bottom=-700.
# Then gap to Interpreter top (-577) = 123px. Enough space.
# ─────────────────────────────────────────────
orch_x, orch_y, orch_w, orch_h = -580, -790, 240, 90
new_elems.append(make_rect(orch_x, orch_y, orch_w, orch_h, zi)); zi+=1
new_elems.append(make_textbox(orch_x+10, orch_y+10, orch_w-20, orch_h-20,
    'ORCHESTRATOR LOOP\ndispatch + retry arch.',
    zi, scale=0.8)); zi+=1

# ─────────────────────────────────────────────
# ROUTER  (between Interpreter bottom and ReAct top)
# Interpreter bottom-center: x=-183, y=-362
# ReAct top-center:          x=-181, y=-96
# Gap = 266px. Router h=110, placed at y=-320 → bottom=-210.
# 114px clearance above ReAct. Fine.
# x: same center as Interpreter/ReAct. Interpreter rect x=-295, w=224.
# Router same: x=-295, w=224 → x range [-295,-71].
# Consolidation mem is at x=[-629,-396] and y=[-382,-269].
# Router is at y=[-320,-210]. Consolidation mem bottom is y=-269 which is above Router top -320.
# So there is NO y-overlap: consol mem ends at y=-269, router starts at y=-320. Fine.
# ─────────────────────────────────────────────
router_x, router_y, router_w, router_h = -295, -295, 224, 110
new_elems.append(make_rect(router_x, router_y, router_w, router_h, zi)); zi+=1
new_elems.append(make_textbox(router_x+10, router_y+10, router_w-20, router_h-20,
    'ROUTER\n\nFAST  /  CHAT  /  DELIBERATE\nconfidence >= 0.75\nuncertainty <= 0.30',
    zi, scale=0.75)); zi+=1

# =====================
# NEW ARROWS
# =====================
# All arrows defined as: make_arrow(start_x, start_y, [[dx,dy],...], zi)
# Points are relative offsets from start.

# 1. Background Scheduler right-center → EventQueue left-center
#    BS right-center:  x=-950, y=-900+100=-800
#    EQ left-center:   x=-580, y=-900+45=-855
#    Route: elbow — go right to x=-580, adjust y
#    From (-950,-800): right 370, then up 55
bs_right_cx = bs_x + bs_w          # -950
bs_right_cy = bs_y + bs_h // 2     # -800
eq_left_cx  = eq_x                  # -580
eq_left_cy  = eq_y + eq_h // 2     # -855
dx = eq_left_cx - bs_right_cx       # 370
dy = eq_left_cy - bs_right_cy       # -55
new_elems.append(make_arrow(bs_right_cx, bs_right_cy, [[0,0],[dx,0],[dx,dy]], zi)); zi+=1

# 2. Environment Watcher right-center → EventQueue left-center
#    EW right-center:  x=-950, y=-660+100=-560
#    EQ left-center:   x=-580, y=-855
#    From (-950,-560): right 370, then up 295
ew_right_cx = ew_x + ew_w           # -950
ew_right_cy = ew_y + ew_h // 2     # -560
dx2 = eq_left_cx - ew_right_cx      # 370
dy2 = eq_left_cy - ew_right_cy     # -295
new_elems.append(make_arrow(ew_right_cx, ew_right_cy, [[0,0],[dx2,0],[dx2,dy2]], zi)); zi+=1

# 3. EventQueue bottom-center → OrchestratorLoop top-center (straight down)
#    EQ bottom-center:   x=-460, y=-810
#    Orch top-center:    x=-460, y=-790
#    Straight down 20px
eq_bot_cx  = eq_x + eq_w // 2      # -460
eq_bot_y   = eq_y + eq_h           # -810
orch_top_y = orch_y                 # -790
new_elems.append(make_arrow(eq_bot_cx, eq_bot_y, [[0,0],[0, orch_top_y - eq_bot_y]], zi)); zi+=1

# 4. OrchestratorLoop bottom-center → Interpreter top-center
#    Orch bottom-center:    x=-460, y=-700
#    Interpreter top-center: x=-183, y=-577
#    Route: elbow right then down
#    From (-460,-700): right 277, then down 123
orch_bot_cx  = orch_x + orch_w // 2  # -460
orch_bot_y   = orch_y + orch_h       # -700
interp_top_cx = -295 + 224//2        # -183
interp_top_y  = -577
dx4 = interp_top_cx - orch_bot_cx   # 277
dy4 = interp_top_y - orch_bot_y     # 123
new_elems.append(make_arrow(orch_bot_cx, orch_bot_y, [[0,0],[dx4,0],[dx4,dy4]], zi)); zi+=1

# 5. Interpreter bottom-center → Router top-center (straight down)
#    Interpreter bottom-center: x=-183, y=-362
#    Router top-center:         x=-183, y=-320
#    Straight down 42px
interp_bot_y  = -362
router_top_y  = router_y            # -295
new_elems.append(make_arrow(interp_top_cx, interp_bot_y, [[0,0],[0, router_top_y - interp_bot_y]], zi)); zi+=1

# 6. Router bottom-center → ReAct top-center (straight down, DELIBERATE path)
#    Router bottom-center: x=-183, y=-210
#    ReAct top-center:     x=-181, y=-96
#    Nearly straight down 114px
router_bot_y = router_y + router_h  # -210
react_top_y  = -96
new_elems.append(make_arrow(interp_top_cx, router_bot_y, [[0,0],[2, react_top_y - router_bot_y]], zi)); zi+=1

# 7. Router right-center → Backend rect (FAST/CHAT path)
#    Router right-center:  x=-71, y=-265
#    Backend left-center:  x=89,  y=61  (backend rect y=[-202,324], center y=61)
#    Route: elbow — right 160 then down 326
router_right_cx = router_x + router_w        # -71
router_mid_y    = router_y + router_h // 2   # -265
backend_left_x  = 89
backend_mid_y   = (-202 + 324) // 2          # 61
dx7 = backend_left_x - router_right_cx       # 160
dy7 = backend_mid_y - router_mid_y           # 326
new_elems.append(make_arrow(router_right_cx, router_mid_y, [[0,0],[dx7,0],[dx7,dy7]], zi)); zi+=1

elements.extend(new_elems)

# =====================
# SAVE
# =====================
with open('clara_diagram_updated.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, separators=(',',':'))

box_count = 5  # BS, EW, EQ, Orch, Router
arrow_count = 7
print(f'New elements added: {len(new_elems)} ({box_count} boxes x2 rect+text + {arrow_count} arrows)')
print(f'Total elements now: {len(elements)}')
print()
print('Layout summary:')
print(f'  Background Scheduler:   x=[{bs_x},{bs_x+bs_w}]  y=[{bs_y},{bs_y+bs_h}]')
print(f'  Environment Watcher:    x=[{ew_x},{ew_x+ew_w}]  y=[{ew_y},{ew_y+ew_h}]')
print(f'  Event Queue:            x=[{eq_x},{eq_x+eq_w}]  y=[{eq_y},{eq_y+eq_h}]')
print(f'  Orchestrator Loop:      x=[{orch_x},{orch_x+orch_w}]  y=[{orch_y},{orch_y+orch_h}]')
print(f'  Router:                 x=[{router_x},{router_x+router_w}]  y=[{router_y},{router_y+router_h}]')
print()
print('Saved: clara_diagram_updated.json')
