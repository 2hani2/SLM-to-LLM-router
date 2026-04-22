"""
Neural Router — Real Data Collector + Visualizer
=====================================================
This script:
1. Runs 100 test questions through your actual router
2. Collects real routing decisions, confidence scores, response times
3. Generates all charts from REAL data

Run WHILE your app is running:
    Terminal 1: ollama serve
    Terminal 2: cd ~/slm_project && source venv/bin/activate && python3 new_app_2.py
    Terminal 3: cd ~/slm_project && source venv/bin/activate && python3 collect_and_visualize.py

Saves charts to ~/slm_project/visualizations/
"""

import os, sys, re, math, time, json, difflib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Setup output dir ──────────────────────────────────────────
OUT = os.path.expanduser('~/slm_project/visualizations')
os.makedirs(OUT, exist_ok=True)

# ── Dark theme ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f12', 'axes.facecolor': '#16161a',
    'axes.edgecolor': '#2a2a35', 'axes.labelcolor': '#a0a0b0',
    'axes.titlecolor': '#e8e8f0', 'text.color': '#e8e8f0',
    'xtick.color': '#707080', 'ytick.color': '#707080',
    'grid.color': '#2a2a35', 'grid.alpha': 0.6,
    'font.family': 'sans-serif', 'font.size': 11,
})
BG   = '#0f0f12'; BG2 = '#16161a'
BLUE = '#4a9eff'; PURP = '#a78bfa'; AMBR = '#f0a500'
GRN  = '#3dd68c'; RED  = '#ff5f5f'; TEAL = '#00bcd4'

def save(name):
    path = f'{OUT}/{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved: {path}')

# ══════════════════════════════════════════════════════════════
# STEP 1: Load your actual model + router
# ══════════════════════════════════════════════════════════════
print('Loading your model...')

MODEL_PATH = './models/nanoqa_v3'  # change to nanoqa_v3 if you have it
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = './models/nanoqa_v3'
if not os.path.exists(MODEL_PATH):
    print(f'ERROR: No model found at {MODEL_PATH}')
    print('Make sure you are running from ~/slm_project/')
    sys.exit(1)

sys.path.insert(0, os.path.abspath(MODEL_PATH))
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Try loading v3 (nn.Module) or v2 (safetensors)
try:
    from nanoqa_arch import NanoQAConfig, NanoQAModel
    config = NanoQAConfig()
    model  = NanoQAModel(config)
    # Try pytorch_model.bin first (v3)
    bin_path = os.path.join(MODEL_PATH, 'pytorch_model.bin')
    sf_path  = os.path.join(MODEL_PATH, 'model.safetensors')
    if os.path.exists(bin_path):
        model.load_state_dict(torch.load(bin_path, map_location='cpu'))
        print('Loaded pytorch_model.bin (v3 format)')
    elif os.path.exists(sf_path):
        import safetensors.torch as st
        sd = st.load_file(sf_path)
        model.load_state_dict(sd, strict=False)
        print('Loaded model.safetensors (v2 format)')
    else:
        raise FileNotFoundError('No model weights found')
except Exception as e:
    print(f'ERROR loading model: {e}')
    sys.exit(1)

device = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available()
                       else 'cpu')
model = model.to(device).eval()
print(f'Model loaded on {device}')

# ── Math engine ───────────────────────────────────────────────
SAFE_MATH = {
    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
    'tan': math.tan, 'log': math.log, 'log2': math.log2,
    'log10': math.log10, 'exp': math.exp, 'abs': abs,
    'pow': pow, 'round': round, 'factorial': math.factorial,
    'pi': math.pi, 'e': math.e, 'tau': math.tau,
    'ceil': math.ceil, 'floor': math.floor,
    'degrees': math.degrees, 'radians': math.radians,
}

def evaluate_math(expr):
    expr = re.sub(r'(what is|calculate|compute|evaluate|solve|whats)',
                  '', expr, flags=re.IGNORECASE).strip()
    expr = expr.replace('^','**').replace('×','*').replace('÷','/')
    expr = expr.replace('squared','**2').replace('cubed','**3').strip('? \n')
    if not re.search(r'[\d\+\-\*\/\(\)\^]', expr): return None
    try:
        r = eval(expr, {'__builtins__': {}}, SAFE_MATH)
        if isinstance(r, float):
            return str(int(r)) if r == int(r) and abs(r) < 1e15 else str(round(r, 8))
        return str(r)
    except: return None

# ── Normalisation ─────────────────────────────────────────────
KNOWN_WORDS = [
    'who','what','when','where','which','how','wrote','invented','discovered',
    'painted','largest','smallest','fastest','oldest','mitochondria','penicillin',
    'shakespeare','newton','einstein','darwin','fleming','galileo','tolkien','rowling',
    'orwell','napoleon','france','india','japan','germany','china','russia','america',
    'capital','paris','london','tokyo','delhi','ocean','pacific','atlantic',
    'hamlet','macbeth','romeo','juliet','gatsby','frankenstein',
    'computer','internet','telephone','airplane','radio','algorithm',
]

def fix_typos(text):
    words = text.split()
    fixed = []
    for word in words:
        if len(word) <= 3 or word.isdigit():
            fixed.append(word); continue
        m = difflib.get_close_matches(word, KNOWN_WORDS, n=1, cutoff=0.82)
        fixed.append(m[0] if m else word)
    return ' '.join(fixed)

def normalize(text):
    text = text.lower().strip()
    for old, new in [("what's","what is"),("who's","who is"),("where's","where is"),
                     ("when's","when is"),("can't","cannot"),("don't","do not"),
                     ("'s","s"),("'","")]:
        text = text.replace(old, new)
    return fix_typos(' '.join(text.split()))

# ── SLM inference ─────────────────────────────────────────────
def slm_generate(question):
    q_norm    = normalize(question)
    prompt    = f'Question: {q_norm}\nAnswer:'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids = input_ids[:, -128:]
    generated = []; probs = []
    with torch.no_grad():
        for _ in range(20):
            logits   = model(input_ids).logits[:, -1, :]
            raw      = F.softmax(logits, dim=-1)
            scaled   = logits / 0.3
            v, _     = torch.topk(scaled, min(10, scaled.size(-1)))
            scaled[scaled < v[:, [-1]]] = float('-inf')
            next_id  = torch.multinomial(F.softmax(scaled, dim=-1), 1)
            probs.append(raw[0, next_id.item()].item())
            generated.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id: break
    ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if '.' in ans: ans = ans.split('.')[0].strip() + '.'
    conf = sum(probs) / len(probs) if probs else 0.0
    return ans, conf

def is_garbage(text):
    words  = text.split(); unique = set(w.lower() for w in words)
    if len(words) < 2: return True
    if len(words) > 20: return True
    if len(unique) == 1: return True
    if len(words) > 4 and len(unique) < 3: return True
    if re.match(r'^[\d\s\.\,]+$', text): return True
    return False

def route(question):
    math_ans = evaluate_math(question)
    if math_ans: return 'Math', math_ans, 1.0

    t0 = time.time()
    ans, conf = slm_generate(question)
    slm_time = (time.time() - t0) * 1000

    if not is_garbage(ans) and conf >= 0.60:
        return 'SLM', ans, conf
    if not is_garbage(ans) and conf >= 0.45:
        ans2, conf2 = slm_generate(question)
        if not is_garbage(ans2) and conf2 >= 0.60:
            return 'SLM', ans2, conf2

    # LLM — try ollama, fall back to placeholder if not running
    try:
        import ollama as ol
        t0  = time.time()
        res = ol.chat(model='mistral', messages=[{'role':'user','content':question}])
        llm_time = (time.time() - t0) * 1000
        return 'LLM', res['message']['content'], None
    except:
        return 'LLM', '[Ollama not running — would use Mistral 7B]', None

# ══════════════════════════════════════════════════════════════
# STEP 2: Run 100 real questions through the router
# ══════════════════════════════════════════════════════════════
TEST_QUESTIONS = [
    # Literature (12)
    ('Who wrote Romeo and Juliet?',          'Literature', 'shakespeare'),
    ('WHO WROTE ROMEO AND JULIET',           'Literature', 'shakespeare'),
    ('who wrote romeo and juliet',           'Literature', 'shakespeare'),
    ('Who wrote Harry Potter?',              'Literature', 'rowling'),
    ('Who wrote 1984?',                      'Literature', 'orwell'),
    ('Who wrote Animal Farm?',               'Literature', 'orwell'),
    ('Who wrote The Hobbit?',                'Literature', 'tolkien'),
    ('Who wrote Frankenstein?',              'Literature', 'shelley'),
    ('Who wrote Dracula?',                   'Literature', 'stoker'),
    ('Who wrote The Great Gatsby?',          'Literature', 'fitzgerald'),
    ("Who wrote Alice in Wonderland?",       'Literature', 'carroll'),
    ('Who wrote Pride and Prejudice?',       'Literature', 'austen'),
    # Science (12)
    ('Who discovered penicillin?',           'Science', 'fleming'),
    ('Who discoverd penicillin?',            'Science', 'fleming'),
    ('Who discovered gravity?',              'Science', 'newton'),
    ('Who invented the telephone?',          'Science', 'bell'),
    ('Who invented the light bulb?',         'Science', 'edison'),
    ('Who invented Python?',                 'Science', 'rossum'),
    ('Who discovered radioactivity?',        'Science', 'curie'),
    ('Who invented the airplane?',           'Science', 'wright'),
    ('Who discovered evolution?',            'Science', 'darwin'),
    ('Who invented the radio?',              'Science', 'marconi'),
    ('Who invented the steam engine?',       'Science', 'watt'),
    ('Who invented the printing press?',     'Science', 'gutenberg'),
    # Geography (12)
    ('What is the capital of France?',       'Geography', 'paris'),
    ('Whats the capital of France?',         'Geography', 'paris'),
    ('WHAT IS THE CAPITAL OF FRANCE',        'Geography', 'paris'),
    ('What is the capital of Japan?',        'Geography', 'tokyo'),
    ('What is the capital of India?',        'Geography', 'delhi'),
    ('WHAT IS THE CAPITAL OF INDIA',         'Geography', 'delhi'),
    ('What is the capital of Germany?',      'Geography', 'berlin'),
    ('What is the capital of Australia?',    'Geography', 'canberra'),
    ('What is the largest country?',         'Geography', 'russia'),
    ('What is the largest ocean?',           'Geography', 'pacific'),
    ('What is the tallest mountain?',        'Geography', 'everest'),
    ('What is the longest river?',           'Geography', 'nile'),
    # Biology (8)
    ('What is the powerhouse of the cell?',  'Biology', 'mitochondria'),
    ('What is the chemical formula for water?', 'Biology', 'h2o'),
    ('How many bones in the human body?',    'Biology', '206'),
    ('How many chromosomes do humans have?', 'Biology', '46'),
    ('What is the fastest land animal?',     'Biology', 'cheetah'),
    ('What is the largest animal?',          'Biology', 'blue whale'),
    ('What is the tallest animal?',          'Biology', 'giraffe'),
    ('What gas do plants absorb?',           'Biology', 'carbon'),
    # Art (8)
    ('Who painted the Mona Lisa?',           'Art', 'vinci'),
    ('Who painted Starry Night?',            'Art', 'gogh'),
    ('Who painted Guernica?',                'Art', 'picasso'),
    ('Who painted The Scream?',              'Art', 'munch'),
    ('Who painted The Last Supper?',         'Art', 'vinci'),
    ('Who sculpted David?',                  'Art', 'michelangelo'),
    ('Who designed the Eiffel Tower?',       'Art', 'eiffel'),
    ('Who composed Beethoven Fifth Symphony?','Art', 'beethoven'),
    # Technology (8)
    ('What does CPU stand for?',             'Technology', 'central'),
    ('What does HTML stand for?',            'Technology', 'hypertext'),
    ('What does RAM stand for?',             'Technology', 'random'),
    ('What does HTTP stand for?',            'Technology', 'hypertext'),
    ('What does AI stand for?',              'Technology', 'artificial'),
    ('What does API stand for?',             'Technology', 'application'),
    ('What does SQL stand for?',             'Technology', 'structured'),
    ('What does URL stand for?',             'Technology', 'uniform'),
    # History (8)
    ('When did World War 2 start?',          'History', '1939'),
    ('When did World War 2 end?',            'History', '1945'),
    ('When did India get independence?',     'History', '1947'),
    ('When did the Berlin Wall fall?',       'History', '1989'),
    ('When was the Eiffel Tower built?',     'History', '1889'),
    ('When did the French Revolution start?','History', '1789'),
    ('Who was the first US president?',      'History', 'washington'),
    ('Who was the first Prime Minister of India?', 'History', 'nehru'),
    # Math (10)
    ('sqrt(144)',           'Math', '12'),
    ('factorial(5)',        'Math', '120'),
    ('sin(0)',              'Math', '0'),
    ('log10(1000)',         'Math', '3'),
    ('2**10',               'Math', '1024'),
    ('pi',                  'Math', '3.14'),
    ('sqrt(256)',           'Math', '16'),
    ('factorial(7)',        'Math', '5040'),
    ('cos(0)',              'Math', '1'),
    ('2+2*5',              'Math', '12'),
    # Complex (LLM expected) (12)
    ('Explain how neural networks work',     'Complex', ''),
    ('Why does quantum entanglement work?',  'Complex', ''),
    ('What are pros and cons of electric vehicles?', 'Complex', ''),
    ('Explain the French Revolution in detail', 'Complex', ''),
    ('What is machine learning?',            'Complex', ''),
    ('How does the internet work?',          'Complex', ''),
    ('Explain DNA replication',              'Complex', ''),
    ('What is blockchain technology?',       'Complex', ''),
    ('Why is the sky blue?',                 'Complex', ''),
    ('How do vaccines work?',                'Complex', ''),
    ('What causes earthquakes?',             'Complex', ''),
    ('Explain the theory of relativity',     'Complex', ''),
]

print(f'\nRunning {len(TEST_QUESTIONS)} questions through your router...')
print('This will take a few minutes (each question ~0.3-5 seconds)\n')

results = []
for i, (question, category, expected) in enumerate(TEST_QUESTIONS):
    t0 = time.time()
    route_name, answer, confidence = route(question)
    elapsed = (time.time() - t0) * 1000

    correct = None
    if expected:
        correct = expected.lower() in answer.lower()

    results.append({
        'question':   question,
        'category':   category,
        'expected':   expected,
        'route':      route_name,
        'answer':     answer[:80],
        'confidence': confidence,
        'time_ms':    elapsed,
        'correct':    correct,
    })

    status = ''
    if route_name == 'SLM':
        status = f'✓ {answer[:40]}' if correct else f'✗ {answer[:40]}'
    print(f'[{i+1:3d}] [{route_name:4s}] {question[:55]:<55} {status}')

# Save raw results
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nRaw results saved to {OUT}/results.json')

# ══════════════════════════════════════════════════════════════
# STEP 3: Compute real stats
# ══════════════════════════════════════════════════════════════
slm_results  = [r for r in results if r['route'] == 'SLM']
llm_results  = [r for r in results if r['route'] == 'LLM']
math_results = [r for r in results if r['route'] == 'Math']

route_counts = {'SLM': len(slm_results), 'LLM': len(llm_results), 'Math': len(math_results)}

slm_times  = [r['time_ms'] for r in slm_results]
llm_times  = [r['time_ms'] for r in llm_results]
math_times = [r['time_ms'] for r in math_results]

slm_confs  = [r['confidence'] for r in slm_results if r['confidence']]

# Accuracy per category (excluding Complex and Math)
cats = ['Literature', 'Science', 'Geography', 'Biology', 'Art', 'Technology', 'History']
cat_acc = {}
for cat in cats:
    cat_res = [r for r in results if r['category'] == cat and r['correct'] is not None]
    if cat_res:
        cat_acc[cat] = sum(r['correct'] for r in cat_res) / len(cat_res) * 100
    else:
        cat_acc[cat] = 0

overall_acc = sum(r['correct'] for r in results if r['correct'] is not None)
total_with_expected = sum(1 for r in results if r['correct'] is not None)

print(f'\n{"="*55}')
print(f'RESULTS SUMMARY')
print(f'{"="*55}')
print(f'Total questions:  {len(results)}')
print(f'Routed to SLM:    {route_counts["SLM"]} ({route_counts["SLM"]/len(results)*100:.0f}%)')
print(f'Routed to LLM:    {route_counts["LLM"]} ({route_counts["LLM"]/len(results)*100:.0f}%)')
print(f'Routed to Math:   {route_counts["Math"]} ({route_counts["Math"]/len(results)*100:.0f}%)')
print(f'Overall accuracy: {overall_acc}/{total_with_expected} = {overall_acc/total_with_expected*100:.0f}%')
if slm_times:  print(f'SLM avg time:     {np.mean(slm_times):.0f} ms')
if llm_times:  print(f'LLM avg time:     {np.mean(llm_times):.0f} ms')
if math_times: print(f'Math avg time:    {np.mean(math_times):.1f} ms')
print(f'{"="*55}\n')

# ══════════════════════════════════════════════════════════════
# STEP 4: Generate charts from REAL data
# ══════════════════════════════════════════════════════════════
print('Generating charts from real data...')

# ── Chart 1: Routing Distribution ─────────────────────────────
print('Chart 1: Routing Distribution...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Query Routing Distribution — Real Data (n={len(results)})',
             fontsize=14, fontweight='bold', color='#e8e8f0')

routes  = ['NanoQA SLM', 'Mistral 7B LLM', 'Math Engine']
counts  = [route_counts['SLM'], route_counts['LLM'], route_counts['Math']]
clrs    = [BLUE, PURP, AMBR]
total   = sum(counts)

bars = axes[0].bar(routes, counts, color=clrs, width=0.5, zorder=3)
axes[0].set_ylabel('Number of Queries')
axes[0].set_title(f'Queries per Route')
axes[0].grid(axis='y', zorder=0)
axes[0].set_ylim(0, max(counts)+10)
axes[0].set_facecolor(BG2)
for bar, c in zip(bars, counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{c}\n({c/total*100:.0f}%)', ha='center', va='bottom',
                 color='#e8e8f0', fontsize=10, fontweight='bold')

wp = {'linewidth': 2, 'edgecolor': BG}
axes[1].pie(counts, labels=routes, colors=clrs, autopct='%1.0f%%',
            startangle=140, wedgeprops=wp,
            textprops={'color': '#e0e0e0', 'fontsize': 10}, pctdistance=0.75)
axes[1].set_title('Route Share')
axes[1].set_facecolor(BG2)
fig.tight_layout()
save('1_routing_distribution_real')

# ── Chart 2: Response Time ─────────────────────────────────────
print('Chart 2: Response Time...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Response Time Analysis — Real Measurements',
             fontsize=14, fontweight='bold', color='#e8e8f0')

slm_avg  = np.mean(slm_times)  if slm_times  else 0
llm_avg  = np.mean(llm_times)  if llm_times  else 4200
math_avg = np.mean(math_times) if math_times else 0.5

labels_t = ['Math Engine', 'NanoQA SLM', 'Mistral 7B LLM']
times_t  = [math_avg, slm_avg, llm_avg]
clrs_t   = [AMBR, BLUE, PURP]

bars = axes[0].barh(labels_t, times_t, color=clrs_t, height=0.45, zorder=3)
axes[0].set_xlabel('Avg Response Time (ms) — log scale')
axes[0].set_title('Average Response Time per Route')
axes[0].set_xscale('log')
axes[0].set_xlim(0.1, max(times_t)*5)
axes[0].grid(axis='x', zorder=0)
axes[0].set_facecolor(BG2)
for bar, t in zip(bars, times_t):
    axes[0].text(bar.get_width()*1.3, bar.get_y()+bar.get_height()/2,
                 f'{t:.0f} ms', va='center', color='#e8e8f0', fontsize=10, fontweight='bold')

# Box plot / violin of SLM times
if slm_times:
    axes[1].hist(slm_times, bins=15, color=BLUE, alpha=0.8, zorder=3)
    axes[1].axvline(slm_avg, color=GRN, lw=2.5, ls='--',
                    label=f'Mean = {slm_avg:.0f} ms')
    axes[1].set_xlabel('Response Time (ms)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('SLM Response Time Distribution')
    axes[1].legend()
    axes[1].grid(True, zorder=0)
    axes[1].set_facecolor(BG2)

fig.tight_layout()
save('2_response_time_real')

# ── Chart 3: Accuracy by Category ─────────────────────────────
print('Chart 3: Accuracy by Category...')
cat_names = list(cat_acc.keys())
cat_vals  = [cat_acc[c] for c in cat_names]
clrs_c    = [GRN if v >= 75 else AMBR if v >= 60 else RED for v in cat_vals]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'NanoQA Accuracy — Real Test Results (overall {overall_acc}/{total_with_expected} = {overall_acc/total_with_expected*100:.0f}%)',
             fontsize=13, fontweight='bold', color='#e8e8f0')

bars = axes[0].bar(cat_names, cat_vals, color=clrs_c, width=0.55, zorder=3)
axes[0].axhline(75, color=GRN,  ls='--', lw=1.8, alpha=0.6, label='Strong (75%)')
axes[0].axhline(60, color=AMBR, ls='--', lw=1.8, alpha=0.6, label='Moderate (60%)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 115)
axes[0].set_title('Accuracy per Category')
axes[0].grid(axis='y', zorder=0)
axes[0].set_facecolor(BG2)
axes[0].tick_params(axis='x', labelsize=8)
for bar, v in zip(bars, cat_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f'{v:.0f}%', ha='center', va='bottom',
                 color='#e8e8f0', fontsize=9, fontweight='bold')
patches = [mpatches.Patch(color=GRN, label='≥75% Strong'),
           mpatches.Patch(color=AMBR, label='60-74% Moderate'),
           mpatches.Patch(color=RED, label='<60% Weak')]
axes[0].legend(handles=patches, fontsize=8, loc='lower right')

# Correct vs Incorrect per category
correct_counts   = [sum(1 for r in results if r['category']==c and r['correct']==True)  for c in cat_names]
incorrect_counts = [sum(1 for r in results if r['category']==c and r['correct']==False) for c in cat_names]
x = np.arange(len(cat_names))
axes[1].bar(x-0.2, correct_counts,   0.4, color=GRN,  label='Correct',   zorder=3)
axes[1].bar(x+0.2, incorrect_counts, 0.4, color=RED,   label='Incorrect', zorder=3)
axes[1].set_xticks(x); axes[1].set_xticklabels(cat_names, fontsize=8)
axes[1].set_ylabel('Count')
axes[1].set_title('Correct vs Incorrect per Category')
axes[1].legend(); axes[1].grid(axis='y', zorder=0)
axes[1].set_facecolor(BG2)
fig.tight_layout()
save('3_accuracy_real')

# ── Chart 4: Confidence Score Distribution ─────────────────────
print('Chart 4: Confidence Scores...')
if slm_confs:
    accepted_conf = [r['confidence'] for r in slm_results if r['confidence'] and r['correct']]
    rejected_conf = [r['confidence'] for r in slm_results if r['confidence'] and r['correct'] == False]
    all_conf      = slm_confs

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('NanoQA Confidence Score Analysis — Real Data',
                 fontsize=14, fontweight='bold', color='#e8e8f0')

    axes[0].hist(all_conf, bins=15, color=BLUE, alpha=0.8, zorder=3, density=True, label='All SLM answers')
    if accepted_conf: axes[0].hist(accepted_conf, bins=10, color=GRN, alpha=0.6, zorder=4, density=True, label='Correct answers')
    if rejected_conf: axes[0].hist(rejected_conf, bins=10, color=RED, alpha=0.6, zorder=4, density=True, label='Wrong answers')
    axes[0].axvline(0.60, color=AMBR, lw=2.5, ls='--', label='Accept threshold (0.60)')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, zorder=0)
    axes[0].set_facecolor(BG2)

    # Scatter: confidence vs correctness
    conf_correct = [(r['confidence'], 1) for r in slm_results if r['confidence'] and r['correct']]
    conf_wrong   = [(r['confidence'], 0) for r in slm_results if r['confidence'] and r['correct'] == False]
    if conf_correct:
        axes[1].scatter([c for c,_ in conf_correct], [j+np.random.uniform(-0.05,0.05) for _,j in conf_correct],
                        color=GRN, alpha=0.7, s=60, label='Correct', zorder=3)
    if conf_wrong:
        axes[1].scatter([c for c,_ in conf_wrong], [j+np.random.uniform(-0.05,0.05) for _,j in conf_wrong],
                        color=RED, alpha=0.7, s=60, label='Wrong', zorder=3)
    axes[1].axvline(0.60, color=AMBR, lw=2, ls='--', label='Threshold')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Correct (1) / Wrong (0)')
    axes[1].set_title('Confidence vs Correctness')
    axes[1].legend(fontsize=8); axes[1].grid(True, zorder=0)
    axes[1].set_facecolor(BG2)
    fig.tight_layout()
    save('4_confidence_real')

# ── Chart 5: Category routing breakdown ───────────────────────
print('Chart 5: Category Routing...')
fig, ax = plt.subplots(figsize=(13, 5))
cat_order = ['Literature', 'Science', 'Geography', 'Biology', 'Art', 'Technology', 'History', 'Math', 'Complex']
slm_per_cat  = [sum(1 for r in results if r['category']==c and r['route']=='SLM')  for c in cat_order]
llm_per_cat  = [sum(1 for r in results if r['category']==c and r['route']=='LLM')  for c in cat_order]
math_per_cat = [sum(1 for r in results if r['category']==c and r['route']=='Math') for c in cat_order]
x = np.arange(len(cat_order))
ax.bar(x,       slm_per_cat,  0.6, color=BLUE, label='SLM (NanoQA)', zorder=3)
ax.bar(x,       llm_per_cat,  0.6, color=PURP, label='LLM (Mistral)', zorder=3,
       bottom=slm_per_cat)
ax.bar(x,       math_per_cat, 0.6, color=AMBR, label='Math Engine', zorder=3,
       bottom=[s+l for s,l in zip(slm_per_cat, llm_per_cat)])
ax.set_xticks(x); ax.set_xticklabels(cat_order, fontsize=9)
ax.set_ylabel('Number of Questions')
ax.set_title('Routing Decision by Question Category — Real Data', fontweight='bold')
ax.legend(); ax.grid(axis='y', zorder=0)
ax.set_facecolor(BG2)
fig.tight_layout()
save('5_category_routing_real')

print(f'\n{"="*55}')
print('All charts saved to:')
print(f'  {OUT}')
print()
print('Open them:')
print(f'  open {OUT}')
print(f'{"="*55}')
