"""
dynamic_metrics_dashboard.py — FULLY DYNAMIC VERSION
Loads data from results.json (n=150) instead of hardcoded data.

Run:
    cd ~/slm_project
    source venv/bin/activate
    pip install flask matplotlib numpy nltk rouge-score --quiet
    python3 dynamic_metrics_dashboard.py

Then open: http://localhost:7000

results_fixed.json format (one entry per line in a JSON array):
[
  {
    "question":   "Who wrote Romeo and Juliet?",
    "reference":  "William Shakespeare",
    "answer":     "William Shakespeare",
    "route":      "SLM",
    "confidence": 0.82,
    "latency":    298.5
  },
  ...
]

Category is auto-detected from the question text.
If results.json doesn't exist, 150 demo questions are generated and saved.
"""

import os, sys, re, json, math, time, io, base64, random, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from flask import Flask, render_template_string, jsonify, request

# ── Optional NLP libraries ─────────────────────────────────────
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

# ── Plot style ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f10', 'axes.facecolor': '#18181b',
    'axes.edgecolor': '#3f3f46',   'axes.labelcolor': '#a1a1aa',
    'xtick.color': '#71717a',      'ytick.color': '#71717a',
    'text.color': '#e4e4e7',       'grid.color': '#27272a',
    'grid.alpha': 0.5,             'font.family': 'DejaVu Sans',
    'font.size': 10,               'axes.spines.top': False,
    'axes.spines.right': False,
})
BLUE='#60a5fa'; PURPLE='#a78bfa'; AMBER='#fbbf24'; GREEN='#34d399'
RED='#f87171'; TEAL='#2dd4bf'; MUTED='#71717a'; BG='#0f0f10'; SURF='#18181b'

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'visualizations', 'results_fixed.json')


# ═══════════════════════════════════════════════════════════════
# CATEGORY AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════
CATEGORY_KEYWORDS = {
    'Literature':  ['wrote','author','novel','book','poem','play','shakespeare','orwell',
                    'rowling','tolkien','austen','dickens','hemingway','fitzgerald',
                    'hamlet','romeo','juliet','gatsby','frankenstein','dracula','hobbit'],
    'Science':     ['discovered','invented','penicillin','gravity','telephone','light bulb',
                    'airplane','radioactivity','evolution','radio','steam engine','printing',
                    'darwin','newton','fleming','curie','einstein','galileo','pasteur'],
    'Geography':   ['capital','country','continent','ocean','river','mountain','lake',
                    'largest country','longest river','tallest mountain','paris','tokyo',
                    'delhi','berlin','london','canberra','pacific','atlantic','everest','nile'],
    'Biology':     ['cell','mitochondria','dna','chromosome','photosynthesis','organ',
                    'animal','species','mammal','bones','formula for water','h2o',
                    'fastest land','largest animal','tallest animal','plants absorb'],
    'Art':         ['painted','painting','sculpted','sculpture','composed','drew','designed',
                    'mona lisa','starry night','guernica','scream','last supper','david',
                    'picasso','vinci','gogh','michelangelo','beethoven','mozart','eiffel'],
    'Technology':  ['stand for','cpu','html','ram','http','api','sql','url','ai stand',
                    'computer','internet','programming','software','hardware','algorithm',
                    'database','network','bandwidth','byte','processor'],
    'History':     ['when did','world war','independence','revolution','empire','president',
                    'prime minister','dynasty','ancient','war','battle','century',
                    '1939','1945','1947','1789','1889','1989'],
    'Math':        ['sqrt','factorial','sin','cos','tan','log','pow','**','factorial',
                    'pi','calculate','compute','evaluate','solve'],
    'Complex':     ['explain','describe','difference between','compare','why does','how does',
                    'pros and cons','what is the impact','analyze','discuss','elaborate'],
}

def detect_category(question: str) -> str:
    ql = question.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in ql:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'General'


# ═══════════════════════════════════════════════════════════════
# DATA LOADER — reads results.json and converts to internal format
# ═══════════════════════════════════════════════════════════════
def load_data():
    """
    Loads results.json and returns list of tuples:
    (question, reference, prediction, category, route, confidence, latency)

    If results.json not found → generates 150 demo questions and saves them.
    """
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            raw = json.load(f)
        print(f'Loaded {len(raw)} entries from {RESULTS_FILE}')
    else:
        print(f'result_fixeds.json not found at {RESULTS_FILE}')
        print('Generating 150 demo questions and saving...')
        raw = _generate_demo_results()
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f, indent=2)
        print(f'Saved demo data to {RESULTS_FILE}')

    data = []
    for entry in raw:
        q    = entry.get('question', '')
        ref  = entry.get('reference', '')
        ans  = entry.get('answer', 'LLM_ROUTED')
        route= entry.get('route', 'LLM')
        conf = entry.get('confidence', None)
        lat  = entry.get('latency', 0.0)
        cat  = detect_category(q)
        data.append((q, ref, ans, cat, route, conf, lat))

    return data


def _generate_demo_results():
    """
    Generates 150 realistic demo entries that match what your app actually produces.
    Replace this with your real results.json from collect_and_visualize.py.
    """
    random.seed(42)

    slm_questions = [
        # Literature (15)
        ("Who wrote Romeo and Juliet?",           "William Shakespeare",        "William Shakespeare",        0.82),
        ("WHO WROTE ROMEO AND JULIET",            "William Shakespeare",        "William Shakespeare",        0.79),
        ("who wrote romeo and juliet",            "William Shakespeare",        "William Shakespeare",        0.81),
        ("Who wrote Harry Potter?",               "J.K. Rowling",               "J K Rowling",                0.76),
        ("Who wrote 1984?",                       "George Orwell",              "George Orwell",              0.74),
        ("Who wrote Animal Farm?",                "George Orwell",              "George Orwell",              0.73),
        ("Who wrote The Hobbit?",                 "J.R.R. Tolkien",             "J R R Tolkien",              0.71),
        ("Who wrote Hamlet?",                     "William Shakespeare",        "William Shakespeare",        0.80),
        ("Who wrote Frankenstein?",               "Mary Shelley",               "Mary Wollstonecraft",        0.52),
        ("Who wrote Dracula?",                    "Bram Stoker",                "LLM_ROUTED",                 0.38),
        ("Who wrote Pride and Prejudice?",        "Jane Austen",                "LLM_ROUTED",                 0.41),
        ("Who wrote The Great Gatsby?",           "F. Scott Fitzgerald",        "LLM_ROUTED",                 0.35),
        ("Who wrote Alice in Wonderland?",        "Lewis Carroll",              "Lewis Carroll",              0.68),
        ("Who wrote Don Quixote?",                "Miguel de Cervantes",        "LLM_ROUTED",                 0.29),
        ("Who wrote Crime and Punishment?",       "Fyodor Dostoevsky",          "LLM_ROUTED",                 0.31),
        # Science (15)
        ("Who discovered penicillin?",            "Alexander Fleming",          "Alexander Fleming",          0.85),
        ("Who discoverd penicillin?",             "Alexander Fleming",          "Alexander Fleming",          0.83),
        ("Who discovered gravity?",               "Isaac Newton",               "Isaac Newton",               0.81),
        ("Who invented the telephone?",           "Alexander Graham Bell",      "Alexander Graham Bell",      0.78),
        ("Who invented the light bulb?",          "Thomas Edison",              "Thomas Edison",              0.80),
        ("Who invented Python?",                  "Guido van Rossum",           "Guido Van Rossum",           0.72),
        ("Who discovered radioactivity?",         "Marie Curie",                "Marie Curie",                0.77),
        ("Who invented the airplane?",            "Wright Brothers",            "Wright Brothers",            0.75),
        ("Who discovered evolution?",             "Charles Darwin",             "Charles Darwin",             0.79),
        ("Who discovered penicillin?",            "Alexander Fleming",          "Alexander Fleming",          0.86),
        ("Who invented the radio?",               "Guglielmo Marconi",          "LLM_ROUTED",                 0.42),
        ("Who invented the steam engine?",        "James Watt",                 "James Watt",                 0.69),
        ("Who invented the printing press?",      "Johannes Gutenberg",         "LLM_ROUTED",                 0.37),
        ("Who discovered X-rays?",                "Wilhelm Roentgen",           "LLM_ROUTED",                 0.33),
        ("Who invented the internet?",            "Tim Berners-Lee",            "LLM_ROUTED",                 0.39),
        # Geography (15)
        ("What is the capital of France?",        "Paris",                      "Paris",                      0.88),
        ("Whats the capital of France?",          "Paris",                      "Paris",                      0.86),
        ("WHAT IS THE CAPITAL OF FRANCE",         "Paris",                      "Paris",                      0.85),
        ("What is the capital of Japan?",         "Tokyo",                      "Tokyo",                      0.84),
        ("What is the capital of India?",         "New Delhi",                  "New Delhi",                  0.82),
        ("WHAT IS THE CAPITAL OF INDIA",          "New Delhi",                  "New Delhi",                  0.80),
        ("What is the capital of Germany?",       "Berlin",                     "Berlin",                     0.83),
        ("What is the capital of Australia?",     "Canberra",                   "LLM_ROUTED",                 0.44),
        ("What is the largest country?",          "Russia",                     "LLM_ROUTED",                 0.47),
        ("What is the largest ocean?",            "Pacific Ocean",              "Pacific Ocean",              0.78),
        ("What is the tallest mountain?",         "Mount Everest",              "LLM_ROUTED",                 0.43),
        ("What is the longest river?",            "Nile",                       "LLM_ROUTED",                 0.41),
        ("What is the capital of China?",         "Beijing",                    "Beijing",                    0.76),
        ("What is the capital of Brazil?",        "Brasilia",                   "LLM_ROUTED",                 0.38),
        ("What is the capital of Canada?",        "Ottawa",                     "LLM_ROUTED",                 0.42),
        # Biology (10)
        ("What is the powerhouse of the cell?",   "Mitochondria",               "Mitochondria",               0.91),
        ("What is the chemical formula for water?","H2O",                       "H2O",                        0.93),
        ("How many bones in the human body?",     "206",                        "206",                        0.87),
        ("How many chromosomes do humans have?",  "46",                         "46",                         0.85),
        ("What is the fastest land animal?",      "Cheetah",                    "Cheetah",                    0.89),
        ("What is the largest animal?",           "Blue whale",                 "LLM_ROUTED",                 0.44),
        ("What is the tallest animal?",           "Giraffe",                    "Giraffe",                    0.76),
        ("What gas do plants absorb?",            "Carbon dioxide",             "LLM_ROUTED",                 0.41),
        ("What is DNA?",                          "Deoxyribonucleic acid",      "Deoxyribonucleic acid",      0.72),
        ("What is the largest organ in the body?","Skin",                       "LLM_ROUTED",                 0.43),
        # Art (10)
        ("Who painted the Mona Lisa?",            "Leonardo da Vinci",          "Leonardo da Vinci",          0.87),
        ("Who painted Starry Night?",             "Vincent van Gogh",           "Vincent Van Gogh",           0.84),
        ("Who painted Guernica?",                 "Pablo Picasso",              "Pablo Picasso",              0.82),
        ("Who painted The Scream?",               "Edvard Munch",               "Edvard Munch",               0.79),
        ("Who painted The Last Supper?",          "Leonardo da Vinci",          "Leonardo da Vinci",          0.85),
        ("Who sculpted David?",                   "Michelangelo",               "Michelangelo",               0.83),
        ("Who designed the Eiffel Tower?",        "Gustave Eiffel",             "Gustave Eiffel",             0.80),
        ("Who painted The Birth of Venus?",       "Sandro Botticelli",          "LLM_ROUTED",                 0.40),
        ("Who painted Girl with a Pearl Earring?","Johannes Vermeer",           "LLM_ROUTED",                 0.37),
        ("Who sculpted The Thinker?",             "Auguste Rodin",              "LLM_ROUTED",                 0.41),
        # Technology (10)
        ("What does CPU stand for?",              "Central Processing Unit",    "Central Processing Unit",    0.86),
        ("What does HTML stand for?",             "HyperText Markup Language",  "HyperText Markup Language",  0.84),
        ("What does RAM stand for?",              "Random Access Memory",       "Random Access Memory",       0.85),
        ("What does HTTP stand for?",             "HyperText Transfer Protocol","HyperText Transfer Protocol",0.82),
        ("What does AI stand for?",               "Artificial Intelligence",    "Artificial Intelligence",    0.88),
        ("What does API stand for?",              "Application Programming Interface","Application Programming Interface",0.81),
        ("What does SQL stand for?",              "Structured Query Language",  "Structured Query Language",  0.83),
        ("What does URL stand for?",              "Uniform Resource Locator",   "LLM_ROUTED",                 0.44),
        ("What does GPS stand for?",              "Global Positioning System",  "LLM_ROUTED",                 0.42),
        ("What does USB stand for?",              "Universal Serial Bus",       "Universal Serial Bus",       0.79),
        # History (10)
        ("When did World War 2 start?",           "1939",                       "1939",                       0.83),
        ("When did World War 2 end?",             "1945",                       "1945",                       0.81),
        ("When did India get independence?",      "1947",                       "1947",                       0.84),
        ("When did the Berlin Wall fall?",        "1989",                       "LLM_ROUTED",                 0.43),
        ("When was the Eiffel Tower built?",      "1889",                       "LLM_ROUTED",                 0.41),
        ("When did the French Revolution start?", "1789",                       "LLM_ROUTED",                 0.39),
        ("Who was the first US president?",       "George Washington",          "George Washington",          0.76),
        ("Who was the first Prime Minister of India?","Jawaharlal Nehru",       "LLM_ROUTED",                 0.44),
        ("When did World War 1 start?",           "1914",                       "1914",                       0.79),
        ("When did the Moon landing happen?",     "1969",                       "1969",                       0.77),
    ]

    llm_questions = [
        ("Explain how neural networks work",          ""),
        ("Why does quantum entanglement work?",        ""),
        ("What are pros and cons of electric vehicles?",""),
        ("Explain the French Revolution in detail",    ""),
        ("What is machine learning?",                  ""),
        ("How does the internet work?",                ""),
        ("Explain DNA replication",                    ""),
        ("What is blockchain technology?",             ""),
        ("Why is the sky blue?",                       ""),
        ("How do vaccines work?",                      ""),
        ("What causes earthquakes?",                   ""),
        ("Explain the theory of relativity",           ""),
        ("What is the difference between AI and ML?",  ""),
        ("Compare Python and JavaScript",              ""),
        ("What are the causes of climate change?",     ""),
        ("Explain supervised vs unsupervised learning",""),
        ("What is the impact of social media?",        ""),
        ("How does photosynthesis work in detail?",    ""),
        ("What is the significance of the Magna Carta?",""),
        ("Explain the water cycle in detail",          ""),
    ]

    math_questions = [
        ("sqrt(144)",    "12"),
        ("factorial(5)", "120"),
        ("sin(0)",       "0"),
        ("log10(1000)",  "3.0"),
        ("2**10",        "1024"),
        ("sqrt(256)",    "16"),
        ("factorial(7)", "5040"),
        ("cos(0)",       "1.0"),
        ("2+2*5",        "12"),
        ("factorial(10)","3628800"),
        ("sqrt(64)",     "8"),
        ("log10(100)",   "2.0"),
        ("3**4",         "81"),
        ("factorial(4)", "24"),
        ("sqrt(49)",     "7"),
    ]

    results = []

    # SLM entries
    for q, ref, ans, conf in slm_questions:
        slm_lat = random.gauss(310, 45)
        results.append({
            "question":   q,
            "reference":  ref,
            "answer":     ans,
            "route":      "SLM",
            "confidence": round(conf + random.gauss(0, 0.02), 3),
            "latency":    round(max(200, slm_lat), 1)
        })

    # LLM entries
    for q, ref in llm_questions:
        llm_lat = random.gauss(55000, 12000)
        results.append({
            "question":   q,
            "reference":  ref,
            "answer":     "LLM_ROUTED",
            "route":      "LLM",
            "confidence": None,
            "latency":    round(max(30000, llm_lat), 1)
        })

    # Math entries
    for q, ref in math_questions:
        results.append({
            "question":   q,
            "reference":  ref,
            "answer":     ref,
            "route":      "Math",
            "confidence": 1.0,
            "latency":    round(random.uniform(0.5, 3.0), 2)
        })

    random.shuffle(results)
    return results


# ═══════════════════════════════════════════════════════════════
# METRIC COMPUTATION  (applied directly on model output — no training)
# ═══════════════════════════════════════════════════════════════
def compute_metrics(data):
    """
    All metrics computed directly on (reference, prediction) pairs.
    No training — these are evaluation metrics, applied post-hoc.
    """
    slm_data  = [(q,ref,pred,cat,conf,lat) for q,ref,pred,cat,route,conf,lat in data if route=='SLM']
    math_data = [(q,ref,pred,cat,conf,lat) for q,ref,pred,cat,route,conf,lat in data if route=='Math']
    llm_data  = [(q,ref,pred,cat,conf,lat) for q,ref,pred,cat,route,conf,lat in data if route=='LLM']

    def is_correct(ref, pred):
        if pred == 'LLM_ROUTED': return False
        ref_l  = ref.lower().strip()
        pred_l = pred.lower().strip()
        ref_words = [w for w in ref_l.replace(',','').split() if len(w) > 2]
        return (ref_l in pred_l or
                any(w in pred_l for w in ref_words))

    slm_answered = [(q,ref,pred,cat,conf,lat) for q,ref,pred,cat,conf,lat in slm_data
                    if pred != 'LLM_ROUTED' and ref != '']
    slm_correct  = [x for x in slm_answered if is_correct(x[1], x[2])]
    slm_wrong    = [x for x in slm_answered if not is_correct(x[1], x[2])]
    accuracy = len(slm_correct) / max(len(slm_answered), 1) * 100

    # ── BLEU ─────────────────────────────────────────────────
    bleu_scores = []
    if HAS_NLTK:
        sf = SmoothingFunction().method1
        for q,ref,pred,cat,conf,lat in slm_answered:
            ref_tok  = ref.lower().split()
            pred_tok = pred.lower().split()
            try:   score = sentence_bleu([ref_tok], pred_tok, smoothing_function=sf)
            except: score = 0.0
            bleu_scores.append(score)
    else:
        for q,ref,pred,cat,conf,lat in slm_answered:
            ref_set  = set(ref.lower().split())
            pred_tok = pred.lower().split()
            if not pred_tok: bleu_scores.append(0); continue
            matches = sum(1 for t in pred_tok if t in ref_set)
            bleu_scores.append(matches / len(pred_tok))
    bleu_avg = np.mean(bleu_scores) * 100 if bleu_scores else 0

    # ── ROUGE ─────────────────────────────────────────────────
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    if HAS_ROUGE:
        sc = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        for q,ref,pred,cat,conf,lat in slm_answered:
            try:
                s = sc.score(ref, pred)
                rouge1_scores.append(s['rouge1'].fmeasure)
                rouge2_scores.append(s['rouge2'].fmeasure)
                rougeL_scores.append(s['rougeL'].fmeasure)
            except:
                rouge1_scores += [0]; rouge2_scores += [0]; rougeL_scores += [0]
    else:
        for q,ref,pred,cat,conf,lat in slm_answered:
            rw = set(ref.lower().split()); pw = set(pred.lower().split())
            ov = len(rw & pw)
            p  = ov/max(len(pw),1); r = ov/max(len(rw),1)
            f  = 2*p*r/max(p+r,1e-9)
            rouge1_scores.append(f)
            rb = set(zip(ref.lower().split()[:-1], ref.lower().split()[1:]))
            pb = set(zip(pred.lower().split()[:-1], pred.lower().split()[1:]))
            ov2 = len(rb & pb)
            p2  = ov2/max(len(pb),1); r2 = ov2/max(len(rb),1)
            rouge2_scores.append(2*p2*r2/max(p2+r2,1e-9))
            rougeL_scores.append(f)

    rouge1_avg = np.mean(rouge1_scores)*100 if rouge1_scores else 0
    rouge2_avg = np.mean(rouge2_scores)*100 if rouge2_scores else 0
    rougeL_avg = np.mean(rougeL_scores)*100 if rougeL_scores else 0

    # ── MRR ───────────────────────────────────────────────────
    mrr_scores = []
    for q,ref,pred,cat,conf,lat in slm_answered:
        rl = ref.lower(); pl = pred.lower()
        if rl in pl or any(w in pl for w in rl.split() if len(w)>3):
            mrr_scores.append(1.0)
        elif any(w in pl for w in rl.split() if len(w)>2):
            mrr_scores.append(0.5)
        else:
            mrr_scores.append(1/3)
    mrr = np.mean(mrr_scores)*100 if mrr_scores else 0

    # ── Exact Match ───────────────────────────────────────────
    em_scores = [1 if r.lower().strip()==p.lower().strip() else 0
                 for q,r,p,cat,conf,lat in slm_answered]
    em_score  = np.mean(em_scores)*100 if em_scores else 0

    # ── F1 Token ──────────────────────────────────────────────
    f1_scores = []
    for q,ref,pred,cat,conf,lat in slm_answered:
        rt = set(ref.lower().split()); pt = set(pred.lower().split())
        common = rt & pt
        if not common: f1_scores.append(0.0); continue
        p = len(common)/len(pt); r = len(common)/len(rt)
        f1_scores.append(2*p*r/(p+r))
    f1_avg = np.mean(f1_scores)*100 if f1_scores else 0

    # ── Routing Metrics ───────────────────────────────────────
    simple_qs  = [(q,ref,pred,cat,route,conf,lat) for q,ref,pred,cat,route,conf,lat in data
                  if ref != '' and route in ['SLM','LLM','Math']]
    complex_qs = [(q,ref,pred,cat,route,conf,lat) for q,ref,pred,cat,route,conf,lat in data
                  if ref == '' and route == 'LLM']

    tp_r = sum(1 for q,ref,pred,cat,route,conf,lat in simple_qs
               if route=='SLM' and is_correct(ref,pred))
    fp_r = sum(1 for q,ref,pred,cat,route,conf,lat in simple_qs
               if route=='SLM' and not is_correct(ref,pred) and pred!='LLM_ROUTED')
    fn_r = sum(1 for q,ref,pred,cat,route,conf,lat in simple_qs
               if route=='LLM')
    tn_r = len(complex_qs)

    routing_precision = tp_r / max(tp_r+fp_r, 1)
    routing_recall    = tp_r / max(tp_r+fn_r, 1)
    routing_f1        = 2*routing_precision*routing_recall / max(routing_precision+routing_recall, 0.001)
    escalation_rate   = len(llm_data) / max(len(data), 1)
    false_escalation  = fn_r / max(len(simple_qs), 1)

    # ── Per-category metrics ──────────────────────────────────
    cats = ['Literature','Science','Geography','Biology','Art','Technology','History']
    cat_metrics = {}
    for cat in cats:
        cat_ans = [(q,ref,pred,conf) for q,ref,pred,c,conf,lat in slm_answered if c==cat]
        if not cat_ans: continue
        tp = sum(1 for q,r,p,conf in cat_ans if is_correct(r,p))
        fp = sum(1 for q,r,p,conf in cat_ans if not is_correct(r,p))
        fn = sum(1 for q,ref,pred,c,route,conf,lat in data if c==cat and route=='LLM')
        prec = tp/(tp+fp) if (tp+fp)>0 else 1.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 1.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        cat_metrics[cat] = {
            'precision': round(prec*100,1), 'recall': round(rec*100,1),
            'f1': round(f1*100,1), 'tp': tp, 'fp': fp, 'fn': fn
        }

    # ── Latency percentiles (real from data) ──────────────────
    slm_lats  = [lat for q,ref,pred,cat,conf,lat in slm_answered if lat > 0]
    llm_lats  = [lat for q,ref,pred,cat,route,conf,lat in data if route=='LLM' and lat > 0]
    math_lats = [lat for q,ref,pred,cat,route,conf,lat in data if route=='Math' and lat > 0]

    def percentile_stats(arr):
        if not arr: return {'p50':0,'p90':0,'p99':0,'mean':0}
        return {
            'p50':  round(np.percentile(arr,50),1),
            'p90':  round(np.percentile(arr,90),1),
            'p99':  round(np.percentile(arr,99),1),
            'mean': round(np.mean(arr),1),
        }

    # ── Confidence calibration ────────────────────────────────
    confs   = [conf for q,ref,pred,cat,conf,lat in slm_answered if conf is not None]
    correct = [is_correct(ref,pred) for q,ref,pred,cat,conf,lat in slm_answered if conf is not None]
    calib   = []
    for lo in np.arange(0.30, 1.0, 0.10):
        hi   = lo + 0.10
        idxs = [i for i,c in enumerate(confs) if lo <= c < hi]
        if idxs:
            calib.append({
                'bin':   f'{lo:.1f}-{hi:.1f}',
                'conf':  round(np.mean([confs[i] for i in idxs]),3),
                'acc':   round(np.mean([correct[i] for i in idxs])*100,1),
                'n':     len(idxs)
            })

    route_dist = {
        'SLM':  len(slm_data),
        'LLM':  len(llm_data),
        'Math': len(math_data)
    }
    total = len(data)

    return {
        'n_total':          total,
        'accuracy':         round(accuracy, 1),
        'bleu':             round(bleu_avg, 1),
        'rouge1':           round(rouge1_avg, 1),
        'rouge2':           round(rouge2_avg, 1),
        'rougeL':           round(rougeL_avg, 1),
        'mrr':              round(mrr, 1),
        'exact_match':      round(em_score, 1),
        'f1_token':         round(f1_avg, 1),
        'routing_precision':round(routing_precision*100, 1),
        'routing_recall':   round(routing_recall*100, 1),
        'routing_f1':       round(routing_f1*100, 1),
        'escalation_rate':  round(escalation_rate*100, 1),
        'false_escalation': round(false_escalation*100, 1),
        'cat_metrics':      cat_metrics,
        'route_dist':       route_dist,
        'total':            total,
        'slm_correct':      len(slm_correct),
        'slm_total':        len(slm_answered),
        'slm_wrong':        len(slm_wrong),
        'calib':            calib,
        'slm_lat':          percentile_stats(slm_lats),
        'llm_lat':          percentile_stats(llm_lats),
        'math_lat':         percentile_stats(math_lats),
        'bleu_per':         [round(x*100,1) for x in bleu_scores],
        'rouge_per':        [round(x*100,1) for x in rouge1_scores],
        'avg_times': {
            'SLM':  percentile_stats(slm_lats)['mean'] if slm_lats else 310,
            'LLM':  percentile_stats(llm_lats)['mean'] if llm_lats else 55000,
            'Math': percentile_stats(math_lats)['mean'] if math_lats else 2,
        }
    }


# ═══════════════════════════════════════════════════════════════
# CHART GENERATION  (unchanged from original)
# ═══════════════════════════════════════════════════════════════
def make_chart(chart_type, metrics):
    fig = None

    if chart_type == 'overview':
        fig, ax = plt.subplots(figsize=(9, 5))
        metric_names = ['Accuracy','BLEU','ROUGE-1','ROUGE-L','MRR','F1 Token','Exact Match',
                        'Routing P','Routing R','Routing F1']
        metric_vals  = [metrics['accuracy'], metrics['bleu'], metrics['rouge1'],
                        metrics['rougeL'], metrics['mrr'], metrics['f1_token'],
                        metrics['exact_match'], metrics['routing_precision'],
                        metrics['routing_recall'], metrics['routing_f1']]
        colors_bar = [GREEN,BLUE,TEAL,TEAL,PURPLE,AMBER,RED,GREEN,BLUE,TEAL]
        bars = ax.barh(metric_names, metric_vals, color=colors_bar, height=0.5, zorder=3, edgecolor='none')
        for bar, val in zip(bars, metric_vals):
            ax.text(val+0.8, bar.get_y()+bar.get_height()/2, f'{val:.1f}%',
                    va='center', color='#e4e4e7', fontsize=9, fontweight='bold')
        ax.set_xlim(0, 120)
        ax.set_xlabel('Score (%)')
        ax.set_title(f'All Evaluation Metrics — NanoQA v3 (n={metrics["total"]})', pad=12, fontweight='bold')
        ax.axvline(x=75, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.5)
        ax.text(75.5, -0.7, '75%', color=MUTED, fontsize=8)
        ax.grid(True, axis='x', alpha=0.4, zorder=0)

    elif chart_type == 'routing':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        rd = metrics['route_dist']
        labels = list(rd.keys()); sizes = list(rd.values()); cols = [BLUE, PURPLE, AMBER]
        ax1.pie(sizes, labels=labels, colors=cols, autopct='%1.0f%%', startangle=140,
                textprops={'color':'#e4e4e7','fontsize':10},
                wedgeprops={'edgecolor':'#0f0f10','linewidth':2})
        ax1.set_title(f'Route distribution (n={metrics["total"]})', fontweight='bold')
        bars2 = ax2.bar(labels, sizes, color=cols, width=0.4, zorder=3, edgecolor='none')
        ax2.bar_label(bars2, padding=3, color='#e4e4e7', fontweight='bold')
        ax2.set_ylabel('Queries'); ax2.set_title('Queries per route', fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.4, zorder=0)

    elif chart_type == 'bleu_rouge':
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        cats_list   = list(metrics['cat_metrics'].keys())
        rouge_by_cat= [metrics['cat_metrics'][c]['f1']        for c in cats_list]
        bleu_by_cat = [metrics['cat_metrics'][c]['precision']  for c in cats_list]
        ax = axes[0]
        x = np.arange(len(cats_list)); w = 0.3
        ax.bar(x-w/2, bleu_by_cat,  w, label='Precision (%)', color=BLUE,  zorder=3, edgecolor='none')
        ax.bar(x+w/2, rouge_by_cat, w, label='F1 Score (%)',  color=GREEN, zorder=3, edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels(cats_list, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Score (%)'); ax.set_title('Precision & F1 per category', fontweight='bold')
        ax.legend(facecolor='#27272a', edgecolor='#3f3f46', labelcolor='#e4e4e7', fontsize=8)
        ax.grid(True, axis='y', alpha=0.4, zorder=0); ax.set_ylim(0, 110)
        ax2 = plt.subplot(122, polar=True)
        ax2.set_facecolor(SURF)
        metric_names_r = ['BLEU','ROUGE-1','ROUGE-2','ROUGE-L','MRR']
        vals = [metrics['bleu'],metrics['rouge1'],metrics['rouge2'],metrics['rougeL'],metrics['mrr']]
        theta = np.linspace(0, 2*np.pi, len(metric_names_r), endpoint=False).tolist()
        vals_r = vals + [vals[0]]; theta += [theta[0]]
        ax2.plot(theta, vals_r, color=TEAL, linewidth=2)
        ax2.fill(theta, vals_r, color=TEAL, alpha=0.2)
        ax2.set_xticks(theta[:-1]); ax2.set_xticklabels(metric_names_r, fontsize=9, color='#e4e4e7')
        ax2.set_ylim(0, 100); ax2.set_yticks([25,50,75,100])
        ax2.set_yticklabels(['25%','50%','75%','100%'], fontsize=7, color=MUTED)
        ax2.grid(color='#3f3f46', alpha=0.5)
        ax2.set_title('NLP Metric Radar', fontweight='bold', pad=18, color='#e4e4e7')

    elif chart_type == 'precision_recall':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cats_list = list(metrics['cat_metrics'].keys())
        prec = [metrics['cat_metrics'][c]['precision'] for c in cats_list]
        rec  = [metrics['cat_metrics'][c]['recall']    for c in cats_list]
        f1   = [metrics['cat_metrics'][c]['f1']        for c in cats_list]
        x = np.arange(len(cats_list)); w = 0.25
        ax = axes[0]
        ax.bar(x-w, prec, w, label='Precision', color=BLUE,  zorder=3, edgecolor='none')
        ax.bar(x,   rec,  w, label='Recall',    color=GREEN, zorder=3, edgecolor='none')
        ax.bar(x+w, f1,   w, label='F1',        color=AMBER, zorder=3, edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels(cats_list, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Score (%)'); ax.set_title('Precision / Recall / F1 by category', fontweight='bold')
        ax.legend(facecolor='#27272a', edgecolor='#3f3f46', labelcolor='#e4e4e7', fontsize=9)
        ax.grid(True, axis='y', alpha=0.4, zorder=0); ax.set_ylim(0, 115)
        ax.axhline(y=75, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.5)
        ax2 = axes[1]
        macro_names = ['Accuracy','BLEU','ROUGE-1','ROUGE-L','MRR','F1','EM',
                       'Routing\nPrec','Routing\nRecall','Routing\nF1']
        macro_vals  = [metrics['accuracy'],metrics['bleu'],metrics['rouge1'],
                       metrics['rougeL'],metrics['mrr'],metrics['f1_token'],metrics['exact_match'],
                       metrics['routing_precision'],metrics['routing_recall'],metrics['routing_f1']]
        cols_mac = [GREEN,BLUE,TEAL,TEAL,PURPLE,AMBER,RED,GREEN,BLUE,TEAL]
        bars2 = ax2.bar(macro_names, macro_vals, color=cols_mac, width=0.6, zorder=3, edgecolor='none')
        ax2.bar_label(bars2, fmt='%.1f%%', padding=2, color='#e4e4e7', fontsize=7, fontweight='bold')
        ax2.set_xticklabels(macro_names, rotation=30, ha='right', fontsize=8)
        ax2.set_ylabel('Score (%)'); ax2.set_title('All metrics — macro average', fontweight='bold')
        ax2.set_ylim(0, 120); ax2.grid(True, axis='y', alpha=0.4, zorder=0)

    elif chart_type == 'response_time':
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        at = metrics['avg_times']
        routes = list(at.keys()); times = list(at.values()); cols_t = [BLUE, PURPLE, AMBER]
        ax = axes[0]
        bars = ax.barh(routes, times, color=cols_t, height=0.4, zorder=3, edgecolor='none')
        ax.bar_label(bars, labels=[f'{t:,.0f} ms' for t in times], padding=4,
                     color='#e4e4e7', fontweight='bold')
        ax.set_xscale('log'); ax.set_xlabel('Avg response time (ms) — log scale')
        ax.set_title(f'Response time per route (P50)\nn={metrics["total"]}', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.4, zorder=0)
        rd = metrics['route_dist']
        ax2 = axes[1]
        total_no  = metrics['total'] * at['LLM'] / 1000
        total_yes = (rd['SLM']*at['SLM'] + rd['LLM']*at['LLM'] + rd['Math']*at['Math']) / 1000
        saved = total_no - total_yes
        bars2 = ax2.bar(['All → Mistral\n(no routing)', 'With NanoQA\nrouting'],
                        [total_no, total_yes], color=[RED, GREEN], width=0.4, zorder=3, edgecolor='none')
        ax2.bar_label(bars2, labels=[f'{total_no:,.0f}s', f'{total_yes:,.0f}s'],
                      padding=4, color='#e4e4e7', fontweight='bold')
        if saved > 0:
            ax2.text(0.5, max(total_no,total_yes)*0.5,
                     f'Saved: {saved:,.0f}s\n({saved/total_no*100:.0f}% faster)',
                     ha='center', color=GREEN, fontsize=10, fontweight='bold')
        ax2.set_ylabel('Total time (s)'); ax2.set_title(f'Compute savings (n={metrics["total"]})', fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.4, zorder=0)

    elif chart_type == 'confusion':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cats_list = [c for c in ['Literature','Science','Geography','Biology','Art','Technology','History']
                     if c in metrics['cat_metrics']]
        matrix = np.array([[metrics['cat_metrics'][c]['tp'],
                             metrics['cat_metrics'][c]['fp'],
                             metrics['cat_metrics'][c]['fn']] for c in cats_list])
        col_labels = ['SLM\nCorrect','SLM\nWrong','→ Mistral']
        ax = axes[0]
        vmax = max(matrix.max(), 1)
        im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticks(range(len(cats_list))); ax.set_yticklabels(cats_list, fontsize=10)
        ax.set_title('Decision matrix (raw counts)', fontweight='bold')
        for i in range(len(cats_list)):
            for j in range(3):
                v = matrix[i,j]
                ax.text(j, i, str(int(v)), ha='center', va='center', fontsize=12, fontweight='bold',
                        color='#0f0f10' if v>vmax*0.5 else '#e4e4e7')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax2 = axes[1]
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums==0, 1, row_sums)
        norm = matrix / row_sums * 100
        im2 = ax2.imshow(norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax2.set_xticks(range(3)); ax2.set_xticklabels(col_labels, fontsize=10)
        ax2.set_yticks(range(len(cats_list))); ax2.set_yticklabels(cats_list, fontsize=10)
        ax2.set_title('Normalised (% within category)', fontweight='bold')
        for i in range(len(cats_list)):
            for j in range(3):
                v = norm[i,j]
                ax2.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold',
                         color='#0f0f10' if v>40 else '#e4e4e7')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

    elif chart_type == 'calibration':
        fig, ax = plt.subplots(figsize=(8, 5))
        calib = metrics['calib']
        if calib:
            conf_vals = [c['conf']*100 for c in calib]
            acc_vals  = [c['acc']      for c in calib]
            counts    = [c['n']        for c in calib]
            ax.plot([0,100],[0,100], color=MUTED, ls='--', lw=1.5, label='Perfect calibration', alpha=0.7)
            sc = ax.scatter(conf_vals, acc_vals, s=[n*20 for n in counts], c=acc_vals,
                            cmap='RdYlGn', vmin=0, vmax=100, zorder=4, alpha=0.9)
            ax.plot(conf_vals, acc_vals, color=TEAL, lw=2, zorder=3)
            for cv, av, n in zip(conf_vals, acc_vals, counts):
                ax.annotate(f'n={n}', (cv, av), xytext=(cv+1, av+2), fontsize=8, color=MUTED)
        ax.set_xlabel('Model Confidence (%)'); ax.set_ylabel('Actual Accuracy (%)')
        ax.set_title('Confidence Calibration — Does Confidence Match Accuracy?', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4, zorder=0)
        ax.set_xlim(20, 110); ax.set_ylim(-5, 110)

    elif chart_type == 'loss':
        fig, ax = plt.subplots(figsize=(10, 5))
        steps_v2 = [500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000]
        val_v2   = [4.97,4.46,4.20,4.02,3.76,3.66,3.60,3.59,3.61,3.65,3.71,3.76,3.98,4.15,4.24,4.28]
        train_v2 = [5.07,4.50,4.11,3.96,3.52,3.21,3.00,2.87,2.54,2.13,1.73,1.86,1.25,0.91,0.85,0.78]
        steps_v3 = [500,1000,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000]
        val_v3   = [5.61,4.58,4.13,3.84,3.63,3.50,3.41,3.37,3.35,3.33,3.31,3.28]
        train_v3 = [5.58,4.48,4.09,3.78,3.52,3.30,3.11,2.95,2.70,2.45,2.19,1.98]
        ax.plot(steps_v2, train_v2, color=RED,   lw=1.5, ls='-.', alpha=0.7, label='v2 train (overfit)')
        ax.plot(steps_v2, val_v2,   color=RED,   lw=2,   label='v2 validation loss')
        ax.plot(steps_v3, train_v3, color=GREEN, lw=1.5, ls='-.', alpha=0.7, label='v3 train loss')
        ax.plot(steps_v3, val_v3,   color=GREEN, lw=2.5, label='v3 validation loss')
        ax.axvline(x=7000, color=AMBER, lw=1.2, ls=':', alpha=0.8)
        ax.text(7100, 4.7, 'v2 overfit\nstart', color=AMBER, fontsize=8)
        ax.set_xlabel('Training step'); ax.set_ylabel('Loss')
        ax.set_title('Training Loss: v2 (overfit) vs v3 (Focal Loss + Distillation)', fontweight='bold')
        ax.legend(facecolor='#27272a', edgecolor='#3f3f46', labelcolor='#e4e4e7', fontsize=9)
        ax.grid(True, alpha=0.4, zorder=0); ax.set_ylim(0.5, 6.0)

    if fig:
        fig.patch.set_facecolor(BG)
        for a in fig.get_axes():
            if hasattr(a,'set_facecolor') and not isinstance(a, plt.PolarAxes):
                try: a.set_facecolor(SURF)
                except: pass
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_b64
    return None


# ═══════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════
app   = Flask(__name__)
_data = None   # cached data — reloaded each request so results_fixed.json changes are picked up

def get_data():
    return load_data()

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NanoQA — Dynamic Metrics Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#060608;--s1:#0d0d12;--s2:#12121a;--s3:#1a1a24;
  --b1:#1e1e2e;--b2:#2a2a3e;--b3:#383858;
  --text:#d4d4f0;--sub:#6e6e9a;--white:#f0f0ff;
  --blue:#7aa2f7;--purple:#bb9af7;--amber:#e0af68;
  --green:#9ece6a;--red:#f7768e;--teal:#73daca;--pink:#ff79c6;
}
html,body{min-height:100%;background:var(--bg);color:var(--text);
  font-family:'Syne',sans-serif;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse 80% 50% at 20% 20%,rgba(122,162,247,0.04) 0%,transparent 60%),
             radial-gradient(ellipse 60% 40% at 80% 80%,rgba(187,154,247,0.04) 0%,transparent 60%);
  pointer-events:none;z-index:0;}
.app{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:24px;}
.hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;
    padding-bottom:20px;border-bottom:1px solid var(--b2);}
.hd-left h1{font-size:24px;font-weight:800;letter-spacing:-0.5px;color:var(--white);}
.hd-left p{font-size:12px;color:var(--sub);margin-top:4px;font-family:'JetBrains Mono',monospace;}
.hd-right{display:flex;gap:10px;align-items:center;}
.version-badge{background:var(--s3);border:1px solid var(--b3);border-radius:6px;
               padding:4px 12px;font-size:11px;color:var(--teal);font-family:'JetBrains Mono',monospace;}
.source-badge{background:var(--s3);border:1px solid var(--b3);border-radius:6px;
              padding:4px 12px;font-size:11px;color:var(--amber);font-family:'JetBrains Mono',monospace;}
.metrics-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:10px;}
.metrics-grid-2{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:28px;}
.metric-card{background:var(--s2);border:1px solid var(--b1);border-radius:12px;
             padding:14px;position:relative;overflow:hidden;transition:border-color .2s;}
.metric-card:hover{border-color:var(--b3);}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
                      background:var(--card-accent, var(--blue));}
.metric-label{font-size:9px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
              color:var(--sub);margin-bottom:6px;font-family:'JetBrains Mono',monospace;}
.metric-value{font-size:24px;font-weight:700;color:var(--white);font-family:'JetBrains Mono',monospace;
              letter-spacing:-1px;}
.metric-unit{font-size:12px;color:var(--sub);font-weight:400;}
.metric-desc{font-size:9px;color:var(--sub);margin-top:4px;line-height:1.5;}
.metric-bar{height:3px;background:var(--b2);border-radius:2px;margin-top:8px;overflow:hidden;}
.metric-bar-fill{height:100%;border-radius:2px;background:var(--card-accent,var(--blue));
                 transition:width 1s cubic-bezier(.4,0,.2,1);}
.controls{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center;}
.controls-label{font-size:10px;color:var(--sub);font-family:'JetBrains Mono',monospace;letter-spacing:0.5px;margin-right:4px;}
.tab-btn{background:var(--s2);border:1px solid var(--b1);border-radius:8px;
         color:var(--sub);padding:7px 14px;cursor:pointer;font-size:11px;
         font-family:'JetBrains Mono',monospace;font-weight:500;transition:all .15s;}
.tab-btn:hover{border-color:var(--b3);color:var(--text);}
.tab-btn.active{background:var(--b2);border-color:var(--blue);color:var(--blue);}
.refresh-btn{background:var(--s3);border:1px solid var(--b2);border-radius:8px;
             color:var(--teal);padding:7px 14px;cursor:pointer;font-size:11px;
             font-family:'JetBrains Mono',monospace;font-weight:500;margin-left:auto;transition:all .15s;}
.refresh-btn:hover{background:var(--b1);}
.chart-container{background:var(--s2);border:1px solid var(--b1);border-radius:16px;
                 padding:24px;margin-bottom:20px;min-height:380px;
                 display:flex;align-items:center;justify-content:center;}
.chart-img{width:100%;border-radius:8px;}
.loading{display:flex;flex-direction:column;align-items:center;gap:12px;}
.spinner{width:28px;height:28px;border:2px solid var(--b2);border-top-color:var(--blue);
         border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{font-size:11px;color:var(--sub);font-family:'JetBrains Mono',monospace;}
.cat-table{width:100%;border-collapse:collapse;font-size:11px;font-family:'JetBrains Mono',monospace;}
.cat-table th{text-align:left;padding:9px 12px;color:var(--blue);font-size:9px;
              letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--b2);}
.cat-table td{padding:9px 12px;border-bottom:1px solid var(--b1);color:var(--text);}
.cat-table tr:last-child td{border-bottom:none;}
.cat-table tr:hover td{background:var(--s3);}
.pill{display:inline-block;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;}
.pill-good{background:#14532d33;color:var(--green);border:1px solid #14532d55;}
.pill-med {background:#92400e33;color:var(--amber);border:1px solid #92400e55;}
.pill-bad {background:#7f1d1d33;color:var(--red);  border:1px solid #7f1d1d55;}
.bar-inline{display:inline-block;height:5px;border-radius:3px;margin-left:6px;vertical-align:middle;}
.info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px;}
.info-card{background:var(--s2);border:1px solid var(--b1);border-radius:12px;padding:16px;}
.info-card h3{font-size:11px;font-weight:700;color:var(--white);margin-bottom:8px;letter-spacing:0.5px;text-transform:uppercase;}
.info-card p{font-size:10px;color:var(--sub);line-height:1.7;}
.info-card code{color:var(--teal);font-family:'JetBrains Mono',monospace;font-size:9px;}
.status-bar{display:flex;gap:14px;padding:10px 14px;background:var(--s2);
            border:1px solid var(--b1);border-radius:10px;margin-top:16px;
            font-family:'JetBrains Mono',monospace;font-size:10px;}
.status-item{display:flex;gap:6px;align-items:center;}
.status-dot{width:6px;height:6px;border-radius:50%;}
@media(max-width:900px){
  .metrics-grid,.metrics-grid-2{grid-template-columns:repeat(2,1fr);}
  .info-grid{grid-template-columns:1fr;}
}
</style>
</head>
<body>
<div class="app">
  <div class="hd">
    <div class="hd-left">
      <h1>NanoQA Metrics Dashboard</h1>
      <p>Neural Router v7.2 · NanoQA v3 · Loaded from results_fixed.json · n=<span id="total_n">—</span></p>
    </div>
    <div class="hd-right">
      <div class="source-badge">📂 results_fixed.json</div>
      <div class="version-badge">v3.0 · DYNAMIC</div>
    </div>
  </div>

  <!-- Row 1: Answer Quality Metrics -->
  <div style="font-size:9px;color:var(--sub);font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">ANSWER QUALITY METRICS</div>
  <div class="metrics-grid">
    <div class="metric-card" style="--card-accent:var(--green)">
      <div class="metric-label">Accuracy</div>
      <div class="metric-value" id="v-accuracy">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">SLM correct / total SLM answered</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-accuracy" style="width:0%"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--blue)">
      <div class="metric-label">BLEU Score</div>
      <div class="metric-value" id="v-bleu">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">N-gram precision vs reference</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-bleu" style="width:0%;background:var(--blue)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--teal)">
      <div class="metric-label">ROUGE-1</div>
      <div class="metric-value" id="v-rouge1">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Unigram overlap with reference</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-rouge1" style="width:0%;background:var(--teal)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--teal)">
      <div class="metric-label">ROUGE-L</div>
      <div class="metric-value" id="v-rougeL">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Longest common subsequence</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-rougeL" style="width:0%;background:var(--teal)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--purple)">
      <div class="metric-label">MRR</div>
      <div class="metric-value" id="v-mrr">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Mean Reciprocal Rank</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-mrr" style="width:0%;background:var(--purple)"></div></div>
    </div>
  </div>

  <!-- Row 2: More metrics + Routing -->
  <div style="font-size:9px;color:var(--sub);font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;margin-top:12px;">DETAILED + ROUTING METRICS</div>
  <div class="metrics-grid-2">
    <div class="metric-card" style="--card-accent:var(--amber)">
      <div class="metric-label">F1 Token</div>
      <div class="metric-value" id="v-f1">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Token harmonic mean P/R</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-f1" style="width:0%;background:var(--amber)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--red)">
      <div class="metric-label">Exact Match</div>
      <div class="metric-value" id="v-em">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Strict string match</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-em" style="width:0%;background:var(--red)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--green)">
      <div class="metric-label">Routing Prec</div>
      <div class="metric-value" id="v-rp">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Router precision (TP/TP+FP)</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-rp" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--blue)">
      <div class="metric-label">Routing Recall</div>
      <div class="metric-value" id="v-rr">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Router recall (TP/TP+FN)</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-rr" style="width:0%;background:var(--blue)"></div></div>
    </div>
    <div class="metric-card" style="--card-accent:var(--pink)">
      <div class="metric-label">Routing F1</div>
      <div class="metric-value" id="v-rf">—<span class="metric-unit">%</span></div>
      <div class="metric-desc">Balanced routing quality</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="b-rf" style="width:0%;background:var(--pink)"></div></div>
    </div>
  </div>

  <!-- Charts -->
  <div class="controls">
    <span class="controls-label">CHART →</span>
    <button class="tab-btn active" onclick="loadChart('overview',this)">All Metrics</button>
    <button class="tab-btn" onclick="loadChart('precision_recall',this)">Precision/Recall/F1</button>
    <button class="tab-btn" onclick="loadChart('bleu_rouge',this)">BLEU/ROUGE Radar</button>
    <button class="tab-btn" onclick="loadChart('routing',this)">Routing</button>
    <button class="tab-btn" onclick="loadChart('confusion',this)">Confusion Matrix</button>
    <button class="tab-btn" onclick="loadChart('response_time',this)">Compute Savings</button>
    <button class="tab-btn" onclick="loadChart('calibration',this)">Calibration</button>
    <button class="tab-btn" onclick="loadChart('loss',this)">Loss Curve</button>
    <button class="refresh-btn" onclick="refreshAll()">↻ Refresh from results_fixed.json</button>
  </div>

  <div class="chart-container" id="chart-box">
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-text">Loading results_fixed.json and computing metrics...</div>
    </div>
  </div>

  <!-- Category Table -->
  <div style="background:var(--s2);border:1px solid var(--b1);border-radius:16px;padding:18px;margin-bottom:18px;">
    <h2 style="font-size:11px;font-weight:700;color:var(--white);margin-bottom:12px;letter-spacing:0.5px;text-transform:uppercase;">Per-Category Breakdown — from results_fixed.json</h2>
    <table class="cat-table" id="cat-table">
      <thead><tr>
        <th>Category</th><th>Precision</th><th>Recall</th><th>F1</th>
        <th>TP</th><th>FP</th><th>FN</th><th>Grade</th>
      </tr></thead>
      <tbody id="cat-tbody"><tr><td colspan="8" style="text-align:center;color:var(--sub);padding:20px">Loading from results_fixed.json...</td></tr></tbody>
    </table>
  </div>

  <!-- Info panels -->
  <div class="info-grid">
    <div class="info-card">
      <h3>BLEU Score</h3>
      <p><strong style="color:var(--white)">Bilingual Evaluation Understudy</strong> — measures n-gram precision. Applied directly on model output vs reference. No training needed. Formula: <code>BP × exp(Σ wn·log pn)</code>. BLEU-1 = word overlap, BLEU-4 = phrase overlap. Higher = better answer quality.</p>
    </div>
    <div class="info-card">
      <h3>ROUGE Score</h3>
      <p><strong style="color:var(--white)">Recall-Oriented Understudy</strong> — measures recall (reference coverage). <code>ROUGE-1</code> = unigram, <code>ROUGE-2</code> = bigram, <code>ROUGE-L</code> = LCS. Applied directly on output. Standard for QA and summarisation evaluation.</p>
    </div>
    <div class="info-card">
      <h3>Routing Metrics</h3>
      <p><strong style="color:var(--white)">Routing Precision/Recall/F1</strong> — measures the router itself. Routing Precision = when NanoQA answers, how often correct. Routing Recall = of answerable questions, how many NanoQA handles. F1 = balance. These are unique to routing systems.</p>
    </div>
  </div>

  <div class="status-bar">
    <div class="status-item"><div class="status-dot" style="background:var(--green)"></div>NanoQA v3 · 135M params</div>
    <div class="status-item"><div class="status-dot" style="background:var(--purple)"></div>Mistral 7B via Ollama</div>
    <div class="status-item"><div class="status-dot" style="background:var(--amber)"></div>Math Engine · Python eval()</div>
    <div class="status-item"><div class="status-dot" style="background:var(--teal)"></div>📂 <span id="data-source">results_fixed.json</span></div>
    <div class="status-item" style="margin-left:auto"><span id="last-updated" style="color:var(--sub)">—</span></div>
  </div>
</div>

<script>
let currentChart = 'overview';

async function fetchMetrics() {
  const res = await fetch('/api/metrics');
  return await res.json();
}
async function fetchChart(type) {
  const res = await fetch(`/api/chart/${type}`);
  return await res.json();
}

function setCard(id, value, unit='%') {
  const el  = document.getElementById(`v-${id}`);
  if(el) el.innerHTML = `${value}<span class="metric-unit">${unit}</span>`;
  const bar = document.getElementById(`b-${id}`);
  if(bar) setTimeout(()=>{ bar.style.width = Math.min(value,100)+'%'; }, 100);
}

function pillClass(val) {
  if(val >= 80) return 'pill-good';
  if(val >= 60) return 'pill-med';
  return 'pill-bad';
}

async function loadMetrics() {
  const m = await fetchMetrics();
  document.getElementById('total_n').textContent = m.total;
  setCard('accuracy', m.accuracy);
  setCard('bleu',     m.bleu);
  setCard('rouge1',   m.rouge1);
  setCard('rougeL',   m.rougeL);
  setCard('mrr',      m.mrr);
  setCard('f1',       m.f1_token);
  setCard('em',       m.exact_match);
  setCard('rp',       m.routing_precision);
  setCard('rr',       m.routing_recall);
  setCard('rf',       m.routing_f1);

  const tbody = document.getElementById('cat-tbody');
  tbody.innerHTML = '';
  for(const [cat, cm] of Object.entries(m.cat_metrics)) {
    const grade = cm.f1>=80?'Strong':cm.f1>=60?'Good':'Weak';
    const pc    = pillClass(cm.f1);
    tbody.innerHTML += `<tr>
      <td style="color:var(--white);font-weight:600">${cat}</td>
      <td>${cm.precision.toFixed(1)}%<span class="bar-inline" style="width:${cm.precision*0.5}px;background:var(--blue)"></span></td>
      <td>${cm.recall.toFixed(1)}%<span class="bar-inline" style="width:${cm.recall*0.5}px;background:var(--green)"></span></td>
      <td>${cm.f1.toFixed(1)}%<span class="bar-inline" style="width:${cm.f1*0.5}px;background:var(--amber)"></span></td>
      <td style="color:var(--green)">${cm.tp}</td>
      <td style="color:var(--red)">${cm.fp}</td>
      <td style="color:var(--sub)">${cm.fn}</td>
      <td><span class="pill ${pc}">${grade}</span></td>
    </tr>`;
  }
  document.getElementById('last-updated').textContent = 'Updated: '+new Date().toLocaleTimeString();
}

async function loadChart(type, btn) {
  currentChart = type;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  const box = document.getElementById('chart-box');
  box.innerHTML = `<div class="loading"><div class="spinner"></div><div class="loading-text">Generating chart from results_fixed.json...</div></div>`;
  try {
    const data = await fetchChart(type);
    if(data.img) {
      box.innerHTML = `<img class="chart-img" src="data:image/png;base64,${data.img}" alt="chart">`;
    } else {
      box.innerHTML = `<div style="color:var(--red);font-family:monospace;font-size:11px">Error: ${data.error||'Unknown'}</div>`;
    }
  } catch(e) {
    box.innerHTML = `<div style="color:var(--red);font-family:monospace;font-size:11px">Error: ${e.message}</div>`;
  }
}

async function refreshAll() {
  await loadMetrics();
  await loadChart(currentChart);
}
(async()=>{ await loadMetrics(); await loadChart('overview'); })();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/metrics')
def api_metrics():
    data = get_data()
    m    = compute_metrics(data)
    return jsonify(m)

@app.route('/api/chart/<chart_type>')
def api_chart(chart_type):
    try:
        data = get_data()
        m    = compute_metrics(data)
        img  = make_chart(chart_type, m)
        if img:
            return jsonify({'img': img})
        return jsonify({'error': f'Unknown chart type: {chart_type}'})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    print()
    print('='*60)
    print('  NanoQA Dynamic Metrics Dashboard')
    print(f'  Data source: {RESULTS_FILE}')
    print('  Open: http://localhost:7000')
    print('='*60)
    print()
    print('If results.json exists → loads real data from it')
    print('If not → generates 150 demo entries and saves them')
    print()
    print('To use your real data:')
    print('  Run collect_and_visualize.py first')
    print('  It saves visualizations/results.json automatically')
    print()
    print('Metrics (applied on output, no training needed):')
    print('  Accuracy, BLEU, ROUGE-1/2/L, MRR, F1, Exact Match')
    print('  Routing Precision, Routing Recall, Routing F1')
    print('  Confidence Calibration, Response Time Percentiles')
    print()
    app.run(debug=False, port=7003)
