"""
Neural Router v7.2 — NanoQA v2 + fuzzy typo fixing + normalization
Run:
    Terminal 1: ollama serve
    Terminal 2: cd ~/slm_project && source venv/bin/activate && python3 new_app_2.py
"""

import os, sys, re, math, socket, difflib, torch, torch.nn.functional as F, ollama
from flask import Flask, request, jsonify, render_template_string
from transformers import GPT2Tokenizer

# ── Load NanoQA v2 ────────────────────────────────────────────
MODEL_PATH = "./models/nanoqa_v2"
sys.path.insert(0, os.path.abspath(MODEL_PATH))
from nanoqa_arch import NanoQAConfig, NanoQAModel

print("Loading NanoQA v2...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
config    = NanoQAConfig.from_pretrained(MODEL_PATH)
model     = NanoQAModel(config)

import safetensors.torch
state_dict = safetensors.torch.load_file(os.path.join(MODEL_PATH, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model  = model.to(device)
model.eval()
print(f"NanoQA v2 loaded on {device}")

# ── Math engine ───────────────────────────────────────────────
SAFE_MATH = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "sinh": math.sinh, "cosh": math.cosh,
    "tanh": math.tanh, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "abs": abs,
    "pow": pow, "round": round, "factorial": math.factorial,
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "ceil": math.ceil, "floor": math.floor,
    "degrees": math.degrees, "radians": math.radians,
    "hypot": math.hypot, "gcd": math.gcd,
    "comb": math.comb, "perm": math.perm,
}

def evaluate_math(expr):
    expr = expr.strip()
    expr = re.sub(r'(what is|calculate|compute|evaluate|solve|whats)', '', expr, flags=re.IGNORECASE).strip()
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = expr.replace("squared", "**2").replace("cubed", "**3")
    expr = re.sub(r'(\d)\s*x\s*(\d)', r'\1*\2', expr)
    expr = expr.strip("? \n")
    if not re.search(r'[\d\+\-\*\/\(\)\^]', expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, SAFE_MATH)
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return str(round(result, 8))
        return str(result)
    except:
        return None

# ── Normalization + Typo Fixing ───────────────────────────────

# Vocabulary of words NanoQA was trained on
# Fuzzy matching will correct user typos to these
KNOWN_WORDS = [
    # question words
    "who", "what", "when", "where", "which", "how", "why",
    "is", "are", "was", "were", "did", "does", "do",
    # verbs
    "wrote", "invented", "discovered", "painted", "built",
    "designed", "created", "founded", "established", "composed",
    "sculpted", "directed", "developed",
    # adjectives
    "largest", "smallest", "fastest", "slowest", "oldest",
    "biggest", "tallest", "deepest", "longest", "shortest",
    "hottest", "coldest", "heaviest", "lightest", "first",
    # science
    "mitochondria", "photosynthesis", "gravity", "evolution",
    "penicillin", "radioactivity", "relativity", "quantum",
    "chromosome", "electron", "neutron", "proton", "nucleus",
    "molecule", "atom", "oxygen", "hydrogen", "carbon",
    # people
    "shakespeare", "newton", "einstein", "darwin", "fleming",
    "galileo", "copernicus", "aristotle", "plato", "socrates",
    "napoleon", "caesar", "cleopatra", "beethoven", "mozart",
    "michelangelo", "leonardo", "picasso", "tolkien", "rowling",
    "orwell", "dickens", "shakespeare", "hemingway", "tolstoy",
    # places
    "france", "india", "japan", "germany", "china", "russia",
    "america", "england", "italy", "spain", "australia",
    "capital", "paris", "london", "tokyo", "delhi", "beijing",
    "ocean", "pacific", "atlantic", "indian", "arctic",
    "everest", "sahara", "amazon", "himalaya", "antarctica",
    # books / works
    "hamlet", "macbeth", "othello", "odyssey", "iliad",
    "romeo", "juliet", "gatsby", "frankenstein", "dracula",
    # technology
    "computer", "internet", "telephone", "airplane", "radio",
    "television", "electricity", "battery", "laser", "radar",
    "python", "javascript", "algorithm", "database", "network",
    # general
    "animal", "planet", "country", "language", "currency",
    "element", "chemical", "formula", "theory", "theorem",
    "national", "official", "population", "continent",
]

def fix_typos(text):
    """Fix typos by fuzzy matching against known vocabulary."""
    words = text.split()
    fixed = []
    for word in words:
        # skip short words, numbers, punctuation
        if len(word) <= 3 or word.isdigit():
            fixed.append(word)
            continue
        # try to find close match
        matches = difflib.get_close_matches(
            word,
            KNOWN_WORDS,
            n=1,
            cutoff=0.82  # 82% similarity threshold
        )
        if matches:
            fixed.append(matches[0])
        else:
            fixed.append(word)
    return ' '.join(fixed)

def normalize(text):
    """
    Normalize user input before sending to NanoQA.
    Handles: case, contractions, punctuation, typos.
    """
    # lowercase
    text = text.lower().strip()

    # fix contractions
    text = text.replace("what's", "what is")
    text = text.replace("who's",  "who is")
    text = text.replace("where's","where is")
    text = text.replace("when's", "when is")
    text = text.replace("how's",  "how is")
    text = text.replace("it's",   "it is")
    text = text.replace("that's", "that is")
    text = text.replace("there's","there is")
    text = text.replace("he's",   "he is")
    text = text.replace("she's",  "she is")
    text = text.replace("they're","they are")
    text = text.replace("i'm",    "i am")
    text = text.replace("can't",  "cannot")
    text = text.replace("won't",  "will not")
    text = text.replace("don't",  "do not")
    text = text.replace("didn't", "did not")
    text = text.replace("wasn't", "was not")
    text = text.replace("isn't",  "is not")
    text = text.replace("'s",     "s")
    text = text.replace("'",      "")

    # normalize whitespace
    text = ' '.join(text.split())

    # fix typos
    text = fix_typos(text)

    return text

# ── SLM generation ────────────────────────────────────────────
def slm_generate_with_confidence(question):
    # always normalize before model
    question  = normalize(question)
    prompt    = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids[:, -model.config.max_seq_len:]

    generated_tokens = []
    token_probs      = []

    with torch.no_grad():
        for _ in range(20):
            logits    = model(input_ids).logits[:, -1, :]
            raw_probs = F.softmax(logits, dim=-1)
            scaled    = logits / 0.3
            v, _      = torch.topk(scaled, min(10, scaled.size(-1)))
            scaled[scaled < v[:, [-1]]] = float('-inf')
            probs     = F.softmax(scaled, dim=-1)
            next_id   = torch.multinomial(probs, 1)
            token_probs.append(raw_probs[0, next_id.item()].item())
            generated_tokens.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    ans = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if "." in ans:
        ans = ans.split(".")[0].strip() + "."
    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
    return ans, confidence

def is_garbage(text):
    words  = text.split()
    unique = set(w.lower() for w in words)
    if len(words) < 2:                      return True
    if len(words) > 20:                     return True
    if len(unique) == 1:                    return True
    if len(words) > 4 and len(unique) < 3:  return True
    if re.match(r'^[\d\s\.\,]+$', text):    return True
    return False

def llm_generate(question):
    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": question}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Ollama error: {str(e)}. Is 'ollama serve' running?"

# ── Router ────────────────────────────────────────────────────
def router(question):
    # normalize for model use
    q_norm = normalize(question)

    # 1. Math — use normalized
    math_ans = evaluate_math(q_norm)
    if math_ans:
        return "Math", math_ans

    # 2. SLM — always uses normalized internally
    slm_ans, confidence = slm_generate_with_confidence(q_norm)
    if not is_garbage(slm_ans) and confidence >= 0.60:
        return "SLM", slm_ans

    # retry once if borderline
    if not is_garbage(slm_ans) and confidence >= 0.45:
        slm_ans2, confidence2 = slm_generate_with_confidence(q_norm)
        if not is_garbage(slm_ans2) and confidence2 >= 0.60:
            return "SLM", slm_ans2

    # 3. LLM — use original question (Mistral handles any phrasing)
    return "LLM", llm_generate(question)


app = Flask(__name__)

# ── CHAT HTML ─────────────────────────────────────────────────
CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Neural Router</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  :root{
    --bg:#080808;--s1:#101010;--s2:#161616;
    --b1:#1c1c1c;--b2:#262626;--b3:#363636;
    --text:#d8d8d8;--sub:#555;--sub2:#333;--white:#f0f0f0;
    --blue:#4a9eff;--purple:#a78bfa;--amber:#f0a500;--green:#3dd68c;--red:#ff5f5f;
  }
  html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;overflow:hidden;}
  .app{height:100vh;display:flex;flex-direction:column;}
  .hd{height:50px;flex-shrink:0;display:flex;align-items:center;padding:0 20px;gap:12px;border-bottom:1px solid var(--b1);background:var(--s1);}
  .hd-logo{display:flex;align-items:center;gap:8px;}
  .hd-icon{width:22px;height:22px;border-radius:5px;background:linear-gradient(135deg,#4a9eff22,#3dd68c22);border:1px solid #4a9eff33;display:grid;place-items:center;font-size:11px;}
  .hd-name{font-size:13px;font-weight:500;color:var(--white);letter-spacing:-0.3px;}
  .hd-sep{width:1px;height:16px;background:var(--b2);}
  .hd-meta{font-size:11px;color:var(--sub);}
  .hd-right{margin-left:auto;display:flex;gap:6px;align-items:center;}
  .badge{font-size:9px;font-weight:500;padding:2px 7px;border-radius:3px;letter-spacing:0.4px;text-transform:uppercase;}
  .badge-slm   {color:var(--blue);  background:#4a9eff0f;border:1px solid #4a9eff22;}
  .badge-llm   {color:var(--purple);background:#a78bfa0f;border:1px solid #a78bfa22;}
  .badge-math  {color:var(--amber); background:#f0a5000f;border:1px solid #f0a50022;}
  .calc-btn{display:flex;align-items:center;gap:5px;padding:4px 10px;border-radius:4px;background:var(--s2);border:1px solid var(--b2);color:var(--sub);font-size:10px;font-weight:500;cursor:pointer;text-decoration:none;transition:all .12s;letter-spacing:0.3px;}
  .calc-btn:hover{border-color:var(--b3);color:var(--text);}
  .calc-btn svg{width:12px;height:12px;fill:currentColor;}
  .chat{flex:1;overflow-y:auto;}
  .chat::-webkit-scrollbar{width:2px;}
  .chat::-webkit-scrollbar-thumb{background:var(--b2);}
  .msgs{max-width:700px;margin:0 auto;padding:36px 20px 16px;display:flex;flex-direction:column;gap:28px;min-height:100%;}
  .welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:48px 0;animation:rise .5s ease both;}
  @keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
  .w-ring{width:52px;height:52px;border-radius:14px;border:1px solid var(--b2);display:grid;place-items:center;margin-bottom:20px;background:var(--s1);position:relative;}
  .w-ring::before{content:'';position:absolute;inset:-1px;border-radius:14px;background:linear-gradient(135deg,#4a9eff33 0%,transparent 50%,#3dd68c22 100%);z-index:-1;}
  .w-ring svg{width:20px;height:20px;}
  .welcome h1{font-size:20px;font-weight:400;color:var(--white);letter-spacing:-0.5px;margin-bottom:7px;}
  .welcome p{font-size:12px;color:var(--sub);line-height:1.9;margin-bottom:30px;}
  .chips{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;max-width:540px;}
  .chip{padding:5px 11px;border:1px solid var(--b1);border-radius:4px;font-size:11px;color:var(--sub);cursor:pointer;transition:all .13s;background:transparent;font-family:'Inter',sans-serif;}
  .chip:hover{border-color:var(--b3);color:var(--text);background:var(--s1);}
  .msg{display:flex;gap:12px;animation:rise .22s ease both;}
  .msg.user{flex-direction:row-reverse;}
  .av{width:26px;height:26px;border-radius:6px;display:grid;place-items:center;font-size:9px;font-weight:500;flex-shrink:0;margin-top:2px;letter-spacing:0.3px;}
  .msg.user .av{background:var(--s2);border:1px solid var(--b1);color:var(--sub);}
  .msg.assistant .av{background:var(--s1);border:1px solid var(--b1);color:var(--sub);}
  .mc{max-width:590px;display:flex;flex-direction:column;gap:5px;}
  .msg.user .mc{align-items:flex-end;}
  .route-label{display:inline-flex;align-items:center;gap:5px;font-size:9px;font-weight:500;letter-spacing:0.5px;text-transform:uppercase;padding-left:1px;}
  .route-dot{width:4px;height:4px;border-radius:50%;}
  .rl-slm   {color:var(--blue);}   .rl-slm   .route-dot{background:var(--blue);}
  .rl-llm   {color:var(--purple);} .rl-llm   .route-dot{background:var(--purple);}
  .rl-math  {color:var(--amber);}  .rl-math  .route-dot{background:var(--amber);}
  .rl-error {color:var(--red);}    .rl-error .route-dot{background:var(--red);}
  .rl-default{color:var(--sub);}
  .conf{font-size:9px;color:var(--sub2);margin-top:2px;padding-left:1px;}
  .bub{padding:11px 15px;border-radius:9px;font-size:13px;line-height:1.75;word-break:break-word;}
  .msg.user      .bub{background:var(--s2);border:1px solid var(--b1);color:#c8c8c8;border-top-right-radius:2px;}
  .msg.assistant .bub{background:var(--s1);border:1px solid var(--b1);color:#b8b8b8;border-top-left-radius:2px;}
  .dots{display:flex;gap:4px;padding:3px 0;}
  .dots span{width:4px;height:4px;border-radius:50%;background:var(--b3);animation:bounce 1.4s ease infinite;}
  .dots span:nth-child(2){animation-delay:.18s;}
  .dots span:nth-child(3){animation-delay:.36s;}
  @keyframes bounce{0%,60%,100%{transform:translateY(0);background:var(--b3)}30%{transform:translateY(-5px);background:var(--sub);}}
  .inp-zone{flex-shrink:0;padding:10px 20px 18px;border-top:1px solid var(--b1);background:var(--s1);}
  .inp-inner{max-width:700px;margin:0 auto;}
  .inp-box{display:flex;align-items:flex-end;gap:8px;background:var(--s2);border:1px solid var(--b1);border-radius:9px;padding:10px 13px;transition:border-color .15s;}
  .inp-box:focus-within{border-color:var(--b3);}
  textarea{flex:1;background:transparent;border:none;outline:none;color:var(--text);font-size:13px;resize:none;max-height:100px;line-height:1.65;font-family:'Inter',sans-serif;}
  textarea::placeholder{color:var(--sub2);}
  .go{width:30px;height:30px;background:var(--white);border:none;border-radius:6px;cursor:pointer;display:grid;place-items:center;flex-shrink:0;transition:all .12s;}
  .go:hover{opacity:.8;transform:scale(1.05);}
  .go:disabled{opacity:.15;cursor:not-allowed;transform:none;}
  .go svg{width:12px;height:12px;fill:#000;}
  .hint{text-align:center;margin-top:7px;font-size:9px;color:var(--sub2);letter-spacing:0.3px;}
</style>
</head>
<body>
<div class="app">
  <div class="hd">
    <div class="hd-logo">
      <div class="hd-icon">⚡</div>
      <span class="hd-name">Neural Router</span>
    </div>
    <div class="hd-sep"></div>
    <span class="hd-meta">NanoQA v2 · Mistral 7B · v7.2</span>
    <div class="hd-right">
      <span class="badge badge-slm">SLM</span>
      <span class="badge badge-llm">LLM</span>
      <span class="badge badge-math">Math</span>
      <a href="/calc" class="calc-btn">
        <svg viewBox="0 0 24 24"><path d="M4 2h16a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm2 4v2h2V6H6zm4 0v2h2V6h-2zm4 0v2h4V6h-4zM6 10v2h2v-2H6zm4 0v2h2v-2h-2zm4 0v2h4v-2h-4zM6 14v2h2v-2H6zm4 0v2h2v-2h-2zm4 0v8h4v-8h-4zM6 18v2h2v-2H6zm4 0v2h2v-2h-2z"/></svg>
        Calculator
      </a>
    </div>
  </div>

  <div class="chat" id="chat">
    <div class="msgs" id="msgs"></div>
  </div>

  <div class="inp-zone">
    <div class="inp-inner">
      <div class="inp-box">
        <textarea id="inp" rows="1" placeholder="Ask anything… or type /calc"
          onkeydown="onKey(event)" oninput="rsz(this)"></textarea>
        <button class="go" id="btn" onclick="send()">
          <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
        </button>
      </div>
      <div class="hint">enter to send · shift+enter newline · /calc for calculator</div>
    </div>
  </div>
</div>

<script>
const chat=document.getElementById('chat');
const msgs=document.getElementById('msgs');
const inp=document.getElementById('inp');
const btn=document.getElementById('btn');
const STORE_KEY='nr_chat_history';

function loadHistory(){
  try{
    const saved=sessionStorage.getItem(STORE_KEY);
    if(saved){
      const history=JSON.parse(saved);
      if(history.length>0){
        history.forEach(item=>renderMsg(item.role,item.text,item.route,item.confidence,false));
        chat.scrollTop=chat.scrollHeight;
        return true;
      }
    }
  }catch(e){}
  return false;
}

function saveMsg(role,text,route,confidence){
  try{
    const saved=sessionStorage.getItem(STORE_KEY);
    const history=saved?JSON.parse(saved):[];
    history.push({role,text,route:route||null,confidence:confidence||null});
    if(history.length>100)history.splice(0,history.length-100);
    sessionStorage.setItem(STORE_KEY,JSON.stringify(history));
  }catch(e){}
}

function routeClass(r){
  if(!r)return'rl-default';
  const s=r.toLowerCase();
  if(s==='slm')return'rl-slm';
  if(s==='llm')return'rl-llm';
  if(s==='math')return'rl-math';
  if(s==='error')return'rl-error';
  return'rl-default';
}

function renderMsg(role,text,route,confidence,animate){
  const m=document.createElement('div');
  m.className=`msg ${role}`;
  const av=document.createElement('div');av.className='av';
  av.textContent=role==='user'?'you':'nr';
  const mc=document.createElement('div');mc.className='mc';
  if(role==='assistant'&&route){
    const rl=document.createElement('div');
    rl.className=`route-label ${routeClass(route)}`;
    rl.innerHTML=`<div class="route-dot"></div>${route}`;
    mc.appendChild(rl);
  }
  const b=document.createElement('div');b.className='bub';b.textContent=text;
  mc.appendChild(b);
  if(role==='assistant'&&confidence){
    const cf=document.createElement('div');cf.className='conf';
    cf.textContent=`confidence ${(confidence*100).toFixed(0)}%`;
    mc.appendChild(cf);
  }
  m.appendChild(av);m.appendChild(mc);
  msgs.appendChild(m);
}

function addMsg(role,text,route,confidence){
  renderMsg(role,text,route,confidence,true);
  saveMsg(role,text,route,confidence);
  chat.scrollTop=chat.scrollHeight;
}

function addTyping(){
  const m=document.createElement('div');m.className='msg assistant';m.id='typing';
  m.innerHTML='<div class="av">nr</div><div class="mc"><div class="bub"><div class="dots"><span></span><span></span><span></span></div></div></div>';
  msgs.appendChild(m);chat.scrollTop=chat.scrollHeight;
}

function showWelcome(){
  msgs.innerHTML=`
    <div class="welcome" id="welcome">
      <div class="w-ring">
        <svg viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="1.5" stroke-linecap="round">
          <polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
      </div>
      <h1>What do you want to know?</h1>
      <p>NanoQA handles simple queries · Complex ones go to Mistral 7B<br/>Handles typos and case variations automatically</p>
      <div class="chips">
        <div class="chip" onclick="ex('Who wrote Romeo and Juliet?')">Who wrote Romeo and Juliet?</div>
        <div class="chip" onclick="ex('Whats the capital of India?')">What's capital of India?</div>
        <div class="chip" onclick="ex('Who discoverd penicillin?')">Who discoverd penicillin?</div>
        <div class="chip" onclick="ex('WHAT IS THE LARGEST OCEAN')">WHAT IS THE LARGEST OCEAN</div>
        <div class="chip" onclick="ex('Explain quantum entanglement in detail')">Quantum entanglement</div>
        <div class="chip" onclick="ex('Who painted the Mona Lisa?')">Who painted Mona Lisa?</div>
        <div class="chip" onclick="ex('sqrt(144)')">sqrt(144)</div>
        <div class="chip" onclick="ex('/calc')">Open Calculator</div>
      </div>
    </div>`;
}

const hasHistory=loadHistory();
if(!hasHistory)showWelcome();

function rsz(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,100)+'px';}
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function ex(t){inp.value=t;send();}

async function send(){
  const text=inp.value.trim();
  if(!text)return;
  if(text==='/calc'){window.location.href='/calc';return;}
  document.getElementById('welcome')?.remove();
  btn.disabled=true;inp.value='';inp.style.height='auto';
  addMsg('user',text,null,null);addTyping();
  try{
    const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
    const data=await res.json();
    document.getElementById('typing')?.remove();
    addMsg('assistant',data.answer,data.route,data.confidence);
  }catch(e){
    document.getElementById('typing')?.remove();
    addMsg('assistant','Error — is Ollama running?','error',null);
  }
  btn.disabled=false;inp.focus();
}
</script>
</body>
</html>"""

# ── CALCULATOR HTML ───────────────────────────────────────────
CALC_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Scientific Calculator</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  :root{
    --bg:#050505;--s1:#0c0c0c;--s2:#121212;--s3:#181818;
    --b1:#1f1f1f;--b2:#2a2a2a;--b3:#363636;
    --text:#e0e0e0;--sub:#555;--white:#f5f5f5;
    --blue:#4a9eff;--amber:#f0a500;--green:#3dd68c;--red:#ff5f5f;--purple:#c084fc;
  }
  html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;}
  .back{position:fixed;top:16px;left:16px;display:flex;align-items:center;gap:6px;color:var(--sub);font-size:11px;text-decoration:none;padding:6px 10px;border-radius:5px;border:1px solid var(--b1);background:var(--s1);transition:all .12s;font-weight:500;}
  .back:hover{color:var(--text);border-color:var(--b2);}
  .back svg{width:12px;height:12px;fill:currentColor;}
  .calc{width:380px;background:var(--s1);border:1px solid var(--b1);border-radius:16px;overflow:hidden;box-shadow:0 32px 64px rgba(0,0,0,0.6);}
  .display{padding:20px 20px 14px;background:var(--bg);border-bottom:1px solid var(--b1);min-height:110px;display:flex;flex-direction:column;justify-content:flex-end;position:relative;}
  .display-mode{position:absolute;top:12px;left:16px;font-size:9px;color:var(--sub);font-family:'Inter',sans-serif;letter-spacing:1px;text-transform:uppercase;font-weight:500;}
  .display-history{font-size:11px;color:var(--sub);font-family:'DM Mono',monospace;text-align:right;min-height:16px;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .display-expr{font-size:15px;color:#777;font-family:'DM Mono',monospace;text-align:right;min-height:22px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:4px;font-weight:300;}
  .display-main{font-size:38px;font-weight:300;color:var(--white);font-family:'DM Mono',monospace;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;line-height:1.1;letter-spacing:-1px;}
  .display-main.small{font-size:22px;letter-spacing:-0.5px;}
  .display-main.error{font-size:18px;color:var(--red);letter-spacing:0;}
  .mode-tabs{display:flex;border-bottom:1px solid var(--b1);background:var(--s2);}
  .mode-tab{flex:1;padding:8px 4px;font-size:10px;font-weight:500;color:var(--sub);cursor:pointer;border:none;background:transparent;font-family:'Inter',sans-serif;letter-spacing:0.4px;text-transform:uppercase;border-bottom:2px solid transparent;transition:all .12s;}
  .mode-tab.active{color:var(--blue);border-bottom-color:var(--blue);}
  .mode-tab:hover:not(.active){color:var(--text);}
  .btns{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--b1);padding:1px;}
  .btns.basic{grid-template-columns:repeat(4,1fr);}
  .btn{background:var(--s2);border:none;color:var(--text);font-size:13px;font-weight:400;font-family:'Inter',sans-serif;padding:0;height:52px;cursor:pointer;transition:all .08s;display:flex;align-items:center;justify-content:center;user-select:none;position:relative;overflow:hidden;letter-spacing:0.2px;}
  .btn:hover{background:var(--s3);}
  .btn:active{background:var(--b2);transform:scale(0.97);}
  .btn.fn{background:var(--s1);color:#888;font-size:11px;}
  .btn.fn:hover{background:#161616;color:var(--text);}
  .btn.op{color:var(--amber);}
  .btn.eq{background:var(--blue);color:#000;font-weight:600;}
  .btn.eq:hover{background:#6bb3ff;}
  .btn.clear{color:var(--red);}
  .btn.num{color:var(--white);font-size:15px;font-weight:400;}
  .btn.const{color:var(--green);font-size:13px;font-family:'DM Mono',monospace;}
  .btn.trig{color:var(--purple);font-size:11px;}
  .btn.mem{color:#666;font-size:11px;}
  .btn.span2{grid-column:span 2;}
  .ripple{position:absolute;border-radius:50%;background:rgba(255,255,255,0.08);transform:scale(0);animation:ripple .4s ease;pointer-events:none;}
  @keyframes ripple{to{transform:scale(4);opacity:0;}}
  .hist-panel{display:none;max-height:200px;overflow-y:auto;border-top:1px solid var(--b1);background:var(--bg);}
  .hist-panel.open{display:block;}
  .hist-item{padding:8px 16px;border-bottom:1px solid var(--b1);cursor:pointer;transition:background .1s;}
  .hist-item:hover{background:var(--s1);}
  .hi-expr{color:var(--sub);font-size:10px;font-family:'DM Mono',monospace;}
  .hi-res{color:var(--white);font-size:13px;font-family:'DM Mono',monospace;font-weight:400;}
  .hist-empty{padding:16px;text-align:center;color:var(--sub);font-size:11px;}
  .calc-footer{padding:8px 14px;border-top:1px solid var(--b1);display:flex;justify-content:space-between;align-items:center;background:var(--s2);}
  .calc-footer span{font-size:9px;color:var(--sub);font-family:'DM Mono',monospace;}
  .hist-toggle{font-size:9px;color:var(--sub);cursor:pointer;background:none;border:none;font-family:'Inter',sans-serif;padding:2px 6px;border-radius:3px;border:1px solid var(--b1);transition:all .1s;font-weight:500;}
  .hist-toggle:hover{color:var(--text);border-color:var(--b2);}
</style>
</head>
<body>
<a href="/" class="back">
  <svg viewBox="0 0 24 24"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
  Back to Chat
</a>
<div class="calc">
  <div class="display">
    <div class="display-mode" id="modeLabel">DEG</div>
    <div class="display-history" id="histLine"></div>
    <div class="display-expr" id="exprLine"></div>
    <div class="display-main" id="mainDisplay">0</div>
  </div>
  <div class="mode-tabs">
    <button class="mode-tab active" onclick="switchPanel('sci')">Scientific</button>
    <button class="mode-tab" onclick="switchPanel('basic')">Basic</button>
    <button class="mode-tab" onclick="toggleHist()">History</button>
  </div>
  <div id="panel-sci">
    <div class="btns">
      <button class="btn mem"   onclick="mem('MC')">MC</button>
      <button class="btn mem"   onclick="mem('MR')">MR</button>
      <button class="btn mem"   onclick="mem('MS')">MS</button>
      <button class="btn mem"   onclick="mem('M+')">M+</button>
      <button class="btn clear" onclick="clearAll()">AC</button>
      <button class="btn trig"  onclick="fn('sin(')">sin</button>
      <button class="btn trig"  onclick="fn('cos(')">cos</button>
      <button class="btn trig"  onclick="fn('tan(')">tan</button>
      <button class="btn trig"  onclick="fn('asin(')">sin⁻¹</button>
      <button class="btn trig"  onclick="fn('acos(')">cos⁻¹</button>
      <button class="btn trig"  onclick="fn('atan(')">tan⁻¹</button>
      <button class="btn fn"    onclick="fn('log10(')">log</button>
      <button class="btn fn"    onclick="fn('log(')">ln</button>
      <button class="btn fn"    onclick="fn('log2(')">log₂</button>
      <button class="btn fn"    onclick="fn('exp(')">eˣ</button>
      <button class="btn fn"    onclick="fn('sqrt(')">√</button>
      <button class="btn fn"    onclick="append('**2')">x²</button>
      <button class="btn fn"    onclick="append('**3')">x³</button>
      <button class="btn fn"    onclick="append('**(1/3)')">∛</button>
      <button class="btn fn"    onclick="append('**')">xʸ</button>
      <button class="btn fn"    onclick="fn('factorial(')">n!</button>
      <button class="btn const" onclick="append('pi')">π</button>
      <button class="btn const" onclick="append('e')">e</button>
      <button class="btn fn"    onclick="append('%')">mod</button>
      <button class="btn fn"    onclick="toggleDeg()" id="degBtn">DEG</button>
      <button class="btn fn"    onclick="fn('abs(')">|x|</button>
      <button class="btn fn"    onclick="fn('ceil(')">⌈x⌉</button>
      <button class="btn fn"    onclick="fn('floor(')">⌊x⌋</button>
      <button class="btn op"    onclick="append('(')">(</button>
      <button class="btn op"    onclick="append(')')">)</button>
      <button class="btn num"   onclick="append('7')">7</button>
      <button class="btn num"   onclick="append('8')">8</button>
      <button class="btn num"   onclick="append('9')">9</button>
      <button class="btn op"    onclick="del()">⌫</button>
      <button class="btn op"    onclick="append('/')">÷</button>
      <button class="btn num"   onclick="append('4')">4</button>
      <button class="btn num"   onclick="append('5')">5</button>
      <button class="btn num"   onclick="append('6')">6</button>
      <button class="btn op"    onclick="append('*')">×</button>
      <button class="btn op"    onclick="append('-')">−</button>
      <button class="btn num"   onclick="append('1')">1</button>
      <button class="btn num"   onclick="append('2')">2</button>
      <button class="btn num"   onclick="append('3')">3</button>
      <button class="btn op"    onclick="append('+')">+</button>
      <button class="btn eq"    onclick="calculate()" style="grid-row:span 2;">=</button>
      <button class="btn num span2" onclick="append('0')">0</button>
      <button class="btn num"   onclick="append('.')">.</button>
      <button class="btn op"    onclick="negate()">±</button>
    </div>
  </div>
  <div id="panel-basic" style="display:none;">
    <div class="btns basic">
      <button class="btn clear" onclick="clearAll()">AC</button>
      <button class="btn op"    onclick="negate()">±</button>
      <button class="btn op"    onclick="append('%')">%</button>
      <button class="btn op"    onclick="append('/')">÷</button>
      <button class="btn num"   onclick="append('7')">7</button>
      <button class="btn num"   onclick="append('8')">8</button>
      <button class="btn num"   onclick="append('9')">9</button>
      <button class="btn op"    onclick="append('*')">×</button>
      <button class="btn num"   onclick="append('4')">4</button>
      <button class="btn num"   onclick="append('5')">5</button>
      <button class="btn num"   onclick="append('6')">6</button>
      <button class="btn op"    onclick="append('-')">−</button>
      <button class="btn num"   onclick="append('1')">1</button>
      <button class="btn num"   onclick="append('2')">2</button>
      <button class="btn num"   onclick="append('3')">3</button>
      <button class="btn op"    onclick="append('+')">+</button>
      <button class="btn num span2" onclick="append('0')">0</button>
      <button class="btn num"   onclick="append('.')">.</button>
      <button class="btn eq"    onclick="calculate()">=</button>
    </div>
  </div>
  <div class="hist-panel" id="histPanel">
    <div id="histItems"><div class="hist-empty">No history yet</div></div>
  </div>
  <div class="calc-footer">
    <span id="memLabel">M: —</span>
    <button class="hist-toggle" onclick="toggleHist()">history</button>
  </div>
</div>
<script>
let expr='',result='0',memory=null,isDeg=true,history=[],histOpen=false,currentPanel='sci';
const mainDisplay=document.getElementById('mainDisplay');
const exprLine=document.getElementById('exprLine');
const histLine=document.getElementById('histLine');
const modeLabel=document.getElementById('modeLabel');
const memLabel=document.getElementById('memLabel');

document.addEventListener('keydown',e=>{
  if(e.key>='0'&&e.key<='9')append(e.key);
  else if(e.key==='+')append('+');
  else if(e.key==='-')append('-');
  else if(e.key==='*')append('*');
  else if(e.key==='/'){e.preventDefault();append('/');}
  else if(e.key==='.')append('.');
  else if(e.key==='(')append('(');
  else if(e.key===')')append(')');
  else if(e.key==='%')append('%');
  else if(e.key==='^')append('**');
  else if(e.key==='Enter'||e.key==='=')calculate();
  else if(e.key==='Backspace')del();
  else if(e.key==='Escape')clearAll();
});

function append(val){expr+=val;updateDisplay();}
function fn(name){expr+=name;updateDisplay();}
function del(){expr=expr.slice(0,-1);updateDisplay();}
function clearAll(){expr='';result='0';histLine.textContent='';updateDisplay();}
function negate(){if(expr){expr=`-(${expr})`;}else if(result!=='0'){expr=`-(${result})`;}updateDisplay();}
function toggleDeg(){
  isDeg=!isDeg;
  modeLabel.textContent=isDeg?'DEG':'RAD';
  document.getElementById('degBtn').textContent=isDeg?'DEG':'RAD';
}

function calculate(){
  if(!expr)return;
  let evalExpr=expr;
  if(isDeg){
    evalExpr=evalExpr.replace(/\bsin\(/g,'sin(radians(');
    evalExpr=evalExpr.replace(/\bcos\(/g,'cos(radians(');
    evalExpr=evalExpr.replace(/\btan\(/g,'tan(radians(');
    const open=(evalExpr.match(/\(/g)||[]).length;
    const close=(evalExpr.match(/\)/g)||[]).length;
    if(open>close)evalExpr+=')'.repeat(open-close);
  }
  fetch('/calc_eval',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expr:evalExpr})})
  .then(r=>r.json())
  .then(data=>{
    if(data.error){
      mainDisplay.textContent='Error';mainDisplay.className='display-main error';result='Error';
    }else{
      const res=data.result;
      histLine.textContent=expr+' =';
      history.unshift({expr:expr,result:res});
      updateHistPanel();result=res;expr='';
      mainDisplay.textContent=res;
      mainDisplay.className=res.length>14?'display-main small':'display-main';
    }
  })
  .catch(()=>{mainDisplay.textContent='Error';mainDisplay.className='display-main error';});
}

function updateDisplay(){
  exprLine.textContent=expr||'';
  const disp=expr||result;
  mainDisplay.textContent=disp;
  mainDisplay.className=disp.length>14?'display-main small':'display-main';
}

function mem(op){
  if(op==='MC'){memory=null;memLabel.textContent='M: —';}
  if(op==='MR'&&memory!==null){expr+=memory;updateDisplay();}
  if(op==='MS'){memory=result!=='Error'?result:null;memLabel.textContent=memory?`M: ${memory}`:'M: —';}
  if(op==='M+'&&memory!==null){const v=parseFloat(memory)+parseFloat(result);memory=String(v);memLabel.textContent=`M: ${memory}`;}
}

function switchPanel(name){
  document.getElementById('panel-sci').style.display=name==='sci'?'':'none';
  document.getElementById('panel-basic').style.display=name==='basic'?'':'none';
  document.querySelectorAll('.mode-tab').forEach((t,i)=>{t.classList.toggle('active',(i===0&&name==='sci')||(i===1&&name==='basic'));});
  currentPanel=name;
  if(histOpen){histOpen=false;document.getElementById('histPanel').classList.remove('open');}
}

function toggleHist(){
  histOpen=!histOpen;
  document.getElementById('histPanel').classList.toggle('open',histOpen);
  if(histOpen){
    document.getElementById('panel-sci').style.display='none';
    document.getElementById('panel-basic').style.display='none';
    document.querySelectorAll('.mode-tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.mode-tab')[2].classList.add('active');
  }else{switchPanel(currentPanel);}
}

function updateHistPanel(){
  const c=document.getElementById('histItems');
  if(history.length===0){c.innerHTML='<div class="hist-empty">No history yet</div>';return;}
  c.innerHTML=history.slice(0,20).map(h=>`<div class="hist-item" onclick="useHist('${h.result}')"><div class="hi-expr">${h.expr}</div><div class="hi-res">${h.result}</div></div>`).join('');
}

function useHist(val){expr=val;updateDisplay();toggleHist();}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(CHAT_HTML)

@app.route("/calc")
def calc():
    return render_template_string(CALC_HTML)

@app.route("/calc_eval", methods=["POST"])
def calc_eval():
    data = request.json
    expr = data.get("expr", "").strip()
    result = evaluate_math(expr)
    if result is None:
        try:
            res = eval(expr, {"__builtins__": {}}, SAFE_MATH)
            if isinstance(res, float):
                if res == int(res) and abs(res) < 1e15:
                    res = int(res)
                else:
                    res = round(res, 10)
            result = str(res)
        except Exception as ex:
            return jsonify({"error": str(ex)})
    return jsonify({"result": result})

@app.route("/ask", methods=["POST"])
def ask():
    data     = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"route": "Error", "answer": "Empty question."})
    route, answer = router(question)
    confidence = None
    if route == "SLM":
        _, confidence = slm_generate_with_confidence(question)
    return jsonify({
        "route":      route,
        "answer":     answer,
        "confidence": round(confidence, 3) if confidence else None
    })

if __name__ == "__main__":
    import socket
    def find_free_port(start=5000):
        port = start
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
            port += 1
    port = find_free_port()
    print(f"\n{'='*50}")
    print(f"  Neural Router v7.2")
    print(f"  Chat:       http://localhost:{port}")
    print(f"  Calculator: http://localhost:{port}/calc")
    print(f"{'='*50}\n")
    app.run(debug=False, port=port)