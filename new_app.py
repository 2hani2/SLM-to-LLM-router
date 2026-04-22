"""
Neural Router v7.0 — NanoQA + Scientific Calculator
Run:
    Terminal 1: ollama serve
    Terminal 2: cd /Users/venisa/slm_project && source venv/bin/activate && python3 app.py

Type /calc in the chat to open the scientific calculator
"""

import os, sys, re, math, torch, torch.nn.functional as F, ollama
from flask import Flask, request, jsonify, render_template_string
from transformers import GPT2Tokenizer

# ── Load NanoQA ───────────────────────────────────────────────
MODEL_PATH = "./models/nanoqa"
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
print(f"NanoQA loaded on {device}")

# ── Math engine ───────────────────────────────────────────────
SAFE_MATH = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "abs": abs,
    "pow": pow, "round": round, "factorial": math.factorial,
    "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
    "degrees": math.degrees, "radians": math.radians,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
}

def evaluate_math(expression):
    """Evaluate a math expression safely."""
    expr = expression.strip()
    expr = expr.replace("^", "**").replace("x", "*").replace("×", "*").replace("÷", "/")
    expr = expr.replace("π", str(math.pi))
    if not expr:
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, SAFE_MATH)
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                result = int(result)
            else:
                result = f"{result:.10g}"
        return str(result)
    except:
        return None

def evaluate_math_from_question(question):
    """Try to extract and evaluate math from a natural language question."""
    expr = question.lower()
    expr = re.sub(r'(what is|calculate|compute|evaluate|solve|find|what\'s)', '', expr).strip()
    expr = expr.replace("squared", "**2").replace("cubed", "**3")
    expr = expr.strip("? \n")
    if not re.search(r'[\d\+\-\*\/\(\)\^]', expr):
        return None
    if re.search(r'[a-z]{5,}', re.sub(
        r'sqrt|sin|cos|tan|log|exp|abs|pow|factorial|ceil|floor|asin|acos|atan|sinh|cosh|tanh|degrees|radians',
        '', expr)):
        return None
    return evaluate_math(expr)

# ── SLM generation with confidence ───────────────────────────
def slm_generate_with_confidence(question):
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
    ans_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if "." in ans_text:
        ans_text = ans_text.split(".")[0].strip() + "."
    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
    return ans_text, confidence

# ── Garbage detection ─────────────────────────────────────────
def is_garbage(text):
    words  = text.split()
    unique = set(w.lower() for w in words)
    if len(words) < 2:                      return True
    if len(words) > 20:                     return True
    if len(unique) == 1:                    return True
    if len(words) > 4 and len(unique) < 3: return True
    if re.match(r'^[\d\s\.\,]+$', text):   return True
    return False

# ── Router ────────────────────────────────────────────────────
def router(question):
    q        = question.strip()
    math_ans = evaluate_math_from_question(q)
    if math_ans:
        return "Math", math_ans
    slm_ans, confidence = slm_generate_with_confidence(q)
    if not is_garbage(slm_ans) and confidence >= 0.35:
        return "SLM", slm_ans
    if not is_garbage(slm_ans) and confidence >= 0.15:
        slm_ans2, confidence2 = slm_generate_with_confidence(q)
        if not is_garbage(slm_ans2) and confidence2 >= 0.35:
            return "SLM", slm_ans2
    return "LLM", llm_generate(q)

# ── LLM ───────────────────────────────────────────────────────
def llm_generate(question):
    try:
        response = ollama.chat(
            model    = "mistral",
            messages = [{"role": "user", "content": question}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Ollama error: {str(e)}. Is 'ollama serve' running?"

# ── Flask ─────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Router</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0f0f10; color: #e4e4e7; min-height: 100vh;
  display: flex; flex-direction: column; align-items: center; padding: 24px;
}
.header { text-align: center; margin-bottom: 32px; }
.header h1 { font-size: 28px; font-weight: 600; letter-spacing: -0.5px; }
.header p  { color: #71717a; font-size: 14px; margin-top: 6px; }
.badges { display: flex; gap: 8px; justify-content: center; margin-top: 12px; flex-wrap: wrap; }
.badge { padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
.badge.slm  { background: #1d4ed820; color: #60a5fa; border: 1px solid #1d4ed840; }
.badge.llm  { background: #7c3aed20; color: #a78bfa; border: 1px solid #7c3aed40; }
.badge.math { background: #d9770620; color: #fb923c; border: 1px solid #d9770640; }
.badge.calc { background: #05966920; color: #34d399; border: 1px solid #05966940;
              cursor: pointer; transition: background 0.15s; }
.badge.calc:hover { background: #059669 40; }
.chat { width: 100%; max-width: 720px; display: flex; flex-direction: column; gap: 16px; margin-bottom: 130px; }
.bubble { max-width: 82%; padding: 12px 16px; border-radius: 16px; font-size: 15px; line-height: 1.6; }
.bubble.user { background: #27272a; align-self: flex-end; border-bottom-right-radius: 4px; }
.bubble.bot  { background: #18181b; border: 1px solid #27272a; align-self: flex-start; border-bottom-left-radius: 4px; }
.route-tag   { font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px; opacity: 0.7; }
.route-tag.SLM  { color: #60a5fa; }
.route-tag.LLM  { color: #a78bfa; }
.route-tag.Math { color: #fb923c; }
.route-tag.Calc { color: #34d399; }
.confidence { font-size: 10px; color: #52525b; margin-top: 4px; }
.input-area {
  position: fixed; bottom: 0; width: 100%;
  background: #0f0f10; padding: 16px 24px; border-top: 1px solid #27272a;
}
.input-wrap { max-width: 720px; margin: 0 auto; display: flex; gap: 10px; }
textarea {
  flex: 1; background: #18181b; border: 1px solid #3f3f46; border-radius: 12px;
  color: #e4e4e7; padding: 12px 16px; font-size: 14px; resize: none;
  font-family: inherit; outline: none; max-height: 120px;
}
textarea:focus { border-color: #60a5fa; }
.send-btn {
  background: #2563eb; color: #fff; border: none; border-radius: 12px;
  padding: 12px 20px; cursor: pointer; font-size: 14px; font-weight: 500;
  transition: background 0.15s; white-space: nowrap;
}
.send-btn:hover    { background: #1d4ed8; }
.send-btn:disabled { background: #27272a; color: #52525b; cursor: not-allowed; }
.hint { font-size: 11px; color: #52525b; text-align: center; margin-top: 8px; }
.hint kbd {
  background: #27272a; border: 1px solid #3f3f46; border-radius: 4px;
  padding: 1px 5px; font-size: 10px; color: #a1a1aa; font-family: inherit;
}
.typing { opacity: 0.5; font-style: italic; }

/* ── Calculator overlay ─────────────────────────────────────── */
.calc-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.75); z-index: 100;
  align-items: center; justify-content: center;
}
.calc-overlay.open { display: flex; }
.calc-box {
  background: #18181b; border: 1px solid #3f3f46; border-radius: 24px;
  padding: 24px; width: 360px;
  box-shadow: 0 32px 80px rgba(0,0,0,0.7);
  animation: popIn 0.18s ease;
}
@keyframes popIn {
  from { transform: scale(0.9); opacity: 0; }
  to   { transform: scale(1);   opacity: 1; }
}
.calc-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 16px;
}
.calc-title { font-size: 12px; font-weight: 600; color: #52525b; letter-spacing: 1px; }
.calc-close {
  background: #27272a; border: none; color: #a1a1aa;
  width: 28px; height: 28px; border-radius: 50%;
  cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center;
  transition: background 0.15s;
}
.calc-close:hover { background: #3f3f46; color: #e4e4e7; }
.calc-display {
  background: #0f0f10; border: 1px solid #27272a; border-radius: 14px;
  padding: 16px 20px; margin-bottom: 16px; min-height: 88px;
  display: flex; flex-direction: column; justify-content: flex-end; align-items: flex-end;
  gap: 4px;
}
.calc-expr   { font-size: 13px; color: #52525b; word-break: break-all; min-height: 18px; }
.calc-result { font-size: 32px; font-weight: 400; color: #e4e4e7; word-break: break-all;
               transition: color 0.2s; }
.calc-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
}
.cb {
  border: none; border-radius: 12px; padding: 16px 8px;
  font-size: 15px; font-weight: 500; cursor: pointer;
  transition: opacity 0.1s, transform 0.07s;
  font-family: inherit;
}
.cb:active { transform: scale(0.91); opacity: 0.8; }
.cb.num { background: #27272a; color: #e4e4e7; }
.cb.num:hover { background: #3f3f46; }
.cb.op  { background: #1e3a5f; color: #93c5fd; }
.cb.op:hover  { background: #1d4ed840; }
.cb.fn  { background: #1c1c30; color: #c4b5fd; font-size: 13px; }
.cb.fn:hover  { background: #2d2b55; }
.cb.eq  { background: #2563eb; color: #fff; font-size: 18px; }
.cb.eq:hover  { background: #1d4ed8; }
.cb.ac  { background: #3f1515; color: #fca5a5; }
.cb.ac:hover  { background: #5c1f1f; }
.cb.del { background: #2a1f10; color: #fdba74; }
.cb.del:hover { background: #3d2a10; }
.cb.wide { grid-column: span 2; }
.cb.send {
  grid-column: span 4;
  background: #064e3b; color: #34d399;
  border: 1px solid #065f4630;
  font-size: 13px; margin-top: 2px;
  display: flex; align-items: center; justify-content: center; gap: 6px;
}
.cb.send:hover { background: #065f46; }
</style>
</head>
<body>

<div class="header">
  <h1>Neural Router</h1>
  <p>NanoQA · No lookup table · Mistral fallback</p>
  <div class="badges">
    <span class="badge slm">SLM</span>
    <span class="badge llm">LLM</span>
    <span class="badge math">MATH</span>
    <span class="badge calc" onclick="openCalc()" title="Open scientific calculator">/calc</span>
  </div>
</div>

<div class="chat" id="chat"></div>

<!-- ── Scientific Calculator ── -->
<div class="calc-overlay" id="overlay" onclick="bgClick(event)">
  <div class="calc-box" id="calcBox">
    <div class="calc-header">
      <span class="calc-title">SCIENTIFIC CALCULATOR</span>
      <button class="calc-close" onclick="closeCalc()">✕</button>
    </div>

    <div class="calc-display">
      <div class="calc-expr"   id="cExpr"></div>
      <div class="calc-result" id="cResult">0</div>
    </div>

    <div class="calc-grid">
      <!-- scientific functions row 1 -->
      <button class="cb fn" onclick="ap('sin(')">sin</button>
      <button class="cb fn" onclick="ap('cos(')">cos</button>
      <button class="cb fn" onclick="ap('tan(')">tan</button>
      <button class="cb fn" onclick="ap('log(')">log</button>
      <!-- scientific functions row 2 -->
      <button class="cb fn" onclick="ap('sqrt(')">√x</button>
      <button class="cb fn" onclick="ap('factorial(')">x!</button>
      <button class="cb fn" onclick="ap('**2')">x²</button>
      <button class="cb fn" onclick="ap('**')">xⁿ</button>
      <!-- constants + brackets -->
      <button class="cb fn" onclick="ap('log10(')">log₁₀</button>
      <button class="cb fn" onclick="ap('log2(')">log₂</button>
      <button class="cb fn" onclick="ap('pi')">π</button>
      <button class="cb fn" onclick="ap('e')">e</button>
      <!-- clear row -->
      <button class="cb ac"  onclick="clearAll()">AC</button>
      <button class="cb del" onclick="delChar()">⌫</button>
      <button class="cb op"  onclick="ap('(')"> ( </button>
      <button class="cb op"  onclick="ap(')')"> ) </button>
      <!-- operators -->
      <button class="cb op"  onclick="ap('%')">%</button>
      <button class="cb op"  onclick="ap('/')">÷</button>
      <button class="cb op"  onclick="ap('*')">×</button>
      <button class="cb op"  onclick="ap('-')">−</button>
      <!-- number pad -->
      <button class="cb num" onclick="ap('7')">7</button>
      <button class="cb num" onclick="ap('8')">8</button>
      <button class="cb num" onclick="ap('9')">9</button>
      <button class="cb op"  onclick="ap('+')">+</button>
      <button class="cb num" onclick="ap('4')">4</button>
      <button class="cb num" onclick="ap('5')">5</button>
      <button class="cb num" onclick="ap('6')">6</button>
      <button class="cb num wide" onclick="ap('0')">0</button>
      <button class="cb num" onclick="ap('1')">1</button>
      <button class="cb num" onclick="ap('2')">2</button>
      <button class="cb num" onclick="ap('3')">3</button>
      <button class="cb num" onclick="ap('.')">.</button>
      <!-- equals -->
      <button class="cb eq wide" onclick="calculate()">=</button>
      <!-- send to chat -->
      <button class="cb send" onclick="sendToChat()">↑ Send result to chat</button>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="q" rows="1"
      placeholder="Ask anything… or type /calc to open calculator"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();ask()}"
      oninput="onType(this)"></textarea>
    <button class="send-btn" id="btn" onclick="ask()">Send</button>
  </div>
  <p class="hint">
    <kbd>Enter</kbd> send &nbsp;·&nbsp;
    <kbd>Shift+Enter</kbd> newline &nbsp;·&nbsp;
    type <kbd>/calc</kbd> to open calculator
  </p>
</div>

<script>
const chat = document.getElementById('chat');
const qEl  = document.getElementById('q');
const btn  = document.getElementById('btn');

// ── helpers ──────────────────────────────────────────────────
function addBubble(html, cls) {
  const d = document.createElement('div');
  d.className = 'bubble ' + cls;
  d.innerHTML = html;
  chat.appendChild(d);
  d.scrollIntoView({behavior: 'smooth'});
  return d;
}

function onType(el) {
  if (el.value.trim() === '/calc') { el.value = ''; openCalc(); return; }
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ── chat ─────────────────────────────────────────────────────
async function ask() {
  const text = qEl.value.trim();
  if (!text) return;
  if (text === '/calc') { qEl.value = ''; openCalc(); return; }
  qEl.value = ''; btn.disabled = true; qEl.style.height = 'auto';
  addBubble(text, 'user');
  const typing = addBubble('<span class="typing">thinking...</span>', 'bot');
  try {
    const res  = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: text})
    });
    const d = await res.json();
    const conf = d.confidence
      ? `<div class="confidence">confidence: ${(d.confidence*100).toFixed(0)}%</div>` : '';
    typing.innerHTML = `<div class="route-tag ${d.route}">● ${d.route}</div>${d.answer}${conf}`;
  } catch(e) { typing.innerHTML = 'Error: ' + e.message; }
  btn.disabled = false; qEl.focus();
}

// ── calculator ───────────────────────────────────────────────
let expr = '';

function openCalc() {
  expr = '';
  document.getElementById('cExpr').textContent   = '';
  document.getElementById('cResult').textContent = '0';
  document.getElementById('cResult').style.color = '#e4e4e7';
  document.getElementById('overlay').classList.add('open');
}
function closeCalc() {
  document.getElementById('overlay').classList.remove('open');
  qEl.focus();
}
function bgClick(e) {
  if (e.target === document.getElementById('overlay')) closeCalc();
}
function ap(val) {
  expr += val;
  document.getElementById('cExpr').textContent = expr;
  liveCalc();
}
function clearAll() {
  expr = '';
  document.getElementById('cExpr').textContent   = '';
  document.getElementById('cResult').textContent = '0';
  document.getElementById('cResult').style.color = '#e4e4e7';
}
function delChar() {
  expr = expr.slice(0, -1);
  document.getElementById('cExpr').textContent = expr;
  liveCalc();
}
async function liveCalc() {
  if (!expr) return;
  try {
    const r = await fetch('/calc', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({expr})
    });
    const d = await r.json();
    if (d.result !== null) {
      document.getElementById('cResult').textContent = d.result;
      document.getElementById('cResult').style.color = '#a1a1aa';
    }
  } catch(e) {}
}
async function calculate() {
  if (!expr) return;
  try {
    const r = await fetch('/calc', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({expr})
    });
    const d = await r.json();
    const res = document.getElementById('cResult');
    if (d.result !== null) {
      res.textContent = d.result;
      res.style.color = '#34d399';
      document.getElementById('cExpr').textContent = expr + ' =';
    } else {
      res.textContent = 'Error';
      res.style.color = '#f87171';
    }
  } catch(e) {}
}
function sendToChat() {
  const result = document.getElementById('cResult').textContent;
  if (!expr || result === '0' || result === 'Error') return;
  const dispExpr = expr.replace(/\*\*/g,'ⁿ').replace(/\*/g,'×').replace(/\//g,'÷');
  closeCalc();
  addBubble(`/calc &nbsp;<code>${dispExpr}</code>`, 'user');
  addBubble(
    `<div class="route-tag Calc">● Calc</div><strong>${result}</strong>`,
    'bot'
  );
}

// keyboard support when calc is open
document.addEventListener('keydown', e => {
  const open = document.getElementById('overlay').classList.contains('open');
  if (!open) return;
  if (e.key === 'Escape')    { closeCalc(); return; }
  if (e.key === 'Enter')     { calculate(); return; }
  if (e.key === 'Backspace') { e.preventDefault(); delChar(); return; }
  if (e.key === 'Delete')    { clearAll(); return; }
  if (/^[\d\.\+\-\*\/\(\)\%]$/.test(e.key)) { ap(e.key); }
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

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

@app.route("/calc", methods=["POST"])
def calc():
    """Dedicated calculator endpoint — evaluates expressions directly, no AI involved."""
    data   = request.json
    expr   = data.get("expr", "").strip()
    result = evaluate_math(expr)
    return jsonify({"result": result})

if __name__ == "__main__":
    for port in [5000, 5001, 5002, 5003]:
        try:
            print(f"\nNeural Router v7.0 — Scientific Calculator Edition")
            print(f"Open: http://localhost:{port}")
            print(f"Tip: type /calc in chat to open the scientific calculator")
            app.run(debug=False, port=port)
            break
        except OSError:
            continue
