import socket, math, re, sys, os
from flask import Flask, request, jsonify, render_template_string
import torch, ollama
from transformers import GPT2Tokenizer

# ── LOAD NANOQA ───────────────────────────────────────────────
MODEL_PATH = "./models/nanoqa"
sys.path.insert(0, os.path.abspath(MODEL_PATH))
from nanoqa_arch import NanoQAConfig, NanoQAModel

app = Flask(__name__)

print("Loading NanoQA SLM...")

# Try loading tokenizer from root of nanoqa folder first, then tokenizer subfolder
try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
except Exception:
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"))

tokenizer.pad_token = tokenizer.eos_token

config = NanoQAConfig.from_pretrained(MODEL_PATH)
model  = NanoQAModel(config)
# Load weights manually to avoid transformers 5.x compatibility issues
import safetensors.torch
weights_path = os.path.join(MODEL_PATH, "model.safetensors")
state_dict = safetensors.torch.load_file(weights_path)
model.load_state_dict(state_dict, strict=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model  = model.to(device)
model.eval()
print(f"NanoQA SLM loaded on {device}!")

# ── KNOWLEDGE BASE ────────────────────────────────────────────
CAPITALS = {
    "france":"Paris","germany":"Berlin","italy":"Rome","spain":"Madrid",
    "portugal":"Lisbon","netherlands":"Amsterdam","belgium":"Brussels",
    "switzerland":"Bern","austria":"Vienna","sweden":"Stockholm",
    "norway":"Oslo","denmark":"Copenhagen","finland":"Helsinki",
    "poland":"Warsaw","czech republic":"Prague","czechia":"Prague",
    "hungary":"Budapest","romania":"Bucharest","bulgaria":"Sofia",
    "greece":"Athens","ukraine":"Kyiv","russia":"Moscow",
    "uk":"London","united kingdom":"London","england":"London",
    "ireland":"Dublin","croatia":"Zagreb","serbia":"Belgrade",
    "slovakia":"Bratislava","slovenia":"Ljubljana","albania":"Tirana",
    "moldova":"Chisinau","belarus":"Minsk","latvia":"Riga",
    "lithuania":"Vilnius","estonia":"Tallinn","luxembourg":"Luxembourg City",
    "malta":"Valletta","iceland":"Reykjavik","monaco":"Monaco",
    "andorra":"Andorra la Vella","bosnia":"Sarajevo",
    "north macedonia":"Skopje","montenegro":"Podgorica",
    "kosovo":"Pristina","cyprus":"Nicosia","liechtenstein":"Vaduz",
    "china":"Beijing","japan":"Tokyo","india":"New Delhi",
    "south korea":"Seoul","north korea":"Pyongyang",
    "indonesia":"Jakarta","pakistan":"Islamabad","bangladesh":"Dhaka",
    "vietnam":"Hanoi","thailand":"Bangkok","malaysia":"Kuala Lumpur",
    "philippines":"Manila","myanmar":"Naypyidaw","cambodia":"Phnom Penh",
    "laos":"Vientiane","singapore":"Singapore",
    "nepal":"Kathmandu","sri lanka":"Sri Jayawardenepura Kotte",
    "bhutan":"Thimphu","maldives":"Male","mongolia":"Ulaanbaatar",
    "taiwan":"Taipei","afghanistan":"Kabul","iran":"Tehran",
    "iraq":"Baghdad","turkey":"Ankara","syria":"Damascus",
    "lebanon":"Beirut","jordan":"Amman","israel":"Jerusalem",
    "saudi arabia":"Riyadh","uae":"Abu Dhabi",
    "united arab emirates":"Abu Dhabi","qatar":"Doha",
    "kuwait":"Kuwait City","bahrain":"Manama","oman":"Muscat",
    "yemen":"Sanaa","uzbekistan":"Tashkent","kazakhstan":"Astana",
    "kyrgyzstan":"Bishkek","tajikistan":"Dushanbe",
    "turkmenistan":"Ashgabat","azerbaijan":"Baku",
    "armenia":"Yerevan","georgia":"Tbilisi",
    "usa":"Washington, D.C.","united states":"Washington, D.C.",
    "america":"Washington, D.C.","us":"Washington, D.C.",
    "canada":"Ottawa","mexico":"Mexico City","brazil":"Brasilia",
    "argentina":"Buenos Aires","colombia":"Bogota","chile":"Santiago",
    "peru":"Lima","venezuela":"Caracas","ecuador":"Quito",
    "bolivia":"Sucre","paraguay":"Asuncion","uruguay":"Montevideo",
    "cuba":"Havana","haiti":"Port-au-Prince",
    "dominican republic":"Santo Domingo","jamaica":"Kingston",
    "panama":"Panama City","costa rica":"San Jose",
    "guatemala":"Guatemala City","honduras":"Tegucigalpa",
    "el salvador":"San Salvador","nicaragua":"Managua",
    "nigeria":"Abuja","south africa":"Pretoria","egypt":"Cairo",
    "kenya":"Nairobi","ethiopia":"Addis Ababa","ghana":"Accra",
    "tanzania":"Dodoma","uganda":"Kampala","mozambique":"Maputo",
    "niger":"Niamey","mali":"Bamako","senegal":"Dakar",
    "zambia":"Lusaka","zimbabwe":"Harare","rwanda":"Kigali",
    "somalia":"Mogadishu","sudan":"Khartoum","south sudan":"Juba",
    "libya":"Tripoli","tunisia":"Tunis","algeria":"Algiers",
    "morocco":"Rabat","angola":"Luanda","namibia":"Windhoek",
    "botswana":"Gaborone","malawi":"Lilongwe","congo":"Kinshasa",
    "gabon":"Libreville","australia":"Canberra",
    "new zealand":"Wellington","fiji":"Suva",
    "papua new guinea":"Port Moresby",
}

FACTS = {
    "speed of light":"The speed of light in a vacuum is approximately 299,792,458 m/s (3x10^8 m/s).",
    "boiling point of water":"Water boils at 100°C (212°F) at standard atmospheric pressure.",
    "freezing point of water":"Water freezes at 0°C (32°F) at standard atmospheric pressure.",
    "distance from earth to sun":"The average distance from Earth to the Sun is about 149.6 million km (1 AU).",
    "distance from earth to moon":"The average distance from Earth to the Moon is about 384,400 km.",
    "gravity on earth":"The acceleration due to gravity on Earth is approximately 9.8 m/s².",
    "number of planets":"There are 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
    "largest planet":"Jupiter is the largest planet in our solar system.",
    "smallest planet":"Mercury is the smallest planet in our solar system.",
    "hottest planet":"Venus is the hottest planet, with surface temperatures around 465°C.",
    "chemical formula of water":"The chemical formula of water is H₂O.",
    "chemical formula of salt":"The chemical formula of table salt is NaCl.",
    "chemical formula of co2":"The chemical formula of carbon dioxide is CO₂.",
    "atomic number of hydrogen":"Hydrogen has an atomic number of 1.",
    "atomic number of carbon":"Carbon has an atomic number of 6.",
    "atomic number of oxygen":"Oxygen has an atomic number of 8.",
    "atomic number of gold":"Gold has an atomic number of 79.",
    "atomic number of iron":"Iron has an atomic number of 26.",
    "human body temperature":"Normal human body temperature is approximately 37°C (98.6°F).",
    "number of bones in human body":"The adult human body has 206 bones.",
    "number of chromosomes in humans":"Humans have 46 chromosomes (23 pairs).",
    "largest ocean":"The Pacific Ocean is the largest ocean, covering about 165 million km².",
    "largest continent":"Asia is the largest continent by both area and population.",
    "longest river":"The Nile is often considered the longest river, at about 6,650 km.",
    "highest mountain":"Mount Everest is the highest mountain on Earth at 8,849 metres above sea level.",
    "deepest ocean point":"The Mariana Trench is the deepest known point at about 11,034 metres.",
    "full form of cpu":"CPU stands for Central Processing Unit.",
    "full form of gpu":"GPU stands for Graphics Processing Unit.",
    "full form of ram":"RAM stands for Random Access Memory.",
    "full form of rom":"ROM stands for Read-Only Memory.",
    "full form of http":"HTTP stands for HyperText Transfer Protocol.",
    "full form of https":"HTTPS stands for HyperText Transfer Protocol Secure.",
    "full form of html":"HTML stands for HyperText Markup Language.",
    "full form of css":"CSS stands for Cascading Style Sheets.",
    "full form of sql":"SQL stands for Structured Query Language.",
    "full form of api":"API stands for Application Programming Interface.",
    "full form of ai":"AI stands for Artificial Intelligence.",
    "full form of ml":"ML stands for Machine Learning.",
    "full form of os":"OS stands for Operating System.",
    "full form of url":"URL stands for Uniform Resource Locator.",
    "full form of dns":"DNS stands for Domain Name System.",
    "who invented python":"Python was created by Guido van Rossum, first released in 1991.",
    "who invented the internet":"The internet evolved from ARPANET (1960s). Tim Berners-Lee invented the Web in 1989.",
    "who invented the telephone":"Alexander Graham Bell is credited with inventing the telephone in 1876.",
    "who invented the airplane":"The Wright Brothers made the first powered airplane flight in 1903.",
    "who invented the light bulb":"Thomas Edison developed the first practical incandescent light bulb in 1879.",
    "when did world war 2 start":"World War II started on September 1, 1939, when Germany invaded Poland.",
    "when did world war 2 end":"World War II ended on September 2, 1945, with Japan's formal surrender.",
    "when did world war 1 start":"World War I started on July 28, 1914.",
    "when did world war 1 end":"World War I ended on November 11, 1918.",
    "when did india get independence":"India gained independence on August 15, 1947.",
    "when did usa get independence":"The United States declared independence on July 4, 1776.",
    "value of pi":"Pi (π) ≈ 3.14159265358979...",
    "value of e":"Euler's number e ≈ 2.71828182845904...",
    "pythagorean theorem":"Pythagorean theorem: a² + b² = c², where c is the hypotenuse of a right triangle.",
    "largest country":"Russia is the largest country by area, covering about 17.1 million km².",
    "smallest country":"Vatican City is the smallest country, covering just 0.44 km².",
    "most spoken language":"Mandarin Chinese has the most native speakers. English is most widely spoken overall.",
    "most populous country":"India and China are the most populous, each with over 1.4 billion people.",
    "currency of usa":"The currency of the USA is the US Dollar (USD).",
    "currency of uk":"The currency of the UK is the Pound Sterling (GBP).",
    "currency of india":"The currency of India is the Indian Rupee (INR).",
    "currency of japan":"The currency of Japan is the Japanese Yen (JPY).",
    "currency of china":"The currency of China is the Chinese Yuan (CNY).",
    "currency of europe":"Most European Union countries use the Euro (EUR).",
}

DEFINITIONS = {
    "ai":"AI (Artificial Intelligence) is the simulation of human intelligence by machines.",
    "machine learning":"Machine Learning is a subset of AI where systems learn from data to improve automatically.",
    "deep learning":"Deep Learning uses multi-layered neural networks to learn complex patterns from large datasets.",
    "neural network":"A Neural Network is a computing system inspired by biological brains, made of connected layers of nodes.",
    "python":"Python is a high-level, general-purpose programming language known for its simplicity.",
    "algorithm":"An Algorithm is a step-by-step set of instructions to solve a problem.",
    "database":"A Database is an organized collection of structured data stored electronically.",
    "api":"An API (Application Programming Interface) defines how software components communicate.",
    "cpu":"A CPU (Central Processing Unit) is the primary processor that executes instructions.",
    "gpu":"A GPU (Graphics Processing Unit) is a processor designed for parallel computations.",
    "ram":"RAM (Random Access Memory) is fast temporary memory for active tasks.",
    "html":"HTML (HyperText Markup Language) is the standard language for creating web pages.",
    "css":"CSS (Cascading Style Sheets) controls the visual presentation of web pages.",
    "javascript":"JavaScript is a scripting language that adds interactivity to web pages.",
    "blockchain":"Blockchain is a distributed ledger where data is stored in linked tamper-resistant blocks.",
    "cloud computing":"Cloud Computing delivers computing services over the internet on demand.",
    "encryption":"Encryption converts data into a coded format so only authorized parties can read it.",
    "slm":"An SLM (Small Language Model) is a compact AI model optimized for speed and efficiency.",
    "llm":"An LLM (Large Language Model) is a large AI trained on massive text data to generate text.",
    "transformer":"A Transformer is a neural network architecture using attention mechanisms, foundational to LLMs.",
    "gpt":"GPT (Generative Pre-trained Transformer) is a language model architecture by OpenAI.",
    "photosynthesis":"Photosynthesis is the process by which plants convert sunlight, CO₂ and water into glucose.",
    "gravity":"Gravity is the fundamental force of attraction between objects with mass.",
    "dna":"DNA (Deoxyribonucleic Acid) encodes the genetic instructions for living organisms.",
    "atom":"An Atom is the smallest unit of a chemical element, made of protons, neutrons, and electrons.",
    "inflation":"Inflation is the rate at which the general price level of goods and services rises.",
    "democracy":"Democracy is a system of government where citizens exercise power through elected representatives.",
    "internet":"The Internet is a global system of interconnected computer networks.",
    "software":"Software is a set of instructions that tell a computer how to perform tasks.",
    "hardware":"Hardware refers to the physical components of a computer system.",
    "operating system":"An Operating System manages computer hardware and software resources.",
    "compiler":"A Compiler translates high-level source code into machine code.",
    "recursion":"Recursion is a technique where a function calls itself to solve smaller instances of a problem.",
    "overfitting":"Overfitting occurs when a model memorizes training data and performs poorly on new data.",
    "underfitting":"Underfitting occurs when a model is too simple to capture patterns in the data.",
    "gradient descent":"Gradient Descent iteratively adjusts model parameters to minimize a loss function.",
    "backpropagation":"Backpropagation computes gradients of the loss to train neural networks.",
    "epoch":"An Epoch is one complete pass through the entire training dataset.",
    "learning rate":"Learning Rate controls how much model weights are adjusted per training step.",
    "natural language processing":"NLP enables computers to understand, interpret, and generate human language.",
    "computer vision":"Computer Vision enables machines to interpret and understand visual information.",
    "reinforcement learning":"Reinforcement Learning trains agents to make decisions by rewarding desired behaviors.",
    "random forest":"A Random Forest is an ensemble of decision trees that reduces overfitting.",
    "decision tree":"A Decision Tree makes decisions by splitting data based on feature values.",
    "sql":"SQL (Structured Query Language) is used to manage and query relational databases.",
    "git":"Git is a distributed version control system for tracking code changes.",
    "docker":"Docker packages applications into portable containers.",
    "rest":"REST (Representational State Transfer) is an architectural style for web APIs.",
    "json":"JSON (JavaScript Object Notation) is a lightweight data interchange format.",
    "tcp":"TCP (Transmission Control Protocol) ensures reliable, ordered data delivery over networks.",
    "http":"HTTP (HyperText Transfer Protocol) is the foundation of data communication on the Web.",
    "https":"HTTPS is the secure version of HTTP, encrypting data with SSL/TLS.",
    "ip address":"An IP Address is a unique numerical label assigned to each device on a network.",
    "dns":"DNS (Domain Name System) translates domain names into IP addresses.",
    "firewall":"A Firewall monitors and controls network traffic based on security rules.",
    "vpn":"A VPN (Virtual Private Network) creates an encrypted tunnel for secure internet communication.",
    "bandwidth":"Bandwidth is the maximum data transfer rate across a network.",
    "latency":"Latency is the delay between a request being sent and a response being received.",
    "cache":"A Cache stores frequently used data for faster retrieval.",
    "open source":"Open Source software has source code freely available for anyone to use and modify.",
}

# ── SCIENTIFIC MATH ENGINE ────────────────────────────────────
MATH_SAFE = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sqrt": math.sqrt, "cbrt": lambda x: x**(1/3),
    "ceil": math.ceil, "floor": math.floor, "abs": abs, "round": round,
    "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
    "comb": math.comb, "perm": math.perm,
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "hypot": math.hypot, "degrees": math.degrees, "radians": math.radians,
    "pow": pow,
}

def evaluate_math(question):
    q = question.lower().strip()
    q = re.sub(r'^(what is|whats|calculate|compute|solve|evaluate|find|simplify)\s*', '', q)
    q = q.rstrip("?").strip()
    replacements = [
        (r'\bplus\b', '+'), (r'\bminus\b', '-'), (r'\btimes\b', '*'),
        (r'\bmultiplied by\b', '*'), (r'\bdivided by\b', '/'),
        (r'\bover\b', '/'), (r'\bto the power of\b', '**'),
        (r'\bto the power\b', '**'), (r'\braise to\b', '**'),
        (r'\bsquared\b', '**2'), (r'\bcubed\b', '**3'),
        (r'\bsquare root of\b', 'sqrt('), (r'\bcube root of\b', 'cbrt('),
        (r'\bmod\b', '%'), (r'\bmodulo\b', '%'),
        (r'\bln\b', 'log'), (r'\blog base 2\b', 'log2'),
        (r'\blog base 10\b', 'log10'), (r'\bsine\b', 'sin'),
        (r'\bcosine\b', 'cos'), (r'\btangent\b', 'tan'),
        (r'(\d)\s*x\s*(\d)', r'\1*\2'),
        (r'\^', '**'),
    ]
    for pattern, replacement in replacements:
        q = re.sub(pattern, replacement, q)
    open_p  = q.count('(')
    close_p = q.count(')')
    if open_p > close_p:
        q += ')' * (open_p - close_p)
    if not re.search(r'[\d\+\-\*\/\(\)\.\%]', q):
        return None
    if re.search(r'[a-z_][a-z_]*\s*=', q):
        return None
    try:
        result = eval(q, {"__builtins__": {}}, MATH_SAFE)
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return str(round(result, 8))
        return str(result)
    except:
        return None

def check_capital(question):
    q = question.lower().strip().rstrip("?").strip()
    m = re.search(r'capital\s+of\s+([a-z\s]+)', q)
    if m:
        country = m.group(1).strip()
        if country in CAPITALS:
            return f"The capital of {country.title()} is {CAPITALS[country]}."
    return None

def check_fact(question):
    q = question.lower().strip().rstrip("?").strip()
    q = re.sub(r'^(what is the|what is|what are the|what are|who is|who invented|when did|how many|whats the|whats)', '', q).strip()
    for key, val in FACTS.items():
        if key in q or q in key:
            return val
    return None

def check_definition(question):
    q = question.lower().strip().rstrip("?").strip()
    q = re.sub(r'^(what is (?:a |an |the )?|what are (?:a |an |the )?|define (?:a |an |the )?|meaning of |tell me about (?:a |an |the )?)', '', q).strip()
    if q in DEFINITIONS:
        return DEFINITIONS[q]
    for key in DEFINITIONS:
        if key in q or q in key:
            return DEFINITIONS[key]
    return None

def check_summarization(question):
    q = question.lower().strip()
    if any(t in q for t in ["summarize","summarise","tldr","sum up","brief summary","in short"]):
        return "short" if len(question.split()) <= 80 else "long"
    return None

def complexity_score(question):
    score = 0.0
    q     = question.lower()
    words = len(q.split())
    if words > 20:   score += 0.3
    elif words > 10: score += 0.1
    if any(w in q for w in ["explain","why does","how does","analyze","compare",
                              "difference between","pros and cons","elaborate","discuss",
                              "implications","argue","critically","justify"]):
        score += 0.5
    if any(w in q for w in ["what is","who is","when did","where is","capital",
                              "define","full form","how many","which","who invented"]):
        score -= 0.4
    return max(0.0, min(1.0, score))

# ── SLM INFERENCE (NanoQA) ────────────────────────────────────
def get_confidence(question):
    prompt    = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids[:, -model.config.max_seq_len:]
    with torch.no_grad():
        out   = model(input_ids)
        probs = torch.softmax(out.logits, dim=-1)
        top_p = probs[0, -10:, :].max(dim=-1).values
    return round(top_p.mean().item(), 4)

def slm_generate(question):
    prompt    = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids[:, -model.config.max_seq_len:]
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=20,          # ← reduced from 60 to 20
            temperature=0.3,            # ← lower temperature = more focused
            top_k=10,                   # ← tighter top_k
            top_p=0.85,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    ans  = full.split("Answer:")[-1].strip() if "Answer:" in full else full.strip()
    # Stop at the first sentence — everything after the first period is hallucination
    if "." in ans:
        ans = ans.split(".")[0].strip() + "."
    return ans
# ── LLM (Ollama) ──────────────────────────────────────────────
def llm_generate(question):
    return ollama.chat(
        model="mistral",       # ← run `ollama list` in terminal and put your exact model name here
        messages=[{"role": "user", "content": question}]
    )["message"]["content"]

def is_garbage(text):
    words  = text.split()
    unique = set(words)
    if len(words) < 2:                       return True  # too short
    if len(words) > 20:                      return True  # too long = hallucinating
    if len(unique) == 1:                     return True
    if len(words) > 4 and len(unique) < 3:   return True
    return False
# ── ROUTER ────────────────────────────────────────────────────
def router(question):
    if cap      := check_capital(question):    return "Lookup", cap
    if fact     := check_fact(question):       return "Lookup", fact
    if defn     := check_definition(question): return "Lookup", defn
    if math_ans := evaluate_math(question):    return "Math",   math_ans

    stype = check_summarization(question)
    if stype == "short":
        text = re.sub(r'(summarize|summarise|tldr|sum up|brief summary|in short)[:\s]*','',question,flags=re.IGNORECASE).strip()
        return ("Summary", f"Summary: {text.capitalize()}") if len(text.split())<=15 else ("LLM", llm_generate(f"Summarize in one sentence: {text}"))
    elif stype == "long":
        text = re.sub(r'(summarize|summarise|tldr|sum up|brief summary|in short)[:\s]*','',question,flags=re.IGNORECASE).strip()
        return "LLM", llm_generate(f"Summarize concisely: {text}")

    slm_ans = slm_generate(question)
    if not is_garbage(slm_ans):
        return "SLM", slm_ans

    if complexity_score(question) < 0.5:
        retry = slm_generate(f"Answer briefly: {question}")
        if not is_garbage(retry):
            return "SLM", retry

    return "LLM", llm_generate(question)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Neural Router</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  :root{
    --bg:#080808;--s0:#0d0d0d;--s1:#111;--s2:#161616;
    --b1:#1c1c1c;--b2:#262626;--b3:#303030;
    --text:#d8d8d8;--sub:#555;--sub2:#333;--white:#f0f0f0;
    --blue:#4a9eff;--green:#3dd68c;--amber:#f0a500;--red:#ff5f5f;
  }
  html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;overflow:hidden;}
  .app{height:100vh;display:flex;flex-direction:column;}

  .hd{height:50px;flex-shrink:0;display:flex;align-items:center;padding:0 22px;gap:14px;border-bottom:1px solid var(--b1);background:var(--s0);}
  .hd-logo{display:flex;align-items:center;gap:8px;}
  .hd-icon{width:22px;height:22px;border-radius:5px;background:linear-gradient(135deg,#4a9eff22,#3dd68c22);border:1px solid #4a9eff33;display:grid;place-items:center;font-size:11px;}
  .hd-name{font-size:13px;font-weight:500;color:var(--white);letter-spacing:-0.3px;}
  .hd-sep{width:1px;height:16px;background:var(--b2);}
  .hd-meta{font-size:11px;color:var(--sub);letter-spacing:0.1px;}
  .hd-right{margin-left:auto;display:flex;gap:5px;align-items:center;}
  .badge{font-size:9px;font-weight:500;padding:2px 7px;border-radius:3px;letter-spacing:0.4px;text-transform:uppercase;}
  .badge-slm   {color:#4a9eff;background:#4a9eff0f;border:1px solid #4a9eff22;}
  .badge-llm   {color:#a78bfa;background:#a78bfa0f;border:1px solid #a78bfa22;}
  .badge-lookup{color:#3dd68c;background:#3dd68c0f;border:1px solid #3dd68c22;}
  .badge-math  {color:#f0a500;background:#f0a5000f;border:1px solid #f0a50022;}

  .chat{flex:1;overflow-y:auto;}
  .chat::-webkit-scrollbar{width:2px;}
  .chat::-webkit-scrollbar-thumb{background:var(--b2);}
  .msgs{max-width:700px;margin:0 auto;padding:36px 22px 16px;display:flex;flex-direction:column;gap:28px;min-height:100%;}

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
  .msg.user      .av{background:var(--s2);border:1px solid var(--b1);color:var(--sub);}
  .msg.assistant .av{background:var(--s1);border:1px solid var(--b1);color:var(--sub);}
  .mc{max-width:590px;display:flex;flex-direction:column;gap:5px;}
  .msg.user .mc{align-items:flex-end;}
  .route-label{display:inline-flex;align-items:center;gap:5px;font-size:9px;font-weight:500;letter-spacing:0.5px;text-transform:uppercase;padding-left:1px;}
  .route-dot{width:4px;height:4px;border-radius:50%;}
  .rl-slm   {color:#4a9eff;} .rl-slm    .route-dot{background:#4a9eff;}
  .rl-llm   {color:#a78bfa;} .rl-llm    .route-dot{background:#a78bfa;}
  .rl-lookup{color:#3dd68c;} .rl-lookup .route-dot{background:#3dd68c;}
  .rl-math  {color:#f0a500;} .rl-math   .route-dot{background:#f0a500;}
  .rl-error {color:var(--red);} .rl-error .route-dot{background:var(--red);}
  .rl-default{color:var(--sub);}
  .bub{padding:11px 15px;border-radius:9px;font-size:13px;line-height:1.75;word-break:break-word;}
  .msg.user      .bub{background:var(--s2);border:1px solid var(--b1);color:#c8c8c8;border-top-right-radius:2px;}
  .msg.assistant .bub{background:var(--s1);border:1px solid var(--b1);color:#b8b8b8;border-top-left-radius:2px;}

  .dots{display:flex;gap:4px;padding:3px 0;}
  .dots span{width:4px;height:4px;border-radius:50%;background:var(--b3);animation:bounce 1.4s ease infinite;}
  .dots span:nth-child(2){animation-delay:.18s;}
  .dots span:nth-child(3){animation-delay:.36s;}
  @keyframes bounce{0%,60%,100%{transform:translateY(0);background:var(--b3)}30%{transform:translateY(-5px);background:var(--sub);}}

  .inp-zone{flex-shrink:0;padding:10px 22px 18px;border-top:1px solid var(--b1);background:var(--s0);}
  .inp-inner{max-width:700px;margin:0 auto;}
  .inp-box{display:flex;align-items:flex-end;gap:8px;background:var(--s1);border:1px solid var(--b1);border-radius:9px;padding:10px 13px;transition:border-color .15s;}
  .inp-box:focus-within{border-color:var(--b3);}
  textarea{flex:1;background:transparent;border:none;outline:none;color:var(--text);font-size:13px;resize:none;max-height:100px;line-height:1.65;font-family:'Inter',sans-serif;font-weight:400;}
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
    <span class="hd-meta">NanoQA · Llama · v5.0</span>
    <div class="hd-right">
      <span class="badge badge-slm">SLM</span>
      <span class="badge badge-llm">LLM</span>
      <span class="badge badge-lookup">Lookup</span>
      <span class="badge badge-math">Math</span>
    </div>
  </div>

  <div class="chat" id="chat">
    <div class="msgs" id="msgs">
      <div class="welcome" id="welcome">
        <div class="w-ring">
          <svg viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="1.5" stroke-linecap="round">
            <polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
          </svg>
        </div>
        <h1>What do you want to know?</h1>
        <p>Capitals · Facts · Definitions · Scientific math answered instantly.<br/>Complex questions routed to Llama via Ollama.</p>
        <div class="chips">
          <div class="chip" onclick="ex('Capital of Lithuania?')">Capital of Lithuania?</div>
          <div class="chip" onclick="ex('sqrt(144)')">sqrt(144)</div>
          <div class="chip" onclick="ex('sin(pi/2)')">sin(π/2)</div>
          <div class="chip" onclick="ex('what is machine learning')">What is ML?</div>
          <div class="chip" onclick="ex('log(1000)')">log(1000)</div>
          <div class="chip" onclick="ex('who invented python')">Who invented Python?</div>
          <div class="chip" onclick="ex('factorial(10)')">10!</div>
          <div class="chip" onclick="ex('speed of light')">Speed of light</div>
          <div class="chip" onclick="ex('Explain neural networks in detail')">Explain neural nets</div>
          <div class="chip" onclick="ex('2**10')">2^10</div>
        </div>
      </div>
    </div>
  </div>

  <div class="inp-zone">
    <div class="inp-inner">
      <div class="inp-box">
        <textarea id="inp" rows="1" placeholder="Ask anything or enter a math expression…"
          onkeydown="onKey(event)" oninput="rsz(this)"></textarea>
        <button class="go" id="btn" onclick="send()">
          <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
        </button>
      </div>
      <div class="hint">enter to send &nbsp;·&nbsp; shift+enter for newline &nbsp;·&nbsp; supports sin, cos, sqrt, log, factorial…</div>
    </div>
  </div>
</div>

<script>
const chat=document.getElementById('chat');
const msgs=document.getElementById('msgs');
const inp=document.getElementById('inp');
const btn=document.getElementById('btn');

function rsz(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,100)+'px';}
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function ex(t){inp.value=t;send();}

function routeClass(r){
  if(!r) return 'rl-default';
  const s=r.toLowerCase();
  if(s==='slm')    return 'rl-slm';
  if(s==='llm')    return 'rl-llm';
  if(s==='lookup') return 'rl-lookup';
  if(s==='math')   return 'rl-math';
  if(s==='error')  return 'rl-error';
  return 'rl-default';
}

function addMsg(role,text,route){
  document.getElementById('welcome')?.remove();
  const m=document.createElement('div');m.className=`msg ${role}`;
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
  mc.appendChild(b);m.appendChild(av);m.appendChild(mc);
  msgs.appendChild(m);chat.scrollTop=chat.scrollHeight;
}

function addTyping(){
  const m=document.createElement('div');m.className='msg assistant';m.id='typing';
  m.innerHTML='<div class="av">nr</div><div class="mc"><div class="bub"><div class="dots"><span></span><span></span><span></span></div></div></div>';
  msgs.appendChild(m);chat.scrollTop=chat.scrollHeight;
}

async function send(){
  const text=inp.value.trim();if(!text)return;
  btn.disabled=true;inp.value='';inp.style.height='auto';
  addMsg('user',text,null);addTyping();
  try{
    const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
    const data=await res.json();
    document.getElementById('typing')?.remove();
    addMsg('assistant',data.answer,data.route);
  }catch(e){
    document.getElementById('typing')?.remove();
    addMsg('assistant','Error — is Ollama running?','error');
  }
  btn.disabled=false;inp.focus();
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.json
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"answer": "Please ask something!", "route": "error"})
    route, answer = router(question)
    return jsonify({"answer": answer, "route": route})

if __name__ == "__main__":
    def find_free_port(start=5000):
        port = start
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
            port += 1
    port = find_free_port()
    print("\n" + "="*46)
    print("  Neural Router v5.0  (NanoQA + Llama)")
    print(f"  http://localhost:{port}")
    print("="*46 + "\n")
    app.run(debug=False, port=port)