import torch
import ollama
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Load your fine-tuned SLM ───────────────────────────────────
print("Loading fine-tuned SLM...")
MODEL_PATH = "./models/slm_gpt2"
tokenizer  = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model      = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
device     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model      = model.to(device)
model.eval()
print(f"SLM loaded on {device}!")

# ── Confidence Scorer ──────────────────────────────────────────
def get_confidence(question):
    prompt    = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output    = model(input_ids)
        probs     = torch.softmax(output.logits, dim=-1)
        top_probs = probs[0, -10:, :].max(dim=-1).values
    return round(top_probs.mean().item(), 4)

# ── Math Detection ─────────────────────────────────────────────
def evaluate_math(question):
    clean = question.lower().strip()
    clean = re.sub(r'(what is|what\'s|calculate|compute|solve|evaluate|find)', '', clean)
    clean = clean.replace("plus", "+").replace("minus", "-")
    clean = clean.replace("times", "*").replace("multiplied by", "*")
    clean = clean.replace("divided by", "/").replace("power of", "**")
    clean = clean.replace("^", "**").strip().rstrip("?").strip()
    if re.match(r'^[\d\s\+\-\*\/\(\)\.\%\*]+$', clean) and any(c.isdigit() for c in clean):
        try:
            result = eval(clean)
            return str(round(result, 6)) if isinstance(result, float) else str(result)
        except:
            return None
    return None

# ── Quick Definition Dictionary ────────────────────────────────
QUICK_DEFINITIONS = {
    "ai":               "AI (Artificial Intelligence) is the simulation of human intelligence by machines.",
    "machine learning": "Machine Learning is a subset of AI where systems learn from data to improve automatically.",
    "deep learning":    "Deep Learning is a subset of ML using neural networks with many layers.",
    "neural network":   "A Neural Network is a computing system inspired by biological brains, made of connected nodes.",
    "python":           "Python is a high-level, general-purpose programming language known for simplicity.",
    "algorithm":        "An Algorithm is a step-by-step set of instructions to solve a problem.",
    "database":         "A Database is an organized collection of structured data stored electronically.",
    "api":              "An API (Application Programming Interface) allows different software to communicate.",
    "cpu":              "A CPU (Central Processing Unit) is the primary processor of a computer.",
    "gpu":              "A GPU (Graphics Processing Unit) is a processor designed for parallel computations.",
    "ram":              "RAM (Random Access Memory) is temporary memory used by a computer for active tasks.",
    "http":             "HTTP (HyperText Transfer Protocol) is the foundation of data communication on the web.",
    "html":             "HTML (HyperText Markup Language) is the standard language for creating web pages.",
    "css":              "CSS (Cascading Style Sheets) is used to style and layout web pages.",
    "javascript":       "JavaScript is a programming language that makes web pages interactive.",
    "blockchain":       "Blockchain is a distributed ledger technology where data is stored in linked blocks.",
    "cloud computing":  "Cloud Computing is the delivery of computing services over the internet.",
    "encryption":       "Encryption is the process of converting data into a coded format to prevent unauthorized access.",
    "compiler":         "A Compiler translates high-level programming code into machine code.",
    "os":               "An OS (Operating System) manages hardware and software resources on a computer.",
    "slm":              "An SLM (Small Language Model) is a compact AI model designed for efficient inference.",
    "llm":              "An LLM (Large Language Model) is a large AI model trained on massive text datasets.",
    "transformer":      "A Transformer is a neural network architecture that uses attention mechanisms for sequence tasks.",
    "router":           "A Router directs data or queries to the appropriate handler based on complexity.",
    "internet":         "The Internet is a global network connecting millions of computers worldwide.",
    "software":         "Software is a collection of programs and data that tell a computer how to work.",
    "hardware":         "Hardware refers to the physical components of a computer system.",
    "bandwidth":        "Bandwidth is the maximum rate of data transfer across a network.",
    "cache":            "Cache is a small, fast memory that stores frequently accessed data for quick retrieval.",
    "firewall":         "A Firewall is a security system that monitors and controls network traffic.",
    "vpn":              "A VPN (Virtual Private Network) encrypts internet connections for privacy and security.",
    "binary":           "Binary is a number system using only 0s and 1s, the foundation of computing.",
    "debugging":        "Debugging is the process of finding and fixing errors in code.",
    "framework":        "A Framework is a reusable set of tools and libraries for building software.",
    "recursion":        "Recursion is when a function calls itself to solve smaller instances of a problem.",
    "variable":         "A Variable is a named storage location that holds a value in programming.",
    "loop":             "A Loop is a programming construct that repeats a block of code multiple times.",
    "function":         "A Function is a reusable block of code that performs a specific task.",
    "class":            "A Class is a blueprint for creating objects in object-oriented programming.",
    "object":           "An Object is an instance of a class containing data and methods.",
    "inheritance":      "Inheritance allows a class to reuse properties and methods from another class.",
    "photosynthesis":   "Photosynthesis is the process by which plants convert sunlight into food using CO2 and water.",
    "gravity":          "Gravity is the force that attracts objects with mass toward each other.",
    "democracy":        "Democracy is a system of government where citizens vote to elect their representatives.",
    "inflation":        "Inflation is the rate at which the general price level of goods and services rises over time.",
    "dna":              "DNA (Deoxyribonucleic Acid) is the molecule that carries genetic information in living organisms.",
    "atom":             "An Atom is the smallest unit of a chemical element that retains its properties.",
    "evolution":        "Evolution is the process by which species change over generations through natural selection.",
    "climate change":   "Climate Change refers to long-term shifts in global temperatures and weather patterns.",
    "renaissance":      "The Renaissance was a cultural and intellectual movement in Europe from the 14th to 17th century.",
}

def check_definition(question):
    q = question.lower().strip().rstrip("?").strip()
    patterns = [
        r'what is (?:a |an |the )?(.+)',
        r'what are (?:a |an |the )?(.+)',
        r'define (?:a |an |the )?(.+)',
        r'what does (.+) mean',
        r'meaning of (.+)',
        r'tell me about (?:a |an |the )?(.+)',
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            term = match.group(1).strip().rstrip("?").strip()
            if term in QUICK_DEFINITIONS:
                return "preset", QUICK_DEFINITIONS[term]
            return "unknown", term
    return None, None

# ── Summarization Detection ────────────────────────────────────
def check_summarization(question):
    q = question.lower().strip()
    summarize_triggers = ["summarize", "summarise", "tldr", "sum up", "brief summary", "in short"]
    if any(t in q for t in summarize_triggers):
        words = question.split()
        return "short" if len(words) <= 80 else "long"
    return None

# ── Complexity Scorer ──────────────────────────────────────────
def complexity_score(question):
    score = 0.0
    q     = question.lower()
    words = len(q.split())
    if words > 20:    score += 0.3
    elif words > 10:  score += 0.1
    if any(w in q for w in ["explain", "why", "how does", "analyze", "analyse", "compare",
                              "difference between", "pros and cons", "advantages", "disadvantages",
                              "in detail", "elaborate", "discuss"]):
        score += 0.4
    if any(w in q for w in ["what is", "who is", "when did", "where is", "capital", "define"]):
        score -= 0.2
    return max(0.0, min(1.0, score))

# ── SLM Generate ──────────────────────────────────────────────
def slm_generate(question):
    prompt    = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=60,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in full:
        return full.split("Answer:")[-1].strip()
    return full

# ── LLM Generate via Ollama ────────────────────────────────────
def llm_generate(question):
    print("  [Calling Mistral via Ollama...]")
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": question}]
    )
    return response["message"]["content"]

# ── Smart Definition Handler ───────────────────────────────────
def handle_unknown_definition(term, question):
    confidence = get_confidence(question)
    print(f"  [Unknown definition: '{term}' | SLM confidence: {confidence}]")
    if confidence >= 0.60:
        slm_answer = slm_generate(question)
        words      = slm_answer.split()
        unique     = set(words)
        is_garbage = len(words) < 3 or (len(words) > 5 and len(unique) < 3)
        if not is_garbage:
            print(f"  [Route: SLM (GPT-2) — confident enough]")
            return "SLM (GPT-2)", slm_answer
        else:
            print(f"  [SLM answer looks bad — escalating to Mistral]")
            return "LLM (Mistral)", llm_generate(question)
    else:
        print(f"  [Route: LLM (Mistral) — low confidence for unknown definition]")
        return "LLM (Mistral)", llm_generate(question)

# ── Master Router ──────────────────────────────────────────────
def router(question, confidence_threshold=0.62, complexity_threshold=0.3):

    # 1. Math — instant, no model
    math_result = evaluate_math(question)
    if math_result:
        print(f"  [Route: Math Engine]")
        return "Math Engine", math_result

    # 2. Definition check
    def_type, def_result = check_definition(question)
    if def_type == "preset":
        print(f"  [Route: Definition Engine (preset)]")
        return "Definition Engine", def_result
    elif def_type == "unknown":
        return handle_unknown_definition(def_result, question)

    # 3. Summarization
    summary_type = check_summarization(question)
    if summary_type == "short":
        text = re.sub(r'(summarize|summarise|tldr|sum up|brief summary|in short)[:\s]*', '',
                      question, flags=re.IGNORECASE).strip()
        words = text.split()
        if len(words) <= 15:
            print(f"  [Route: Direct Summarizer]")
            return "Direct Summarizer", f"Summary: {text.capitalize()}"
        else:
            print(f"  [Route: LLM — summarization]")
            return "LLM (Mistral)", llm_generate(f"Summarize this in one sentence: {text}")
    elif summary_type == "long":
        text = re.sub(r'(summarize|summarise|tldr|sum up|brief summary|in short)[:\s]*', '',
                      question, flags=re.IGNORECASE).strip()
        print(f"  [Route: LLM — long summarization]")
        return "LLM (Mistral)", llm_generate(f"Summarize this concisely: {text}")

    # 4. General confidence + complexity routing
    confidence = get_confidence(question)
    complexity = complexity_score(question)
    print(f"  [Confidence: {confidence} | Complexity: {complexity}]")

    if complexity >= complexity_threshold or confidence < confidence_threshold:
        print(f"  [Route: LLM (Mistral)]")
        return "LLM (Mistral)", llm_generate(question)
    else:
        print(f"  [Route: SLM (GPT-2)]")
        return "SLM (GPT-2)", slm_generate(question)

# ── Interactive Chat ───────────────────────────────────────────
print("\n" + "="*60)
print("  AUTOMATIC SLM ↔ LLM ROUTER  v4.0")
print("  Math | Dictionary | Smart Definitions | SLM | Mistral")
print("="*60)
print("\nExamples:")
print("  Math:          '2 + 2', '100 * 5', '50 divided by 2'")
print("  Known def:     'what is AI', 'define machine learning'")
print("  Unknown def:   'what is photosynthesis', 'what is gravity'")
print("  Simple QA:     'what is the capital of France'")
print("  Summarize:     'summarize: I had a great dinner tonight'")
print("  Complex:       'explain how transformers work in detail'")
print("\nType 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        print("Bye!")
        break
    if not question:
        continue
    route, answer = router(question)
    print(f"\n[{route}]: {answer}\n")