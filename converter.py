import json
import re

# ================================
# 🔧 NORMALIZATION
# ================================
def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = text.strip()
    return text

# ================================
# 🔧 EXPAND REFERENCES
# ================================
def expand_reference(ref):
    mapping = {
        "shakespeare": "william shakespeare",
        "rowling": "jk rowling",
        "orwell": "george orwell",
        "tolkien": "jrr tolkien",
        "shelley": "mary shelley",
        "stoker": "bram stoker",
        "fitzgerald": "f scott fitzgerald",
        "carroll": "lewis carroll",
        "austen": "jane austen",
        "fleming": "alexander fleming",
        "newton": "isaac newton",
        "bell": "alexander graham bell",
        "edison": "thomas edison",
        "rossum": "guido van rossum",
        "curie": "marie curie",
        "wright": "wright brothers",
        "darwin": "charles darwin",
        "marconi": "guglielmo marconi",
        "watt": "james watt",
        "gutenberg": "johannes gutenberg",
        "paris": "paris",
        "tokyo": "tokyo",
        "delhi": "new delhi",
        "berlin": "berlin",
        "canberra": "canberra",
        "russia": "russia",
        "pacific": "pacific ocean",
        "everest": "mount everest",
        "nile": "nile river",
        "mitochondria": "mitochondria",
        "h2o": "h2o",
        "cheetah": "cheetah",
        "blue whale": "blue whale",
        "giraffe": "giraffe",
        "carbon": "carbon dioxide",
        "vinci": "leonardo da vinci",
        "gogh": "vincent van gogh",
        "picasso": "pablo picasso",
        "munch": "edvard munch",
        "michelangelo": "michelangelo",
        "eiffel": "gustave eiffel",
        "beethoven": "ludwig van beethoven",
        "central": "central processing unit",
        "hypertext": "hypertext",
        "random": "random access memory",
        "application": "application programming interface",
        "structured": "structured query language",
        "uniform": "uniform resource locator",
        "1939": "1939",
        "1945": "1945",
        "1947": "1947",
        "1989": "1989",
        "1889": "1889",
        "1789": "1789",
        "washington": "george washington",
        "nehru": "jawaharlal nehru"
    }
    return mapping.get(ref.lower(), ref)

# ================================
# 🚀 LOAD ORIGINAL DATA
# ================================
with open("visualizations/results.json") as f:
    data = json.load(f)

new_data = []

# ================================
# 🔄 CONVERT
# ================================
for d in data:
    ref_raw = d.get("expected", "")
    ref_expanded = expand_reference(ref_raw)

    new_data.append({
        "question": d["question"],
        "reference": normalize(ref_expanded),
        "answer": normalize(d["answer"]),
        "route": d["route"],
        "confidence": d.get("confidence"),
        "latency": d.get("time_ms")
    })

# ================================
# 💾 SAVE
# ================================
with open("visualizations/results_fixed.json", "w") as f:
    json.dump(new_data, f, indent=2)

print("✅ Fixed + normalized file created")