"""
generate_training_data.py
Run this on your Mac FIRST before uploading to Kaggle:
    cd /Users/venisa/slm_project
    source venv/bin/activate
    python3 generate_training_data.py

Output: new_handcrafted_qa.json
Then upload that file to Kaggle as a dataset.
"""

import json, random
random.seed(42)

# ─────────────────────────────────────────────
# ALL THE FACTS YOUR SLM NEEDS TO KNOW
# ─────────────────────────────────────────────

CAPITALS = {
    "france":"Paris","germany":"Berlin","italy":"Rome","spain":"Madrid",
    "portugal":"Lisbon","netherlands":"Amsterdam","belgium":"Brussels",
    "switzerland":"Bern","austria":"Vienna","sweden":"Stockholm",
    "norway":"Oslo","denmark":"Copenhagen","finland":"Helsinki",
    "poland":"Warsaw","czech republic":"Prague","hungary":"Budapest",
    "romania":"Bucharest","bulgaria":"Sofia","greece":"Athens",
    "ukraine":"Kyiv","russia":"Moscow","uk":"London",
    "united kingdom":"London","ireland":"Dublin","croatia":"Zagreb",
    "serbia":"Belgrade","slovakia":"Bratislava","albania":"Tirana",
    "belarus":"Minsk","latvia":"Riga","lithuania":"Vilnius",
    "estonia":"Tallinn","luxembourg":"Luxembourg City",
    "iceland":"Reykjavik","cyprus":"Nicosia",
    "china":"Beijing","japan":"Tokyo","india":"New Delhi",
    "south korea":"Seoul","north korea":"Pyongyang",
    "indonesia":"Jakarta","pakistan":"Islamabad","bangladesh":"Dhaka",
    "vietnam":"Hanoi","thailand":"Bangkok","malaysia":"Kuala Lumpur",
    "philippines":"Manila","singapore":"Singapore",
    "nepal":"Kathmandu","mongolia":"Ulaanbaatar",
    "afghanistan":"Kabul","iran":"Tehran","iraq":"Baghdad",
    "turkey":"Ankara","syria":"Damascus","lebanon":"Beirut",
    "jordan":"Amman","israel":"Jerusalem","saudi arabia":"Riyadh",
    "uae":"Abu Dhabi","united arab emirates":"Abu Dhabi",
    "qatar":"Doha","kuwait":"Kuwait City","oman":"Muscat",
    "uzbekistan":"Tashkent","kazakhstan":"Astana",
    "azerbaijan":"Baku","armenia":"Yerevan","georgia":"Tbilisi",
    "usa":"Washington D.C.","united states":"Washington D.C.",
    "america":"Washington D.C.","canada":"Ottawa",
    "mexico":"Mexico City","brazil":"Brasilia",
    "argentina":"Buenos Aires","colombia":"Bogota","chile":"Santiago",
    "peru":"Lima","venezuela":"Caracas","ecuador":"Quito",
    "cuba":"Havana","jamaica":"Kingston","panama":"Panama City",
    "costa rica":"San Jose","guatemala":"Guatemala City",
    "nigeria":"Abuja","south africa":"Pretoria","egypt":"Cairo",
    "kenya":"Nairobi","ethiopia":"Addis Ababa","ghana":"Accra",
    "tanzania":"Dodoma","uganda":"Kampala","rwanda":"Kigali",
    "somalia":"Mogadishu","sudan":"Khartoum","libya":"Tripoli",
    "tunisia":"Tunis","algeria":"Algiers","morocco":"Rabat",
    "angola":"Luanda","namibia":"Windhoek","botswana":"Gaborone",
    "australia":"Canberra","new zealand":"Wellington",
}

FACTS = {
    "speed of light":"approximately 299,792,458 metres per second",
    "boiling point of water":"100 degrees Celsius or 212 degrees Fahrenheit",
    "freezing point of water":"0 degrees Celsius or 32 degrees Fahrenheit",
    "chemical formula of water":"H2O",
    "chemical formula of carbon dioxide":"CO2",
    "chemical formula of salt":"NaCl",
    "atomic number of hydrogen":"1",
    "atomic number of carbon":"6",
    "atomic number of oxygen":"8",
    "atomic number of gold":"79",
    "atomic number of iron":"26",
    "number of bones in the human body":"206",
    "number of chromosomes in humans":"46, arranged in 23 pairs",
    "number of planets in the solar system":"8 — Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune",
    "largest planet":"Jupiter",
    "smallest planet":"Mercury",
    "hottest planet":"Venus",
    "largest ocean":"the Pacific Ocean",
    "largest continent":"Asia",
    "longest river":"the Nile at about 6,650 kilometres",
    "highest mountain":"Mount Everest at 8,849 metres",
    "deepest ocean point":"the Mariana Trench at about 11,034 metres",
    "largest country by area":"Russia",
    "smallest country":"Vatican City",
    "fastest land animal":"the cheetah, reaching up to 120 km/h",
    "largest land animal":"the African elephant",
    "largest animal in the world":"the blue whale",
    "powerhouse of the cell":"the mitochondria — it produces ATP energy",
    "number of elements in the periodic table":"118",
    "value of pi":"approximately 3.14159265358979",
    "value of e":"approximately 2.71828182845904",
    "speed of sound in air":"approximately 343 metres per second",
    "normal human body temperature":"37 degrees Celsius or 98.6 degrees Fahrenheit",
    "distance from earth to sun":"about 149.6 million kilometres",
    "distance from earth to moon":"about 384,400 kilometres",
    "age of the universe":"approximately 13.8 billion years",
    "age of the earth":"approximately 4.5 billion years",
    "number of teeth in adults":"32 teeth including 4 wisdom teeth",
    "national animal of india":"the Bengal Tiger",
    "national animal of australia":"the kangaroo and emu",
    "currency of usa":"the US Dollar",
    "currency of uk":"the Pound Sterling",
    "currency of india":"the Indian Rupee",
    "currency of japan":"the Japanese Yen",
    "currency of europe":"the Euro",
    "how many continents are there":"7 continents",
    "how many oceans are there":"5 oceans",
    "layers of the earth":"inner core, outer core, mantle, and crust",
    "how many chambers does the heart have":"4 chambers",
}

PEOPLE = {
    "who wrote romeo and juliet":"William Shakespeare",
    "who wrote hamlet":"William Shakespeare",
    "who wrote macbeth":"William Shakespeare",
    "who wrote othello":"William Shakespeare",
    "who wrote 1984":"George Orwell",
    "who wrote animal farm":"George Orwell",
    "who wrote harry potter":"J.K. Rowling",
    "who wrote the great gatsby":"F. Scott Fitzgerald",
    "who wrote pride and prejudice":"Jane Austen",
    "who wrote to kill a mockingbird":"Harper Lee",
    "who wrote the hobbit":"J.R.R. Tolkien",
    "who wrote lord of the rings":"J.R.R. Tolkien",
    "who wrote the catcher in the rye":"J.D. Salinger",
    "who wrote don quixote":"Miguel de Cervantes",
    "who wrote moby dick":"Herman Melville",
    "who discovered penicillin":"Alexander Fleming in 1928",
    "who invented the telephone":"Alexander Graham Bell in 1876",
    "who invented the airplane":"the Wright Brothers, Orville and Wilbur Wright, in 1903",
    "who invented the light bulb":"Thomas Edison in 1879",
    "who invented python":"Guido van Rossum, first released in 1991",
    "who invented the car":"Karl Benz in 1885",
    "who invented radio":"Guglielmo Marconi in the 1890s",
    "who invented the computer":"Charles Babbage conceived it; Alan Turing laid modern foundations",
    "who invented the internet":"Tim Berners-Lee invented the World Wide Web in 1989",
    "who painted the mona lisa":"Leonardo da Vinci",
    "who painted starry night":"Vincent van Gogh in 1889",
    "who painted the sistine chapel":"Michelangelo",
    "who discovered gravity":"Isaac Newton formulated the law of universal gravitation",
    "who was albert einstein":"a German-born physicist who developed the theory of relativity",
    "who was isaac newton":"an English mathematician who formulated laws of motion and gravitation",
    "who was charles darwin":"a British naturalist who developed the theory of evolution",
    "who was mahatma gandhi":"an Indian independence leader known for nonviolent resistance",
    "who was napoleon bonaparte":"a French military leader and emperor who conquered much of Europe",
    "who was cleopatra":"the last active ruler of the Ptolemaic Kingdom of Egypt",
    "who was julius caesar":"a Roman general and statesman who played a key role in the fall of the Roman Republic",
    "who was shakespeare":"William Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language",
    "first president of the united states":"George Washington",
    "who was george washington":"the first president of the United States, serving from 1789 to 1797",
}

EVENTS = {
    "when did world war 2 start":"September 1, 1939, when Germany invaded Poland",
    "when did world war 2 end":"September 2, 1945, with Japan's formal surrender",
    "when did world war 1 start":"July 28, 1914",
    "when did world war 1 end":"November 11, 1918",
    "when did india get independence":"August 15, 1947",
    "when did usa get independence":"July 4, 1776",
    "when did the french revolution happen":"1789, lasting until 1799",
    "when did the berlin wall fall":"November 9, 1989",
    "when did the cold war end":"1991 with the dissolution of the Soviet Union",
    "when did man first land on the moon":"July 20, 1969, during the Apollo 11 mission",
}

DEFINITIONS = {
    "artificial intelligence":"the simulation of human intelligence by machines programmed to think and learn",
    "machine learning":"a subset of AI where systems learn automatically from data and improve from experience",
    "deep learning":"a type of machine learning using multi-layered neural networks to learn complex patterns",
    "neural network":"a computing system inspired by the human brain, made of connected layers of nodes",
    "python":"a high-level programming language known for simple syntax and readability",
    "algorithm":"a step-by-step set of instructions designed to solve a problem",
    "database":"an organised collection of structured data stored and accessed electronically",
    "photosynthesis":"the process by which plants convert sunlight, CO2, and water into glucose and oxygen",
    "gravity":"the fundamental force of attraction between objects that have mass",
    "dna":"Deoxyribonucleic Acid — the molecule that carries genetic instructions for living organisms",
    "democracy":"a system of government where citizens exercise power by voting",
    "inflation":"the rate at which the general level of prices rises, reducing purchasing power",
    "evolution":"the process by which species change over generations through natural selection",
    "photon":"the elementary particle of light and electromagnetic radiation",
    "atom":"the smallest unit of a chemical element, with a nucleus of protons and neutrons and surrounding electrons",
    "api":"Application Programming Interface — rules allowing different software to communicate",
    "encryption":"converting data into a coded format so only authorised parties can read it",
    "cloud computing":"delivery of computing services like servers and storage over the internet",
    "recursion":"a programming technique where a function calls itself to solve smaller instances of the same problem",
    "overfitting":"when a model learns training data too well including noise, and performs poorly on new data",
    "blockchain":"a distributed ledger where data is stored in linked tamper-resistant blocks",
    "osmosis":"the movement of water molecules through a semi-permeable membrane from low to high concentration",
    "mitosis":"the process of cell division resulting in two identical daughter cells",
    "newton first law":"an object at rest stays at rest and an object in motion stays in motion unless acted on by a force",
    "newton second law":"force equals mass times acceleration, F equals ma",
    "newton third law":"for every action there is an equal and opposite reaction",
    "pythagorean theorem":"in a right triangle, a squared plus b squared equals c squared",
    "what is html":"HyperText Markup Language — the standard language used to structure web pages",
    "what is css":"Cascading Style Sheets — controls the visual appearance of web pages",
    "what does cpu stand for":"Central Processing Unit — the main processor that executes instructions",
    "what does gpu stand for":"Graphics Processing Unit — designed for parallel processing and graphics",
    "what does ram stand for":"Random Access Memory — short-term memory a computer uses for active tasks",
    "what does html stand for":"HyperText Markup Language",
    "what does http stand for":"HyperText Transfer Protocol — foundation of web communication",
    "what does dna stand for":"Deoxyribonucleic Acid",
    "what does ai stand for":"Artificial Intelligence",
}

# ─────────────────────────────────────────────
# QUESTION TEMPLATES — multiple phrasings per fact
# ─────────────────────────────────────────────

def capital_variants(country, capital):
    c = country.title()
    return [
        (f"What is the capital of {c}?",               f"The capital of {c} is {capital}."),
        (f"What is {c}'s capital city?",                f"{c}'s capital city is {capital}."),
        (f"Capital of {c}?",                            f"{capital}."),
        (f"Where is the capital of {c}?",               f"The capital of {c} is {capital}."),
        (f"Which city is the capital of {c}?",          f"{capital} is the capital of {c}."),
        (f"Name the capital city of {c}.",              f"The capital city of {c} is {capital}."),
        (f"Tell me the capital of {c}.",                f"The capital of {c} is {capital}."),
        (f"What city does {c} govern from?",            f"{c} governs from {capital}."),
        (f"What is the main city of {c}?",              f"The capital and main city of {c} is {capital}."),
        (f"Hey, what is the capital of {c}?",           f"The capital of {c} is {capital}."),
    ]

def fact_variants(key, answer):
    k = key.capitalize()
    return [
        (f"What is the {key}?",              f"The {key} is {answer}."),
        (f"Tell me the {key}.",              f"The {key} is {answer}."),
        (f"What is the value of {key}?",     f"The {key} is {answer}."),
        (f"Can you tell me the {key}?",      f"The {key} is {answer}."),
        (f"{k}?",                            f"The {key} is {answer}."),
        (f"What do you know about {key}?",   f"The {key} is {answer}."),
    ]

def person_variants(key, answer):
    thing = key
    for starter in ["who wrote ","who invented ","who discovered ","who painted ","who was "]:
        if key.startswith(starter):
            thing = key[len(starter):]
            break

    pairs = [
        (f"{key.capitalize()}?",                              f"{answer}."),
        (f"Can you tell me {key}?",                          f"{answer}."),
        (f"I want to know {key}.",                           f"{answer}."),
        (f"Please tell me {key}.",                           f"{answer}."),
        (f"Do you know {key}?",                              f"{answer}."),
        (f"Hey, {key}?",                                     f"{answer}."),
    ]
    if "wrote" in key:
        pairs += [
            (f"Who is the author of {thing}?",               f"The author of {thing} is {answer}."),
            (f"Who created {thing}?",                        f"{thing} was created by {answer}."),
            (f"Who is {thing} written by?",                  f"{thing} is written by {answer}."),
            (f"Tell me the author of {thing}.",              f"The author of {thing} is {answer}."),
        ]
    elif "invented" in key or "discovered" in key:
        pairs += [
            (f"Who created {thing}?",                        f"{thing} was created by {answer}."),
            (f"Who is responsible for {thing}?",             f"{answer} is responsible for {thing}."),
            (f"Who made {thing}?",                           f"{thing} was made by {answer}."),
            (f"Who is credited with inventing {thing}?",     f"{answer} is credited with inventing {thing}."),
        ]
    elif "painted" in key:
        pairs += [
            (f"Who created {thing}?",                        f"{thing} was created by {answer}."),
            (f"Who is the artist of {thing}?",               f"The artist of {thing} is {answer}."),
            (f"Who made {thing}?",                           f"{thing} was made by {answer}."),
        ]
    elif "was" in key:
        pairs += [
            (f"Tell me about {thing}.",                      f"{thing.capitalize()} was {answer}."),
            (f"What do you know about {thing}?",             f"{thing.capitalize()} was {answer}."),
            (f"Describe {thing}.",                           f"{thing.capitalize()} was {answer}."),
        ]
    return pairs

def event_variants(key, answer):
    event = key.replace("when did ", "")
    return [
        (f"{key.capitalize()}?",                        f"{event.capitalize()} happened in {answer}."),
        (f"What year did {event}?",                     f"{event.capitalize()} in {answer}."),
        (f"In what year did {event}?",                  f"{event.capitalize()} in {answer}."),
        (f"Tell me when {event}.",                      f"{event.capitalize()} happened on {answer}."),
        (f"What is the date {event}?",                  f"{answer}."),
        (f"Do you know when {event}?",                  f"Yes, {event} happened in {answer}."),
        (f"What date did {event}?",                     f"{event.capitalize()} on {answer}."),
        (f"Can you tell me when {event}?",              f"{event.capitalize()} in {answer}."),
    ]

def definition_variants(term, definition):
    return [
        (f"What is {term}?",                        f"{term.capitalize()} is {definition}."),
        (f"Define {term}.",                         f"{term.capitalize()} is {definition}."),
        (f"What does {term} mean?",                 f"{term.capitalize()} means {definition}."),
        (f"Explain {term}.",                        f"{term.capitalize()} is {definition}."),
        (f"What is the definition of {term}?",      f"The definition of {term.capitalize()} is {definition}."),
        (f"Can you explain {term}?",                f"{term.capitalize()} is {definition}."),
        (f"Tell me about {term}.",                  f"{term.capitalize()} is {definition}."),
        (f"What do you mean by {term}?",            f"{term.capitalize()} is {definition}."),
    ]

# ─────────────────────────────────────────────
# GENERATE ALL QA PAIRS
# ─────────────────────────────────────────────

all_pairs = []
seen      = set()

def add_pairs(variants):
    for q, a in variants:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            all_pairs.append({"question": q, "answer": a})

# Capitals
for country, capital in CAPITALS.items():
    add_pairs(capital_variants(country, capital))

# Facts
for key, answer in FACTS.items():
    add_pairs(fact_variants(key, answer))

# People
for key, answer in PEOPLE.items():
    add_pairs(person_variants(key, answer))

# Events
for key, answer in EVENTS.items():
    add_pairs(event_variants(key, answer))

# Definitions
for term, definition in DEFINITIONS.items():
    add_pairs(definition_variants(term, definition))

# Extra standalone QA that doesn't fit neatly into categories
standalone = [
    ("How many days are in a year?", "There are 365 days in a regular year and 366 in a leap year."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ("How many sides does a triangle have?", "A triangle has 3 sides."),
    ("How many sides does a square have?", "A square has 4 sides."),
    ("How many sides does a hexagon have?", "A hexagon has 6 sides."),
    ("How many sides does an octagon have?", "An octagon has 8 sides."),
    ("What is the square root of 144?", "The square root of 144 is 12."),
    ("What is 2 to the power of 10?", "2 to the power of 10 is 1024."),
    ("What is the sum of angles in a triangle?", "The sum of angles in a triangle is 180 degrees."),
    ("What are the primary colours?", "The primary colours are red, blue, and yellow."),
    ("What causes rainbows?", "Rainbows are caused by refraction and reflection of sunlight through water droplets."),
    ("Why is the sky blue?", "The sky appears blue because the atmosphere scatters shorter blue wavelengths of sunlight more than other colours."),
    ("What is the water cycle?", "The water cycle is the continuous movement of water through evaporation, condensation, and precipitation."),
    ("What is an ecosystem?", "An ecosystem is a community of living organisms interacting with each other and their environment."),
    ("What is the closest star to Earth?", "The closest star to Earth is the Sun. The nearest star outside our solar system is Proxima Centauri."),
    ("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape."),
    ("What is the Big Bang?", "The Big Bang is the scientific theory that the universe began from an extremely hot dense point about 13.8 billion years ago."),
    ("How many stars are in the Milky Way?", "The Milky Way contains an estimated 100 to 400 billion stars."),
    ("What is the periodic table?", "The periodic table is a tabular arrangement of all known chemical elements organised by atomic number."),
    ("What is a virus?", "A virus is a microscopic infectious agent that replicates inside living cells and can cause disease."),
    ("What is a vaccine?", "A vaccine is a biological preparation that provides immunity against a disease by stimulating the immune system."),
    ("What is the immune system?", "The immune system is the body's defence system that protects against infections and diseases."),
    ("How many chambers does the human heart have?", "The human heart has four chambers: left atrium, right atrium, left ventricle, and right ventricle."),
    ("What is the largest desert?", "Antarctica is the largest desert. The Sahara is the largest hot desert."),
    ("Who was the first person to walk on the moon?", "Neil Armstrong was the first person to walk on the moon on July 20, 1969."),
    ("What is E equals mc squared?", "E equals mc squared is Einstein's equation showing that energy and mass are equivalent, where c is the speed of light."),
    ("How many blood types are there?", "There are four main blood types: A, B, AB, and O."),
    ("What is photosynthesis used for?", "Photosynthesis allows plants to convert sunlight into glucose and oxygen for energy and growth."),
    ("Where is the Eiffel Tower?", "The Eiffel Tower is in Paris, France."),
    ("Where is the Great Wall of China?", "The Great Wall of China is in northern China."),
    ("Where is the Taj Mahal?", "The Taj Mahal is in Agra, India."),
    ("Where is the Amazon rainforest?", "The Amazon rainforest is primarily in Brazil, South America."),
    ("What language do people speak in Brazil?", "People in Brazil speak Portuguese."),
    ("What language do people speak in France?", "People in France speak French."),
    ("What language do people speak in Japan?", "People in Japan speak Japanese."),
    ("What language do people speak in Germany?", "People in Germany speak German."),
    ("What is the most widely spoken language?", "English is the most widely spoken language globally. Mandarin has the most native speakers."),
    ("What is the tallest building in the world?", "The Burj Khalifa in Dubai is the tallest building in the world at 828 metres."),
    ("What is the Amazon?", "The Amazon is the world's largest river by discharge and second longest, flowing through South America."),
    ("Who was Nikola Tesla?", "Nikola Tesla was a Serbian-American inventor known for his contributions to electricity and the AC electrical system."),
    ("Who was Thomas Edison?", "Thomas Edison was an American inventor known for developing the phonograph, motion picture camera, and long-lasting light bulb."),
    ("Who was Marie Curie?", "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes."),
    ("Who was Alan Turing?", "Alan Turing was a British mathematician considered the father of computer science and artificial intelligence."),
    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle, a squared plus b squared equals c squared."),
    ("What is Newton's first law?", "Newton's first law states an object stays at rest or in motion unless acted on by an external force."),
    ("What is Newton's second law?", "Newton's second law states that force equals mass times acceleration."),
    ("What is Newton's third law?", "Newton's third law states that for every action there is an equal and opposite reaction."),
]

for q, a in standalone:
    key = q.lower().strip()
    if key not in seen:
        seen.add(key)
        all_pairs.append({"question": q, "answer": a})

# Shuffle
random.shuffle(all_pairs)

# Save
output_path = "new_handcrafted_qa.json"
with open(output_path, "w") as f:
    json.dump(all_pairs, f, indent=2)

print(f"Generated {len(all_pairs):,} unique QA pairs")
print(f"Saved to: {output_path}")
print()
print("Breakdown:")
cap_count  = sum(1 for p in all_pairs if "capital" in p["question"].lower())
who_count  = sum(1 for p in all_pairs if p["question"].lower().startswith("who"))
when_count = sum(1 for p in all_pairs if p["question"].lower().startswith("when"))
def_count  = sum(1 for p in all_pairs if any(w in p["question"].lower() for w in ["what is","define","explain","mean"]))
print(f"  Capital questions:    {cap_count:,}")
print(f"  Who questions:        {who_count:,}")
print(f"  When questions:       {when_count:,}")
print(f"  Definition questions: {def_count:,}")
print(f"  Other:                {len(all_pairs)-cap_count-who_count-when_count:,}")
print()
print("NEXT STEPS:")
print("1. Upload new_handcrafted_qa.json to Kaggle as a dataset")
print("2. Upload upgraded_training_notebook.ipynb to Kaggle")
print("3. Run all cells — takes ~4-5 hours on P100")
print("4. Download nanoqa_v2.zip")
print("5. Replace ~/slm_project/models/nanoqa/ with new files")
print("6. Replace ~/slm_project/app.py with new_app.py")
