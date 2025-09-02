import random

# --------------------------
# Vocabulary pools
# --------------------------
nouns = [
    "boy", "girl", "dog", "cat", "teacher", "student",
    "doctor", "artist", "child", "friend", "man", "woman", "person"
]

verbs = [
    "see", "like", "chase", "admire", "help", "follow", "find", "watch", "know", "meet"
]

# --------------------------
# Pluralization function
# --------------------------
def pluralize(noun):
    irregulars = {
        "woman": "women",
        "man": "men",
        "child": "children",
        "person": "people",
        "mouse": "mice",
        "goose": "geese",
        "tooth": "teeth",
        "foot": "feet"
    }
    if noun in irregulars:
        return irregulars[noun]
    elif noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    elif noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    else:
        return noun + "s"

# --------------------------
# Verb conjugation
# --------------------------
def conjugate(verb, subject, tense="base"):
    is_plural = subject.endswith("s") or subject in ["children", "men", "women", "people"]
    # Past tense
    if tense == "past":
        irregulars = {"see": "saw", "meet": "met", "find": "found", "know": "knew",
                      "like": "liked", "chase": "chased", "admire": "admired",
                      "help": "helped", "follow": "followed", "watch": "watched"}
        return irregulars.get(verb, verb + "ed")
    # 3rd person singular
    elif tense == "3sg":
        if is_plural:
            return verb  # plural subject uses base
        else:
            if verb.endswith("y") and verb[-2] not in "aeiou":
                return verb[:-1] + "ies"
            elif verb.endswith(("s", "x", "z", "ch", "sh")):
                return verb + "es"
            else:
                return verb + "s"
    # base form
    else:
        return verb

# --------------------------
# Ensure correct tense for subject
# --------------------------
def pick_tense(subject, requested_tense):
    if requested_tense == "base" and not subject.endswith("s"):
        # singular subject: base present -> 3rd person singular
        return "3sg"
    return requested_tense

# --------------------------
# Sentence generation
# --------------------------
def generate_sentence():
    # Pick two distinct nouns
    n1 = random.choice(nouns)
    n2 = random.choice([n for n in nouns if n != n1])

    # Random pluralization
    if random.random() > 0.5:
        n1 = pluralize(n1)
    if random.random() > 0.5:
        n2 = pluralize(n2)

    # Pick two distinct verbs
    v1 = random.choice(verbs)
    v2 = random.choice([v for v in verbs if v != v1])

    # Avoid lemma repetition
    used_lemmas = {n1.rstrip("s"), n2.rstrip("s"), v1, v2}
    if len(used_lemmas) < 4:
        return None  # reject

    # Random tense selection
    tense_options = ["base", "past", "3sg"]
    v1_tense = random.choice(tense_options)  # relative clause verb
    v2_tense = random.choice(tense_options)  # main clause verb

    # Determine actual tenses
    v1_actual = pick_tense(n2, v1_tense)
    v2_actual = pick_tense(n1, v2_tense)

    # Conjugate verbs
    v1_form = conjugate(v1, n2, v1_actual)
    v2_form = conjugate(v2, n1, v2_actual)

    # Debug statements
    print(f"Chosen nouns: n1={n1}, n2={n2}")
    print(f"Chosen verbs: v1={v1}, v2={v2}")
    print(f"Tenses: v1_tense={v1_tense}, v2_tense={v2_tense}")
    print(f"Actual tenses: v1_actual={v1_actual}, v2_actual={v2_actual}")
    print(f"Conjugate: v1_form={v1_form}, v2_form={v2_form}")
    print(f"The {n1} that the {n2} {v1_form} {v2_form}.")

    return f"The {n1} that the {n2} {v1_form} {v2_form}."

# --------------------------
# Generate dataset
# --------------------------
sentences = set()
target_size = 50  # number of sentences
while len(sentences) < target_size:
    s = generate_sentence()
    if s:
        sentences.add(s)
    if len(sentences) % 10000 == 0:
        print(f"Generated {len(sentences)} sentences...")

# --------------------------
# Save to file
# --------------------------
with open("ORC.txt", "w") as f:
    for s in sentences:
        f.write(s + "\n")

print(f"✅ Generated {target_size} sentences in ORC.txt")
