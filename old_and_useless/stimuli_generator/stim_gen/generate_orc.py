import random

# --------------------------
# Vocabulary pools
# --------------------------
nouns = [
    "boy", "girl", "dog", "cat", "teacher", "student",
    "doctor", "artist", "child", "friend", "man", "woman",
    "engineer", "lawyer", "musician", "neighbor", "parent", "sibling",
    "bird", "rabbit", "childhood friend", "coach", "player", "tourist",
    "driver", "passenger", "cook", "chef", "writer", "painter"
]

verbs = [
    "see", "like", "chase", "admire", "help", "follow",
    "find", "watch", "know", "meet", "greet", "call", "invite",
    "thank", "support", "guide", "teach", "protect", "observe"
]

complements = [
    "a dog", "a cat", "a book", "the ball",
    "an apple", "the teacher", "a friend", "the child",
    "the neighbor", "a player", "a musician", "the driver",
    "a chef", "the bird", "the rabbit"
]

animate_nouns = {
    "boy", "girl", "teacher", "student", "doctor", "artist", "child",
    "friend", "man", "woman", "engineer", "lawyer", "musician",
    "neighbor", "parent", "sibling", "coach", "player", "tourist",
    "driver", "cook", "chef", "writer", "painter"
}

animate_verbs = {"admire", "help", "follow", "meet", "greet", "call", "invite", "thank", "support", "guide", "teach", "protect", "observe"}
neutral_verbs = {"see", "like", "chase", "find", "watch", "know"}
# --------------------------
# Pluralization
# --------------------------
def pluralize(noun):
    irregulars = {
        "woman": "women", "man": "men", "child": "children",
        "person": "people", "mouse": "mice", "goose": "geese",
        "tooth": "teeth", "foot": "feet"
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
    is_plural = subject.endswith("s")
    if tense == "past":
        irregulars = {
            "see": "saw", "meet": "met", "find": "found", "know": "knew",
            "like": "liked", "chase": "chased", "admire": "admired",
            "help": "helped", "follow": "followed", "watch": "watched"
        }
        return irregulars.get(verb, verb + "ed")
    elif tense == "3sg":
        if is_plural:
            return verb
        else:
            if verb.endswith("y") and verb[-2] not in "aeiou":
                return verb[:-1] + "ies"
            elif verb.endswith(("s", "x", "z", "ch", "sh")):
                return verb + "es"
            else:
                return verb + "s"
    else:
        return verb

# --------------------------
# Pick tense with subject agreement
# --------------------------
def pick_tense(subject, requested_tense):
    if requested_tense == "base" and not subject.endswith("s"):
        return "3sg"
    return requested_tense

# --------------------------
# Sentence completion
# --------------------------
def complete_sentence(main_verb_form, used_nouns):
    # Pick a complement not repeating previous nouns
    possible_complements = [c for c in complements if not any(n in c for n in used_nouns)]
    complement = random.choice(possible_complements) if possible_complements else random.choice(complements)
    return f"{main_verb_form} {complement}"

# --------------------------
# Generate one sentence
# --------------------------
def generate_sentence():
    for _ in range(20):  # try multiple times to satisfy constraints
        n1 = random.choice(nouns)
        n2 = random.choice([n for n in nouns if n != n1])

        if random.random() > 0.5:
            n1 = pluralize(n1)
        if random.random() > 0.5 and not n1.endswith("s"):
            n2 = pluralize(n2)

        v1 = random.choice(verbs)
        v2 = random.choice([v for v in verbs if v != v1])

        # Avoid lemma repetition
        used_lemmas = {n1.rstrip("s"), n2.rstrip("s"), v1, v2}
        if len(used_lemmas) < 4:
            continue

        # Animate constraints
        if v1 in animate_verbs and n2.rstrip("s") not in animate_nouns:
            continue
        if v2 in animate_verbs and n1.rstrip("s") not in animate_nouns:
            continue

        # Random tense
        tense_options = ["base", "past", "3sg"]
        v1_tense = pick_tense(n2, random.choice(tense_options))
        v2_tense = pick_tense(n1, random.choice(tense_options))

        # Conjugate respecting subject number
        v1_form = conjugate(v1, n2, v1_tense)
        v2_form = conjugate(v2, n1, v2_tense)

        # Complete sentence
        v2_completed = complete_sentence(v2_form, {n1, n2})

        return f"The {n1} that the {n2} {v1_form} {v2_completed}."
    return None

# --------------------------
# Generate dataset
# --------------------------
def generate_dataset(num_sentences=1000000, output_file="ORC.txt"):
    sentences = set()
    while len(sentences) < num_sentences:
        s = generate_sentence()
        if s:
            sentences.add(s)
    with open(output_file, "w") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"✅ Generated {len(sentences)} sentences in {output_file}")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    generate_dataset(num_sentences=1000000)
