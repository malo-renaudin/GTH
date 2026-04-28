import random

# Vocabulary pools
nouns = [
    "boy", "girl", "dog", "cat", "teacher", "student", "doctor", "artist",
    "child", "friend", "man", "woman", "musician", "coach", "player",
    "tourist", "engineer", "scientist", "author", "painter", "chef",
    "driver", "athlete", "actor", "singer", "neighbor", "professor",
    "pilot", "dancer", "farmer", "soldier", "officer", "nurse", "child",
    "grandparent", "neighbor", "politician", "waiter", "gardener"
]

verbs = [
    "see", "like", "chase", "admire", "help", "follow", "find", "watch",
    "know", "meet", "greet", "teach", "support", "approach", "recognize",
    "catch", "visit", "praise", "avoid", "respect", "encourage", "invite",
    "push", "hold", "protect", "observe", "consult", "inspire"
]

wh_words_object = ["what", "whom", "which", "whose"]

# Irregular plurals
irregular_plurals = {
    "woman": "women",
    "man": "men",
    "child": "children",
    "person": "people",
    "mouse": "mice",
    "goose": "geese",
    "tooth": "teeth",
    "foot": "feet"
}

# Irregular past verbs
irregular_past = {
    "see": "saw",
    "meet": "met",
    "find": "found",
    "know": "knew",
    "like": "liked",
    "chase": "chased",
    "admire": "admired",
    "help": "helped",
    "follow": "followed",
    "watch": "watched",
    "teach": "taught",
    "greet": "greeted",
    "support": "supported",
    "approach": "approached",
    "recognize": "recognized",
    "catch": "caught",
    "visit": "visited",
    "praise": "praised",
    "avoid": "avoided",
    "respect": "respected",
    "encourage": "encouraged",
    "invite": "invited",
    "push": "pushed",
    "hold": "held",
    "protect": "protected",
    "observe": "observed",
    "consult": "consulted",
    "inspire": "inspired"
}

# Pluralize nouns
def pluralize(noun):
    if noun in irregular_plurals:
        return irregular_plurals[noun]
    elif noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    elif noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    else:
        return noun + "s"

# Pick tense based on subject
def pick_tense(subject, requested_tense):
    if requested_tense == "base" and not subject.endswith("s"):
        return "3sg"
    return requested_tense

# Conjugate verb based on subject number and tense
def conjugate(verb, subject, tense="base"):
    is_plural = subject.endswith("s")
    if tense == "past":
        return irregular_past.get(verb, verb + "ed")
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
    else:
        return verb

# Generate one object wh-question
def generate_wh_question():
    subj = random.choice(nouns)
    obj = random.choice([n for n in nouns if n != subj])
    # Randomly pluralize subjects and objects
    if random.random() > 0.5:
        subj = pluralize(subj)
    if random.random() > 0.5:
        obj = pluralize(obj)
    
    verb = random.choice(verbs)

    # Randomly pick tense for the main verb
    tense_options = ["base", "past"]
    main_tense = random.choice(tense_options)
    
    # Pick WH-word
    wh_word = random.choice(wh_words_object)
    
    # Determine auxiliary for question formation
    aux = ""
    if main_tense == "past":
        aux = "did"
    elif main_tense == "base":
        aux = "does" if not subj.endswith("s") else "do"
    
    # Verb form in base (with subject-verb agreement handled by auxiliary)
    verb_form = conjugate(verb, subj, "base")
    
    # Avoid nonsensical lemma repetition
    if subj.rstrip("s") == obj.rstrip("s") or verb_form == subj.rstrip("s") or verb_form == obj.rstrip("s"):
        return None
    
    sentence = f"{wh_word.capitalize()} {aux} the {subj} {verb_form}?"
    return sentence

# Generate dataset
sentences = set()
while len(sentences) < 1000000:
    s = generate_wh_question()
    if s:
        sentences.add(s)

# Save to file
with open("WHQ.txt", "w") as f:
    for s in sentences:
        f.write(s + "\n")

print("✅ Generated 1000000 object WH-questions in WH_object_questions.txt")
