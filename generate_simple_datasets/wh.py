import itertools
import argparse

parser = argparse.ArgumentParser(description="Generate a large corpus of sentences based on specific structures.")
parser.add_argument("--output_file", type=str, default="datasets/wh.txt", help="Path to save the generated corpus.")
parser.add_argument("--structures", type=int, default=1, help="Structures to generate: 1 for WH-Questions, 2 for Declaratives.")

args = parser.parse_args()
struct = args.structures
output_file = args.output_file

# Nouns (Human/Social Roles)
n1_opts = ["student", "doctor", "pilot", "officer", "athlete",
           "artist"]#, "scientist"]# "teacher", "lawyer", "judge"]

n2_opts = ["child", "girl", "boy", "patient", "client", 
           "tourist"]#, "driver"]# "neighbor", "guest", "worker"]

# Verbs (Action/Social - Transitive)
# Format: (Base Form, Progressive -ing Form)
# v_opts = [
#     ("visit", "visiting"),
#     ("help", "helping"),
#     ("greet", "greeting"),
#     ("follow", "following"),
#     ("avoid", "avoiding"),
#     ("call", "calling"),
#     ("observe", "observing"),
#     # ("assist", "assisting"),
#     # ("meet", "meeting"),
#     # ("see", "seeing")
# ]
v_opts = [
    ("visit", "visited", "visiting"),
    ("help", "helped", "helping"),
    ("greet", "greeted", "greeting"),
    ("follow", "followed", "following"),
    ("avoid", "avoided", "avoiding"),
    ("call", "called", "calling"),
    ("observe", "observed", "observing"),
    # ("assist", "assisted", "assisting"),
    # ("meet", "met", "meeting"),
    # ("see", "saw", "seeing")
]

# Adjectives (Personality/Status)
adj1_opts = ["young", "tall", "smart", "brave", "kind", 
             "famous"]#, "honest"]#"strong", "busy", "calm"]

adj2_opts = ["creative", "serious", "friendly", "quiet", "active"]#, 
            #  "careful"]#, "proud", "gentle", "fair", "eager"]

# Adverbs (Frequency - Placement is before the verb)
# adv1_opts = ["often", "always", "rarely", "secretly", "usually"]#, 
            # "quickly"]#, "quietly", "clearly", "daily", "never"]

adv1_opts = ["probably", "certainly", "possibly", "apparently", "maybe", "perhaps"]

# Verbs that require animate objects (no "What" questions)
ANIMATE_ONLY_VERBS = {"greet", "call", "help"}

def get_wh_words_for_verb(v_base):
    """Return list of wh-words appropriate for the given verb."""
    if v_base in ANIMATE_ONLY_VERBS:
        return ["Who", "Whom"]
    return ["What", "Who", "Whom"]

def generate_wh_variants(n1, n2, v_pair, adj1, adj2, adv1, n1_plur, n2_plur):
    v_base, _, v_ing = v_pair
    curr_n1 = pluralize_noun(n1) if n1_plur else n1
    curr_n2 = pluralize_noun(n2) if n2_plur else n2
    
    # Modifiers
    mods = {'a1': [adj1, ""], 'a2': [adj2, ""], 'ad1': [adv1, ""]}
    combos = list(itertools.product(*mods.values()))
    
    results = []
    
    # Determine which wh-words are appropriate for this verb
    wh_words = get_wh_words_for_verb(v_base)

    # Case A: N1 is the Subject (What/Who/Whom or Who/Whom)
    # Aux agrees with N1
    aux_list_n1 = [
        ("did", v_base), 
        ("does" if not n1_plur else "do", v_base),
        ("is" if not n1_plur else "are", v_ing)
    ]

    for aux, v_fin in aux_list_n1:
        for a1, a2, ad1 in combos:
            # Template 1: What/Who/Whom variants (filtered by verb requirements)
            for wh_word in wh_words:
                q_wh_n1 = f"{wh_word} {aux} the {a1} {curr_n1} {ad1} {v_fin}?"
                results.append(" ".join(q_wh_n1.split()).capitalize())
            
            # Template 2: Which (N1 is Subject, N2 is Object)
            q_which = f"Which {a2} {curr_n2} {aux} the {a1} {curr_n1} {ad1} {v_fin}?"
            results.append(" ".join(q_which.split()).capitalize())

    # Case B: N2 is the Subject - Skip for animate-only verbs (they need an object)
    # Only generate for verbs that can appear with N2 as subject
    if v_base not in ANIMATE_ONLY_VERBS:
        # Aux agrees with N2
        aux_list_n2 = [
            ("did", v_base), 
            ("does" if not n2_plur else "do", v_base),
            ("is" if not n2_plur else "are", v_ing)
        ]

        for aux, v_fin in aux_list_n2:
            for a1, a2, ad1 in combos:
                # Template 1: What/Who/Whom variants
                for wh_word in wh_words:
                    q_wh_n2 = f"{wh_word} {aux} the {a2} {curr_n2} {ad1} {v_fin}?"
                    results.append(" ".join(q_wh_n2.split()).capitalize())
            
    return results

def generate_declarative_variants(n1, n2, v_tuple, adj1, adj2, adv1, n1_plur, n2_plur):
    # v_tuple example: ("visit", "visited", "visiting")
    v_base, v_past, v_ing = v_tuple
    
    curr_n1 = pluralize_noun(n1) if n1_plur else n1
    curr_n2 = pluralize_noun(n2) if n2_plur else n2
    
    # 1. Handle Verb Agreement for the 3 Tenses
    # Present Tense Agreement
    v_pres = v_base if n1_plur else (v_base + "s" if not v_base.endswith('s') else v_base + "es")
    # Progressive Agreement
    aux_prog = "are" if n1_plur else "is"
    
    tenses = [
        ("past", v_past),      # Past: "The doctor visited..."
        ("present", v_pres),   # Present: "The doctor visits..."
        ("prog", v_ing)        # Progressive: "The doctor is visiting..."
    ]
    
    # 2. Modifier Combinations (2^3 = 8)
    mods = {'a1': [adj1, ""], 'a2': [adj2, ""], 'ad1': [adv1, ""]}
    combos = list(itertools.product(*mods.values()))
    
    declaratives = []
    for tense_type, verb_form in tenses:
        for a1, a2, ad1 in combos:
            if tense_type == "prog":
                # Place adverb between auxiliary and participle: "is probably visiting"
                verb_phrase = f"{aux_prog} {ad1} {verb_form}" if ad1 else f"{aux_prog} {verb_form}"
            else:
                # For simple past/present keep adverb before the lexical verb.
                verb_phrase = f"{ad1} {verb_form}" if ad1 else verb_form

            # Structure: The [adj1] N1 [Verb Phrase] the [adj2] N2.
            s = f"The {a1} {curr_n1} {verb_phrase} the {a2} {curr_n2}."
            declaratives.append(" ".join(s.split()).capitalize())
            
    return declaratives

def pluralize_noun(noun):
    irregulars = {"child": "children", "man": "men", "woman": "women"}
    return irregulars.get(noun, noun + "s")

def generate_final_corpus_to_file(struct, filename):
    word_products = itertools.product(n1_opts, n2_opts, v_opts, adj1_opts, adj2_opts, adv1_opts)
    
    seen = set()
    with open(filename, "w") as f:
        for n1, n2, v_pair, adj1, adj2, adv1 in word_products:
            # Generate all 4 number scenarios
            for n1_p, n2_p in itertools.product([False, True], [False, True]):
                if struct == 1:
                    # Generate all WH variants for this combination
                    wh_variants = generate_wh_variants(n1, n2, v_pair, adj1, adj2, adv1, n1_p, n2_p)
                    for sentence in wh_variants:
                        if sentence not in seen:
                            seen.add(sentence)
                            f.write(sentence + "\n")    
                elif struct == 2:
                    # Fetch all grammatical variants (What/Which/Tenses/Modifiers)
                    batch = generate_declarative_variants(n1, n2, v_pair, adj1, adj2, adv1, n1_p, n2_p)
                    for sentence in batch:
                        if sentence not in seen:
                            seen.add(sentence)
                            f.write(sentence + "\n")

if __name__== "__main__":
    generate_final_corpus_to_file(struct, output_file)