import random
import itertools
import argparse

parser = argparse.ArgumentParser(description="Generate a large corpus of sentences based on specific structures.")
parser.add_argument("--output_file", type=str, default="datasets/orc.txt", help="Path to save the generated corpus.")
parser.add_argument("--structures", type=int, default=1, help="Structures to generate: 1 for ORC, 2 for SVO.")

args = parser.parse_args()
struct = args.structures
output_file = args.output_file

print(f"Generating sentences for structures: {struct}")

# Verb-adverb compatibility rules
# Some verbs don't work well with certain adverbs
ADVERB_RESTRICTIONS = {
    "follows": {"secretly"},  # Only frequency adverbs work well
    "greets": {"secretly"},# Can't greet secretly

}

PAST_PARTICIPLES = {
    "visits":  "visited",
    "helps":   "helped",
    "avoids":  "avoided",
    "follows": "followed",
    "greets":  "greeted",
}

# Auxiliaries for the active ORC variant (modal / do-support; verb takes base form)
AUX_LIST = ["", "did ", "could ", "couldn't ", "didn't ", "would ", "wouldn't ", "should ", "shouldn't ", "might "]

def is_valid_verb_adverb_combo(verb, adverb):
    """Check if verb + adverb combination is natural."""
    if verb in ADVERB_RESTRICTIONS:
        return adverb not in ADVERB_RESTRICTIONS[verb]
    return True

def generate_sentence_variants(structure, n1, n2, v1_sing, v1_plur, adj1, adj2, adv1, n1_is_plural=False, n2_is_plural=False):
    current_n1 = pluralize_noun(n1) if n1_is_plural else n1 
    current_n2 = pluralize_noun(n2) if n2_is_plural else n2
    current_v1 = v1_plur if n2_is_plural else v1_sing
    
    modifiers = {
        'adj1': [adj1, ""],
        'adj2': [adj2, ""],
        'adv1': [adv1, ""],
        'rel': ["that", "who", ""]
    }
    if structure == 1:
        modifiers['aux1'] = AUX_LIST

    keys = modifiers.keys()
    combinations = list(itertools.product(*modifiers.values()))
    
    # We return a list of tuples: (core_string, n1_is_plural)
    cores = []

    for combo in combinations:
        d = dict(zip(keys, combo))
        
        # Skip invalid verb-adverb combinations
        if not is_valid_verb_adverb_combo(v1_sing, d['adv1']):
            continue
            
        if structure == 1:
            aux1 = d.get('aux1', '')
            v1_form = v1_plur if aux1 else current_v1  # base form after aux, agreement-based otherwise
            s_core = f"The {d['adj1']} {current_n1} {d['rel']} the {d['adj2']} {current_n2} {aux1}{d['adv1']} {v1_form}"
        elif structure == 2:
            s_core = f"The {d['adj2']} {current_n2} {d['adv1']} {current_v1} the {d['adj1']} {current_n1}"
        
        clean_core = " ".join(s_core.split()).strip()
        cores.append((clean_core, n1_is_plural, structure))

    # Grammatical passive ORC: "The N1 that was/were [adv] V-pp by the N2"
    # was/were agrees with n1 (the passive subject); n2 is the "by" agent
    if structure == 1:
        v1_pp = PAST_PARTICIPLES.get(v1_sing)
        if v1_pp:
            aux_pass = "were " if n1_is_plural else "was "
            pass_modifiers = {'adj1': [adj1, ""], 'adj2': [adj2, ""], 'adv1': [adv1, ""], 'rel': ["that", "who", ""]}
            for combo in itertools.product(*pass_modifiers.values()):
                d = dict(zip(pass_modifiers.keys(), combo))
                if not is_valid_verb_adverb_combo(v1_sing, d['adv1']):
                    continue
                s_core = f"The {d['adj1']} {current_n1} {d['rel']} {aux_pass}{d['adv1']} {v1_pp} by the {d['adj2']} {current_n2}"
                clean_core = " ".join(s_core.split()).strip()
                cores.append((clean_core, n1_is_plural, structure))

    return cores

def generate_full_corpus(struct, n1_opts, n2_opts, v_opts, adj1_opts, adj2_opts, adv1_opts):
    # This set will store (core_string, n1_is_plural, structure)
    # Because it's a set of tuples, it will catch duplicates in the core!
    unique_cores = set()
    
    word_combinations = itertools.product(n1_opts, n2_opts, v_opts, adj1_opts, adj2_opts, adv1_opts)
    
    for n1, n2, v_pair, adj1, adj2, adv1 in word_combinations:
        v_sing, v_plur = v_pair
        
        # We only care about Structure 1 (ORC) based on your loop
        # for struct in [1]:
        for n1_p, n2_p in itertools.product([False, True], [False, True]):
            scenarios = generate_sentence_variants(
                struct, n1, n2, v_sing, v_plur, adj1, adj2, adv1, n1_p, n2_p
            )
            unique_cores.update(scenarios)

    # NOW we add the random continuations to the unique cores
    final_sentences = []
    
    cont_sing = ["is eating an apple", "is watching a movie", "is reading a book", "likes to dance", "enjoys music", "likes climbing"]
    cont_plur = ["are eating an apple", "are watching a movie", "are reading a book", "like to dance", "enjoy music", "like climbing"]

    for core, is_plural, struct in unique_cores:
        if struct == 1:
            tail = random.choice(cont_plur) if is_plural else random.choice(cont_sing)
            full_sent = f"{core} {tail}."
        else:
            full_sent = f"{core}."
            
        final_sentences.append(full_sent.capitalize())

    return final_sentences

# 1. Define the 3-word vocabulary
n1_opts = ["boy", "student", "doctor", "artist", "athlete"]
n2_opts = ["girl", "child", "pilot", "scientist", "engineer"] # 'child' requires an irregular plural check
v_opts = [("visits", "visit"), ("helps", "help"), ("avoids", "avoid"), ("follows", "follow"), ("greets", "greet")]
adj1_opts = ["big", "tall", "young", "strong", "kind"]
adj2_opts = ["beautiful", "smart", "brave", "famous", "honest"]
adv1_opts = ["possibly", "apparently", "secretly", "always", "often", "rarely"]

# 2. Update Noun Pluralization for irregulars
def pluralize_noun(noun):
    irregulars = {"child": "children", "man": "men", "woman": "women"}
    if noun in irregulars:
        return irregulars[noun]
    return noun + "s"

# 3. Generate the massive corpus
# Note: This might take a few seconds to run due to the 93k+ iterations
final_corpus = generate_full_corpus(
    struct, n1_opts, n2_opts, v_opts, 
    adj1_opts, adj2_opts, adv1_opts
)
final_corpus = list(dict.fromkeys(final_corpus))  # remove duplicates, preserve order
with open(output_file, "w") as f:
    for sentence in final_corpus:
        f.write(sentence + "\n")
print(f"Total Sentences in Corpus: {len(final_corpus)}")