import random
import itertools
import argparse

parser = argparse.ArgumentParser(description="Generate minimal pairs of SRC and ORC sentences.")
parser.add_argument("--output_src", type=str, default="src_pairs.txt", help="Path to save the SRC sentences.")
parser.add_argument("--output_orc", type=str, default="orc_pairs.txt", help="Path to save the ORC sentences.")
parser.add_argument("--n_pairs", type=int, default=1000, help="Number of minimal pairs to sample.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

args = parser.parse_args()
random.seed(args.seed)

ADVERB_RESTRICTIONS = {
    "follows": {"secretly"},
    "greets": {"secretly"},
}

def is_valid_verb_adverb_combo(verb, adverb):
    if verb in ADVERB_RESTRICTIONS:
        return adverb not in ADVERB_RESTRICTIONS[verb]
    return True

def pluralize_noun(noun):
    irregulars = {"child": "children", "man": "men", "woman": "women"}
    if noun in irregulars:
        return irregulars[noun]
    return noun + "s"

def build_sentence(structure, n1, n2, v_sing, v_plur, adj1, adj2, adv1, rel, n1_is_plural, n2_is_plural, tail):
    current_n1 = pluralize_noun(n1) if n1_is_plural else n1
    current_n2 = pluralize_noun(n2) if n2_is_plural else n2

    if structure == "SRC":
        # n1 is subject of the embedded verb → verb agrees with n1
        v = v_plur if n1_is_plural else v_sing
        # "The adj1 n1 rel adv1 V the adj2 n2 tail."
        parts = ["The", adj1, current_n1, rel, adv1, v, "the", adj2, current_n2, tail]
    else:  # ORC
        # n2 is subject of the embedded verb → verb agrees with n2
        v = v_plur if n2_is_plural else v_sing
        # "The adj1 n1 rel the adj2 n2 adv1 V tail."
        parts = ["The", adj1, current_n1, rel, "the", adj2, current_n2, adv1, v, tail]

    sentence = " ".join(p for p in parts if p) + "."
    return sentence.capitalize()

def generate_pairs(n1_opts, n2_opts, v_opts, adj1_opts, adj2_opts, adv1_opts):
    cont_sing = ["is eating an apple", "is watching a movie", "is reading a book", "likes to dance", "enjoys music", "likes climbing"]
    cont_plur = ["are eating an apple", "are watching a movie", "are reading a book", "like to dance", "enjoy music", "like climbing"]

    # rel options: "that" and "who" only (SRC requires an overt relativizer)
    rel_opts = ["that", "who"]

    pairs = []
    seen = set()

    word_combinations = itertools.product(
        n1_opts, n2_opts, v_opts, adj1_opts, adj2_opts, adv1_opts,
        rel_opts, [False, True], [False, True]
    )

    for n1, n2, (v_sing, v_plur), adj1, adj2, adv1, rel, n1_p, n2_p in word_combinations:
        if not is_valid_verb_adverb_combo(v_sing, adv1):
            continue

        tail_pool = cont_plur if n1_p else cont_sing
        for tail in tail_pool:
            src = build_sentence("SRC", n1, n2, v_sing, v_plur, adj1, adj2, adv1, rel, n1_p, n2_p, tail)
            orc = build_sentence("ORC", n1, n2, v_sing, v_plur, adj1, adj2, adv1, rel, n1_p, n2_p, tail)
            key = (src, orc)
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    return pairs

n1_opts = ["boy", "student", "doctor", "artist", "athlete"]
n2_opts = ["girl", "child", "pilot", "scientist", "engineer"]
v_opts = [("visits", "visit"), ("helps", "help"), ("avoids", "avoid"), ("follows", "follow"), ("greets", "greet")]
adj1_opts = ["big", "tall", "young", "strong", "kind"]
adj2_opts = ["beautiful", "smart", "brave", "famous", "honest"]
adv1_opts = ["possibly", "apparently", "secretly", "always", "often", "rarely"]

all_pairs = generate_pairs(n1_opts, n2_opts, v_opts, adj1_opts, adj2_opts, adv1_opts)
print(f"Total unique minimal pairs generated: {len(all_pairs)}")

if len(all_pairs) < args.n_pairs:
    print(f"Warning: only {len(all_pairs)} pairs available, fewer than requested {args.n_pairs}.")
    sampled = all_pairs
else:
    sampled = random.sample(all_pairs, args.n_pairs)

with open(args.output_src, "w") as f_src, open(args.output_orc, "w") as f_orc:
    for src, orc in sampled:
        f_src.write(src + "\n")
        f_orc.write(orc + "\n")

print(f"Wrote {len(sampled)} pairs to '{args.output_src}' and '{args.output_orc}'.")