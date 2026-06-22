import itertools
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, default="generate_simple_datasets/generated_simple_datasets/filler_gap_factorial.csv")
args = parser.parse_args()

nouns1        = ["boy", "student", "doctor", "artist", "athlete"]
nouns2        = ["girl", "child", "pilot", "scientist", "engineer"]
continuations = ["is happy", "is tired", "is sleeping", "is smart", "is tall"]
matrix_verbs  = ["He knows", "She says", "They think", "We believe"]

# Only semantically compatible verb-object pairs
VERB_OBJECTS = {
    "likes":   ["football", "music", "painting", "cooking", "reading"],
    "visits":  ["museums", "galleries", "parks", "libraries", "zoos"],
    "helps":   ["friends", "neighbors", "colleagues", "strangers", "classmates"],
    "avoids":  ["football", "crowds", "noise", "trouble", "conflict"],
    "follows": ["rules", "instructions", "advice", "trends", "directions"],
}

rows = []
quad_id = 0
seen = set()

for n1, n2, cont, mv in itertools.product(nouns1, nouns2, continuations, matrix_verbs):
    for v, objects in VERB_OBJECTS.items():
        for obj in objects:
            key = (n1, n2, v, obj, cont, mv)
            if key in seen:
                continue
            seen.add(key)

            pre_gap_filler   = f"The {n1} that the {n2} {v}"
            pre_gap_nofiller = f"{mv} that the {n2} {v}"

            conditions = [
                (f"{pre_gap_filler} {cont}.",    1, 1, cont, pre_gap_filler),    # +filler +gap
                (f"{pre_gap_nofiller} {cont}.",  0, 1, cont, pre_gap_nofiller),  # -filler +gap
                (f"{pre_gap_filler} {obj}.",     1, 0, obj,  pre_gap_filler),    # +filler -gap
                (f"{pre_gap_nofiller} {obj}.",   0, 0, obj,  pre_gap_nofiller),  # -filler -gap
            ]
            for sentence, filler, gap, post_gap_text, pre_gap_text in conditions:
                rows.append({
                    "sentence":       sentence,
                    "quadruplet_id":  quad_id,
                    "filler":         filler,
                    "gap":            gap,
                    "pre_gap_text":   pre_gap_text,
                    "post_gap_text":  post_gap_text,
                    "matrix_verb":    mv if filler == 0 else "",
                })
            quad_id += 1

with open(args.output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "quadruplet_id", "filler", "gap", "pre_gap_text", "post_gap_text", "matrix_verb"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Total quadruplets: {quad_id}")
print(f"Total rows: {len(rows)}")
