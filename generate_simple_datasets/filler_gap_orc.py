import itertools
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--vocab", type=str, default="iv", choices=["iv", "oov"],
                    help="'iv' = in-vocabulary (matches ORC training set); 'oov' = out-of-vocabulary.")
parser.add_argument("--subsample", type=int, default=None,
                    help="If set, randomly keep this many quadruplets (e.g. 1250).")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.output_file is None:
    args.output_file = f"generate_simple_datasets/generated_simple_datasets/filler_gap_factorial_{args.vocab}.csv"

VOCAB = {
    # Words used in orc.py training
    "iv": {
        "nouns1":        ["boy", "student", "doctor", "artist", "athlete"],
        "nouns2":        ["girl", "child", "professor", "scientist", "neighbor"],
        "continuations": ["is happy", "is tired", "is sleeping", "is smart", "is tall"],
        "no_fillers":    ["near", "beside", "behind", "with"],
        "verb_objects": {
            "sees":     ["crowds", "friends", "peers", "staff", "teams"],
            "admires":  ["leaders", "poets", "peers", "heroes", "fans"],
            "avoids":   ["crowds", "noise", "trouble", "conflict", "stress"],
            "observes": ["trends", "birds", "rules", "crowds", "signs"],
            "greets":   ["guests", "fans", "friends", "staff", "crowds"],
        },
    },
    # Words NOT used in orc.py training — nouns can't be verbed; verb bases can't be nouns
    "oov": {
        "nouns1":        ["stranger", "passenger", "prisoner", "soldier", "customer"],
        "nouns2":        ["winner", "victim", "scholar", "tourist", "patron"],
        "continuations": ["is angry", "is anxious", "is silent", "is nervous", "is peaceful"],
        "no_fillers":    ["near", "beside", "behind", "with"],
        "verb_objects": {
            "ignores":   ["rules", "warnings", "signals", "advice", "feedback"],
            "examines":  ["files", "records", "samples", "papers", "details"],
            "inspires":  ["readers", "crowds", "pupils", "viewers", "members"],
            "protects":  ["rights", "assets", "zones", "borders", "lands"],
            "describes": ["events", "images", "patterns", "features", "plans"],
        },
    },
}

vocab         = VOCAB[args.vocab]
nouns1        = vocab["nouns1"]
nouns2        = vocab["nouns2"]
continuations = vocab["continuations"]
no_fillers    = vocab["no_fillers"]
VERB_OBJECTS  = vocab["verb_objects"]

rows = []
quad_id = 0
seen = set()

for n1, n2, cont, prep in itertools.product(nouns1, nouns2, continuations, no_fillers):
    for v, objects in VERB_OBJECTS.items():
        for obj in objects:
            key = (n1, n2, v, obj, cont, prep)
            if key in seen:
                continue
            seen.add(key)

            pre_gap_filler   = f"The {n1} that the {n2} {v}"
            pre_gap_nofiller = f"The {n1} {prep} the {n2} {v}"

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
                    "filler_word":    "that" if filler == 1 else prep,
                })
            quad_id += 1

if args.subsample is not None:
    import random
    random.seed(args.seed)
    all_qids = sorted({r["quadruplet_id"] for r in rows})
    sampled = set(random.sample(all_qids, args.subsample))
    rows = [r for r in rows if r["quadruplet_id"] in sampled]
    old_to_new = {old: new for new, old in enumerate(sorted(sampled))}
    for r in rows:
        r["quadruplet_id"] = old_to_new[r["quadruplet_id"]]
    quad_id = args.subsample

with open(args.output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "quadruplet_id", "filler", "gap", "pre_gap_text", "post_gap_text", "filler_word"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Vocab split      : {args.vocab}")
print(f"Total quadruplets: {quad_id}")
print(f"Total rows: {len(rows)}")
