import itertools
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--vocab", type=str, default="iv", choices=["iv", "oov"],
                    help="'iv' = in-vocabulary (matches WH training set); 'oov' = out-of-vocabulary.")
parser.add_argument("--subsample", type=int, default=None,
                    help="If set, randomly keep this many quadruplets (e.g. 1250).")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.output_file is None:
    args.output_file = f"generate_simple_datasets/generated_simple_datasets/wh_movement_factorial_{args.vocab}.csv"

VOCAB = {
    # Words used in wh.py training
    "iv": {
        "nouns": ["boy", "student", "doctor", "artist", "athlete",
                  "girl", "child", "professor", "scientist", "neighbor"],
        "verbs_ing": ["seeing", "admiring", "avoiding", "observing", "greeting"],
        "verb_objects": {
            "seeing":    ["crowds", "friends", "birds", "signs", "stars"],
            "admiring":  ["peers", "leaders", "heroes", "fans", "crowds"],
            "avoiding":  ["crowds", "noise", "trouble", "conflict", "stress"],
            "observing": ["birds", "trends", "rules", "crowds", "flags"],
            "greeting":  ["guests", "fans", "friends", "staff", "crowds"],
        },
    },
    # Words NOT used in wh.py training — nouns can't be verbed; verb bases/ing forms can't be nouns
    "oov": {
        "nouns": ["stranger", "passenger", "prisoner", "soldier", "customer",
                  "winner", "victim", "scholar", "tourist", "patron"],
        "verbs_ing": ["ignoring", "examining", "inspiring", "preferring", "describing"],
        "verb_objects": {
            "ignoring":   ["rules", "warnings", "signals", "advice", "feedback"],
            "examining":  ["files", "records", "samples", "papers", "details"],
            "inspiring":  ["readers", "crowds", "pupils", "viewers", "members"],
            "preferring": ["options", "items", "choices", "styles", "methods"],
            "describing": ["events", "images", "patterns", "features", "plans"],
        },
    },
}

vocab        = VOCAB[args.vocab]
nouns        = vocab["nouns"]
verbs_ing    = vocab["verbs_ing"]
VERB_OBJECTS = vocab["verb_objects"]

wh_words   = ["What"]                                                  # +filler
no_fillers = ["This", "That", "It", "He", "She"]                      # -filler

rows = []
quad_id = 0
seen = set()

for noun, v_ing, nf in itertools.product(nouns, verbs_ing, no_fillers):
    for obj in VERB_OBJECTS[v_ing]:
        key = (noun, v_ing, obj, nf)
        if key in seen:
            continue
        seen.add(key)

        wh = wh_words[0]
        pre_gap_filler   = f"{wh} is the {noun} {v_ing}"
        pre_gap_nofiller = f"{nf} is the {noun} {v_ing}"

        conditions = [
            (f"{pre_gap_filler}?",        1, 1, "?",   pre_gap_filler),    # +filler +gap
            (f"{pre_gap_nofiller}?",      0, 1, "?",   pre_gap_nofiller),  # -filler +gap
            (f"{pre_gap_filler} {obj}?",  1, 0, obj,   pre_gap_filler),    # +filler -gap
            (f"{pre_gap_nofiller} {obj}?",0, 0, obj,   pre_gap_nofiller),  # -filler -gap
        ]
        for sentence, filler, gap, post_gap_text, pre_gap_text in conditions:
            rows.append({
                "sentence":       sentence,
                "quadruplet_id":  quad_id,
                "filler":         filler,
                "gap":            gap,
                "pre_gap_text":   pre_gap_text,
                "post_gap_text":  post_gap_text,
                "filler_word":    wh if filler == 1 else nf,
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

print(f"Vocab split     : {args.vocab}")
print(f"Total quadruplets: {quad_id}")
print(f"Total rows: {len(rows)}")
