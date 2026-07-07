import itertools
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, default="generate_simple_datasets/generated_simple_datasets/wh_movement_factorial.csv")
args = parser.parse_args()

nouns        = ["boy", "student", "doctor", "artist", "nurse",
                "girl", "pilot", "guard", "judge", "teacher"]
verbs_ing    = ["writing", "reading", "singing", "drawing", "painting",
                "cooking", "playing", "making", "cleaning", "watching"]  # added 5 more verbs
wh_words     = ["What"]                                                  # +filler
no_fillers   = ["This", "That", "It", "He", "She"]                      # expanded: +He, +She

# Only semantically compatible verb-object pairs — single GPT2 tokens only
VERB_OBJECTS = {
    "writing":  ["a song", "a book", "a poem", "a letter", "a story"],
    "reading":  ["a book", "a poem", "a letter", "a story", "a note"],   # added: note
    "singing":  ["a song", "a melody", "a chant", "a tune", "a hymn"],  # added: tune, hymn
    "drawing":  ["a portrait", "a map", "a sketch", "a face", "a chart"],# added: face, chart
    "painting": ["a portrait", "a mural", "a scene", "a wall", "a room"],# added: wall, room
    "cooking":  ["a meal", "a soup", "a cake", "a stew", "a sauce"],
    "playing":  ["a song", "a tune", "a game", "a role", "a trick"],
    "making":   ["a film", "a cake", "a speech", "a deal", "a list"],
    "cleaning": ["a room", "a car", "a desk", "a wall", "a floor"],
    "watching": ["a film", "a game", "a show", "a bird", "a match"],
}

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

with open(args.output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "quadruplet_id", "filler", "gap", "pre_gap_text", "post_gap_text", "filler_word"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Total quadruplets: {quad_id}")
print(f"Total rows: {len(rows)}")
