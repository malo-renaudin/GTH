from python_mg import Lexicon
import random

random.seed(500)

with open("reference.grammar") as f:
    lexicon = f.read()

lexicon = Lexicon(lexicon)

strings: list[str] = []
for p in lexicon.generate_grammar(
    "C", max_strings=20000000, n_beams=None, min_log_prob=None, max_steps=48
):
    strings.append(str(p))

for s in random.choices(strings, k=20):
    print(s)
    print("CONTINUATIONS")
    s = s.split(" ")
    for i in range(1, len(s) + 1):
        prefix = " ".join(s[:i])
        print(
            prefix,
            lexicon.continuations(
                prefix, "C", n_beams=None, min_log_prob=None, max_steps=48
            ),
        )
