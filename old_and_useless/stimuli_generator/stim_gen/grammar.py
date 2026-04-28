from python_mg import Lexicon
import random

random.seed(100)

# proper_names = ["John", "Mary", "Paul", "Sarah", "James", "Lisa"]
proper_names = ["John", "Mary", "Paul", "Sarah", "James", "Lisa", "Michael", "Jennifer",
                "David", "Linda", "Robert", "Patricia", "William", "Elizabeth", "Richard", "Barbara",
                "Ralph", "Amber", "Mason", "Danielle", "Roy", "Rose", "Eugene", "Brittany", "Louis", "Diana", "Philip",
                "Abigail", "Bobby", "Jane"]

# agentive_nouns = ["man", "woman", "boy", "girl", "dog", "cat"]
agentive_nouns = ["man", "woman", "boy", "girl", "dog", "cat", "person", "child",
                  "baby", "adult", "teenager", "student", "teacher", "doctor", "nurse", "worker",
                  "wife", "partner", "girlfriend", "boyfriend", "athlete", "player", "coach", "referee", "spectator",
                  "fan", "tourist", "traveler", "passenger", "pedestrian", "cyclist", "runner", "swimmer", "climber", "hiker"]

objects = ["cake", "apple", "cherry"]
# det = ["the", "a", "every"]
det = ["the", "a", "every", "some", "any", "all", "each", "both", "either", "neither", "this",
       "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
       "many", "few", "several", "most", "much", "little", "enough", "more", "less", "other", "another",
       "such", "what", "which", "whose", "no"]


aux = ["can", "would", "may", "should"]

verbs = [
    "to::theme[an]= p",
    "talk::p= v",
    "see::d[an]= +acc v",
    "see::d[in]= +acc v",
    "devour::d[in]= +acc v",
    "want::d[in]= +acc v",
    "run::v",
]

questions = [
    "does::V= q= +subj3 T",
    "do::V= q= +subj2 T",
    "do::V= q= +subj1 T",
    "did::V= q= +subj3 T",
    "did::V= q= +subj2 T",
    "did::V= q= +subj1 T",
    "::q -q",
]

polar_questions = [
    "::T<= +q Q",
]
wh_words = [
    "what::d[in] -subj3 -q -wh",
    "what::d[in] -acc -wh",
    "who::d[an] -subj3 -q -wh",
    "who::d[an] -acc -wh",
    "::T<= +q +wh Q",
    "::q -q",
]

grammar = (
    polar_questions
    + wh_words
    + questions
    + verbs
    + [
        "you::d[an] -subj2",
        "you::d[an] -acc",
        "I::d[an] -subj1",
        "me::d[an] -acc",
        "he::d[an] -subj3",
        "him::d[an] -acc",
        "she::d[an] -subj3",
        "her::d[an] -acc",
        "::d[an]= +theme theme[an]",
        # "that::C= +r +rel[in] d[in] -acc",
        # "that::C= +r +rel[in] d[in] -subj3",
        # "who::C= +r +rel[an] d[an] -acc",
        # "who::C= +r +rel[an] d[an] -subj3",
        "::=>v =d[an] V",
    ]
)


def irregular_map(s: str) -> str:
    exceptions = [
        ("run-PAST", "ran"),
        ("run-ing", "running"),
        ("see-PAST", "saw"),
        ("try-3PRES", "tries"),
        ("try-PAST", "tried"),
    ]

    regulars = [("-PAST", "ed"), ("-3PRES", "s"),
                ("-2PRES", ""), ("-1PRES", "")]

    for a, b in exceptions + regulars:
        s = s.replace(a, b)

    return s.replace("-", "")


for noun in agentive_nouns:
    grammar.append(f"{noun}::N[an]")

for noun in objects:
    grammar.append(f"{noun}::N[in]")

for d in proper_names:
    grammar.append(f"{d}::d[an] -subj3")
    grammar.append(f"{d}::d[an] -acc")


for d in det:
    for sub_cat in ["in", "an"]:
        grammar.append(f"{d}::N[{sub_cat}]= d[{sub_cat}] -theme")
        grammar.append(f"{d}::N[{sub_cat}]= d[{sub_cat}] -subj3")
        grammar.append(f"{d}::N[{sub_cat}]= d[{sub_cat}] -acc")
        grammar.append(
            f"{d}[OBJ_REL]::N[{sub_cat}]= d[{sub_cat}] -acc -rel[{sub_cat}]")
        grammar.append(
            f"{d}[SUB_REL]::N[{sub_cat}]= d[{sub_cat}] -subj3 -rel[{sub_cat}]"
        )

for a in aux:
    grammar.append(f"{a}::V= +subj3 T")
    grammar.append(f"{a}::V= +subj2 T")
    grammar.append(f"{a}::V= +subj1 T")
    grammar.append(f"{a}::V= q= +subj3 T")
    grammar.append(f"{a}::V= q= +subj2 T")
    grammar.append(f"{a}::V= q= +subj1 T")
    grammar.append(f"{a}::V= r= +subj3 T")
    grammar.append(f"{a}::V= r= +subj2 T")
    grammar.append(f"{a}::V= r= +subj1 T")


progressive = [
    "am::prog= +subj1 T",
    "are::prog= +subj2 T",
    "is::prog= +subj3 T",
    "am::prog= q= +subj1 T",
    "are::prog= q= +subj2 T",
    "is::prog= q= +subj3 T",
    "am::prog= r= +subj1 T",
    "are::prog= r= +subj2 T",
    "is::prog= r= +subj3 T",
    "ing::=>V prog",
]

past = ["PAST::=>V +subj3 t", "PAST::=>V +subj2 t", "PAST::=>V +subj1 t"]

grammar = grammar + progressive + past

grammar = (
    "\n".join(grammar)
    + """
::T= C
::t= T
::t= r= T
::r -r
3PRES::=>V +subj3 t
2PRES::=>V +subj2 t
1PRES::=>V +subj1 t
"""
)

with open("reference.grammar", "w") as f:
    f.write(grammar)

lexicon = Lexicon(grammar)

# strings = []
# for p in lexicon.generate_grammar(
#     "C", max_strings=5000000, n_beams=500000, min_log_prob=-500, max_steps=48
# ):
#     strings.append(p)

# for s in random.choices(strings, k=20):
#     print(irregular_map(str(s)))

print("_" * 10)
strings = []
for p in lexicon.generate_grammar(
    "Q", max_strings=100000000, n_beams=5000000, min_log_prob=-500, max_steps=48
):
    strings.append(p)

with open("questions.txt", "w") as f:
    for string in strings:
        # f.write(str(string) + "\n")
        sentence = irregular_map(str(string))
        # if '[SUB_REL]' in sentence or '[OBJ_REL]' in sentence:
        #     # Remove the bracketed parts
        #     cleaned_sentence = sentence.replace('[SUB_REL]', '').replace('[OBJ_REL]', '')
        #     # Clean up any extra spaces
        #     cleaned_sentence = ' '.join(cleaned_sentence.split())
        #     f.write(cleaned_sentence + '\n')
        f.write(sentence + '?\n')

print("Strings saved to questions.txt")

# for s in random.choices(strings, k=20):
#     print(irregular_map(str(s)))


# with open("generated_sentences.txt", "w") as f:
#     f.write("=" * 20 + "\n")

#     for s in random.choices(strings, k=20):
#         sentence = irregular_map(str(s))
#         print(sentence)
#         f.write(sentence + "\n")

#     f.write("=" * 20 + "\n")

#     strings = []
#     for p in lexicon.generate_grammar(
#         "Q", max_strings=2000000, n_beams=5000000, min_log_prob=-500, max_steps=48
#     ):
#         strings.append(p)

#     for s in random.choices(strings, k=20):
#         sentence = irregular_map(str(s))
#         print(sentence)
#         f.write(sentence + "\n")
