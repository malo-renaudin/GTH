#!/usr/bin/env python3
"""
Generate WH-question and declarative corpora at scale.

Sentence counts (struct=1, default vocabulary)
---------------------------------------------
Template        Description                               Count
──────────────────────────────────────────────────────────────────────
A1 non-anim     obj-wh, N1 subj, stacked adj (×3 wh)    110,073,600
A1 animate      obj-wh, N1 subj, stacked adj (×2 wh)     83,865,600
A2              which-NP bare, N1+N2 cross-product        628,992,000
B1              obj-wh, N2 subj, stacked adj (×3 wh)    110,073,600
──────────────────────────────────────────────────────────────────────
Total WH sentences                                      ~933,004,800  (~933 M)
Distribution: A1 20.8%, A2 67.4%, B1 11.8%

Adj stacking: adj1 × adj2 = 16 × 15 = 240 combinations per noun (A1, B1).
A2 uses only adj1 (16 options) on subject N1; no adjective on the which-NP.

Aux inventory per subject-form (14 types, 98 (aux, v_form) pairs):
  Simple  : did / does–do / is–are              (3)
  Modal   : can could will would should          (9)
             might must shall may
  Perfect : has–have / had                       (2)
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODALS = ["can", "could", "will", "would", "should", "might", "must", "shall", "may"]

ANIMATE_ONLY_VERBS = {
    "help", "greet", "call", "guide", "train", "support", "consult", "assist",
}

# ---------------------------------------------------------------------------
# Vocabulary
# "" at index 0 of ADJ/ADV lists = no modifier — every combination is unique
# by construction, so no deduplication set is needed.
# ---------------------------------------------------------------------------
n1_opts = [
    # ── original 20 ──────────────────────────────────────────────────────
    "student",      "doctor",      "pilot",       "officer",     "athlete",
    "artist",       "teacher",     "lawyer",      "judge",       "engineer",
    "scientist",    "nurse",       "chef",        "manager",     "reporter",
    "banker",       "professor",   "coach",       "captain",     "writer",
    # ── 40 additions ─────────────────────────────────────────────────────
    "accountant",   "architect",   "detective",   "diplomat",    "editor",
    "firefighter",  "librarian",   "mechanic",    "musician",    "pharmacist",
    "photographer", "politician",  "programmer",  "psychologist","surgeon",
    "technician",   "therapist",   "veterinarian","administrator","inspector",
    "electrician",  "consultant",  "counselor",   "analyst",     "designer",
    "director",     "instructor",  "investigator","planner",     "producer",
    "specialist",   "supervisor",  "translator",  "auditor",     "biologist",
    "chemist",      "economist",   "historian",   "geographer",  "philosopher",
]  # 60 nouns

n2_opts = [
    # ── original 20 ──────────────────────────────────────────────────────
    "child",       "girl",        "boy",         "patient",     "client",
    "tourist",     "neighbor",    "guest",       "worker",      "driver",
    "visitor",     "customer",    "assistant",   "colleague",   "intern",
    "resident",    "passenger",   "volunteer",   "rookie",      "spectator",
    # ── 40 additions ─────────────────────────────────────────────────────
    "beginner",    "bystander",   "citizen",     "contestant",  "dancer",
    "employee",    "expert",      "journalist",  "observer",    "pedestrian",
    "scholar",     "singer",      "soldier",     "stranger",    "participant",
    "researcher",  "civilian",    "cadet",        "apprentice",  "recruit",
    "traveler",    "witness",     "refugee",     "activist",    "explorer",
    "migrant",     "pupil",       "disciple",    "follower",    "trainee",
    "listener",    "newcomer",    "attendee",    "initiate",    "learner",
    "mentee",      "delegate",    "guardian",    "correspondent","subordinate",
]  # 60 nouns

v_opts = [
    # (base, past/pp, progressive)
    # ── Non-animate verbs (What / Who / Whom allowed) ────────────────────
    ("visit",     "visited",     "visiting"),
    ("follow",    "followed",    "following"),
    ("avoid",     "avoided",     "avoiding"),
    ("observe",   "observed",    "observing"),
    ("watch",     "watched",     "watching"),
    ("teach",     "taught",      "teaching"),
    ("accompany", "accompanied", "accompanying"),
    # ── Animate-only verbs (Who / Whom only) ─────────────────────────────
    ("help",      "helped",      "helping"),
    ("greet",     "greeted",     "greeting"),
    ("call",      "called",      "calling"),
    ("guide",     "guided",      "guiding"),
    ("train",     "trained",     "training"),
    ("support",   "supported",   "supporting"),
    ("consult",   "consulted",   "consulting"),
    ("assist",    "assisted",    "assisting"),
]  # 15 verbs (7 non-animate, 8 animate-only)

adj1_opts = [
    "", "young", "tall", "smart", "brave", "kind",
    "famous", "strong", "busy", "calm", "honest", "proud",
    "gentle", "fair", "eager", "skilled",
]  # 16 options (index 0 = no adjective)

adj2_opts = [
    "", "creative", "serious", "friendly", "quiet", "active",
    "careful", "thoughtful", "diligent", "talented", "ambitious",
    "capable", "determined", "experienced", "motivated",
]  # 15 options

adv1_opts = [
    "", "probably", "certainly", "possibly", "apparently", "maybe",
    "perhaps", "clearly", "definitely", "obviously", "surely",
    "truly", "reportedly",
]  # 13 options

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_IRREGULARS = {"child": "children", "man": "men", "woman": "women"}


def pluralize_noun(noun: str) -> str:
    return _IRREGULARS.get(noun, noun + "s")


def get_present_3sg(v_base: str) -> str:
    """Return the 3rd-person singular present form."""
    if v_base.endswith(("s", "x", "z", "ch", "sh")):
        return v_base + "es"
    if v_base.endswith("y") and v_base[-2] not in "aeiou":
        return v_base[:-1] + "ies"
    return v_base + "s"


def get_wh_words_for_verb(v_base: str):
    if v_base in ANIMATE_ONLY_VERBS:
        return ("Who", "Whom")
    return ("What", "Who", "Whom")


def _build_aux_list(v_base: str, v_pp: str, v_ing: str, is_plur: bool):
    """
    Return list of (auxiliary, verb_form) pairs for:
      - simple past / present / progressive
      - 9 modals (base form)
      - perfect has–have / had (past-participle form)
    14 pairs total.
    """
    aux_list = [
        ("did",                          v_base),
        ("do" if is_plur else "does",    v_base),
        ("are" if is_plur else "is",     v_ing),
    ]
    for modal in MODALS:
        aux_list.append((modal, v_base))
    has_have = "have" if is_plur else "has"
    aux_list.append((has_have, v_pp))
    aux_list.append(("had",   v_pp))
    return aux_list  # 14 pairs


# ---------------------------------------------------------------------------
# Generation — struct=1 : WH-questions
# ---------------------------------------------------------------------------
def generate_wh_corpus(filename: str) -> int:
    """
    Stream all WH-question sentences to *filename* without a deduplication
    set, by iterating each template over only the parameters it uses.

    Three non-overlapping template families:

      A1  {Wh} {aux} the {adj1?}{adj2?}{N1} {adv?}{V}?
              (object-wh, N1 as subject, two stacked adjectives)

      A2  Which {N2} {aux} the {adj1?}{N1} {adv?}{V}?
              (which-NP question, bare N2, N1 as subject with one adjective)

      B1  {Wh} {aux} the {adj1?}{adj2?}{N2} {adv?}{V}?
              (object-wh, N2 as subject, non-animate verbs, two stacked adjectives)
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    count = 0

    with open(filename, "w", buffering=8 * 1024 * 1024) as fh:

        # ── A1 : object-wh, N1 as subject ─────────────────────────────────
        for n1 in n1_opts:
            for n1_plur in (False, True):
                n1f = pluralize_noun(n1) if n1_plur else n1
                for v_base, v_pp, v_ing in v_opts:
                    wh_words = get_wh_words_for_verb(v_base)
                    aux_n1 = _build_aux_list(v_base, v_pp, v_ing, n1_plur)
                    for adj1 in adj1_opts:
                        a1 = adj1 + " " if adj1 else ""
                        for adj2 in adj2_opts:
                            a2 = adj2 + " " if adj2 else ""
                            for adv in adv1_opts:
                                ad = adv + " " if adv else ""
                                for aux, vf in aux_n1:
                                    for wh in wh_words:
                                        fh.write(f"{wh} {aux} the {a1}{a2}{n1f} {ad}{vf}?\n")
                                        count += 1

        # ── A2 : which-NP, N1 as subject ──────────────────────────────────
        for n1 in n1_opts:
            for n1_plur in (False, True):
                n1f = pluralize_noun(n1) if n1_plur else n1
                for n2 in n2_opts:
                    for n2_plur in (False, True):
                        n2f = pluralize_noun(n2) if n2_plur else n2
                        for v_base, v_pp, v_ing in v_opts:
                            aux_n1 = _build_aux_list(v_base, v_pp, v_ing, n1_plur)
                            for adj1 in adj1_opts:
                                a1 = adj1 + " " if adj1 else ""
                                for adv in adv1_opts:
                                    ad = adv + " " if adv else ""
                                    for aux, vf in aux_n1:
                                        fh.write(
                                            f"Which {n2f} {aux} "
                                            f"the {a1}{n1f} {ad}{vf}?\n"
                                        )
                                        count += 1

        # ── B1 : object-wh, N2 as subject (non-animate verbs only) ────────
        for n2 in n2_opts:
            for n2_plur in (False, True):
                n2f = pluralize_noun(n2) if n2_plur else n2
                for v_base, v_pp, v_ing in v_opts:
                    if v_base in ANIMATE_ONLY_VERBS:
                        continue
                    wh_words = get_wh_words_for_verb(v_base)  # What/Who/Whom
                    aux_n2 = _build_aux_list(v_base, v_pp, v_ing, n2_plur)
                    for adj1 in adj1_opts:
                        a1 = adj1 + " " if adj1 else ""
                        for adj2 in adj2_opts:
                            a2 = adj2 + " " if adj2 else ""
                            for adv in adv1_opts:
                                ad = adv + " " if adv else ""
                                for aux, vf in aux_n2:
                                    for wh in wh_words:
                                        fh.write(f"{wh} {aux} the {a1}{a2}{n2f} {ad}{vf}?\n")
                                        count += 1

    return count


# ---------------------------------------------------------------------------
# Generation — struct=2 : Declaratives
# ---------------------------------------------------------------------------
def generate_declarative_corpus(filename: str) -> int:
    """
    Stream all declarative sentences (3 tenses: past / present / progressive)
    to *filename*. Returns the total sentence count.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    count = 0

    with open(filename, "w", buffering=8 * 1024 * 1024) as fh:
        for n1 in n1_opts:
            for n1_plur in (False, True):
                n1f = pluralize_noun(n1) if n1_plur else n1
                for n2 in n2_opts:
                    for n2_plur in (False, True):
                        n2f = pluralize_noun(n2) if n2_plur else n2
                        for v_base, v_pp, v_ing in v_opts:
                            v_pres = v_base if n1_plur else get_present_3sg(v_base)
                            aux_prog = "are" if n1_plur else "is"
                            for adj1 in adj1_opts:
                                a1 = adj1 + " " if adj1 else ""
                                for adj2 in adj2_opts:
                                    a2 = adj2 + " " if adj2 else ""
                                    for adv in adv1_opts:
                                        ad = adv + " " if adv else ""
                                        # Past
                                        fh.write(f"The {a1}{n1f} {ad}{v_pp} the {a2}{n2f}.\n")
                                        # Present
                                        fh.write(f"The {a1}{n1f} {ad}{v_pres} the {a2}{n2f}.\n")
                                        # Progressive (adv between aux and participle)
                                        fh.write(f"The {a1}{n1f} {aux_prog} {ad}{v_ing} the {a2}{n2f}.\n")
                                        count += 3

    return count


# ---------------------------------------------------------------------------
# Backwards-compatible entry point
# ---------------------------------------------------------------------------
def generate_final_corpus_to_file(struct, filename):
    if struct == 1:
        return generate_wh_corpus(filename)
    elif struct == 2:
        return generate_declarative_corpus(filename)
    else:
        raise ValueError(f"Unknown structure: {struct}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a large corpus of WH-questions or declaratives."
    )
    parser.add_argument("--output_file", type=str, default="datasets/wh.txt",
                        help="Path to save the generated corpus.")
    parser.add_argument("--structures", type=int, default=1,
                        help="Structures to generate: 1 for WH-Questions, 2 for Declaratives.")
    args = parser.parse_args()

    n = generate_final_corpus_to_file(args.structures, args.output_file)
    print(f"Generated {n:,} sentences -> {args.output_file}")


if __name__ == "__main__":
    main()