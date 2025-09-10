# English Stimuli Generator

This repo has a few scripts to generate stimuli in English according to a reference MG grammar.
It uses the [python-mg](github.com/michaelgoodale/python-mg) package.

## Usage

```bash
uv run grammar.py
```

## Disabling subject or objects relative clauses or wh-words

To remove a wh word, just remove one of the following lines of code (`subj3` for subjects and `acc` for accusatives/objects):

```python
    "what::d[in] -subj3 -q -wh",
    "what::d[in] -acc -wh",
    "who::d[an] -subj3 -q -wh",
    "who::d[an] -acc -wh",
```

To remove a relative clause remove one of the following lines of code

```python
        grammar.append(f"{d}[OBJ_REL]::N[{sub_cat}]= d[{sub_cat}] -acc -rel[{sub_cat}]")
        grammar.append(
            f"{d}[SUB_REL]::N[{sub_cat}]= d[{sub_cat}] -subj3 -rel[{sub_cat}]"
        )

Note that you will need to remove `[OBJ_REL]` or `[SUB_REL]` as they tag the determiner, e.g. "The[OBJ_REL] man that I saw".
```
