from python_mg import Lexicon, SyntacticStructure
import random

random.seed(100)

proper_names = [
    "John", "Mary", "Paul", "Sarah", "James", "Lisa", "Michael", "Jennifer",
    "David", "Linda", "Robert", "Patricia", "William", "Elizabeth", "Richard", "Barbara",
    "Ralph", "Amber", "Mason", "Danielle", "Roy", "Rose", "Eugene", "Brittany", "Louis", "Diana", "Philip",
    "Abigail", "Bobby", "Jane", "Thomas", "Susan", "Charles", "Jessica", "Christopher", "Nancy", "Daniel", "Karen",
    "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna",
    "Steven", "Carol", "Kenneth", "Ruth", "Joshua", "Sharon", "Kevin", "Michelle",
    "Brian", "Laura", "George", "Emily", "Edward", "Kimberly", "Ronald", "Deborah",
    "Timothy", "Dorothy", "Jason", "Amy", "Jeffrey", "Angela", "Ryan", "Ashley",
    "Jacob", "Brenda", "Gary", "Emma", "Nicholas", "Olivia", "Eric", "Cynthia",
    "Jonathan", "Marie", "Stephen", "Janet", "Larry", "Catherine", "Justin", "Frances",
    "Scott", "Christine", "Brandon", "Samantha", "Benjamin", "Debra", "Samuel", "Rachel",
    "Gregory", "Carolyn", "Frank", "Virginia", "Raymond", "Maria", "Alexander", "Heather",
    "Patrick", "Diane", "Jack", "Julie", "Dennis", "Joyce", "Jerry", "Victoria",
    "Tyler", "Kelly", "Aaron", "Christina", "Jose", "Joan", "Henry", "Evelyn",
    "Adam", "Lauren", "Douglas", "Judith", "Nathan", "Megan", "Peter", "Cheryl",
    "Zachary", "Andrea", "Kyle", "Hannah", "Noah", "Jacqueline", "Alan", "Martha",
    "Carl", "Gloria", "Harold", "Teresa", "Roger", "Sara", "Arthur", "Janice",
    "Lawrence", "Marie", "Sean", "Julia", "Christian", "Heather", "Albert", "Diane",
    "Wayne", "Julie", "Ralph", "Joyce", "Roy", "Virginia", "Eugene", "Deborah",
    "Louis", "Rachel", "Philip", "Carolyn", "Bobby", "Janet"
]

agentive_nouns = [
    "man", "woman", "boy", "girl", "dog", "cat", "person", "child",
    "baby", "adult", "teenager", "student", "teacher", "doctor", "nurse", "worker",
    "wife", "partner", "girlfriend", "boyfriend", "athlete", "player", "coach", "referee", "spectator",
    "fan", "tourist", "traveler", "passenger", "pedestrian", "cyclist", "runner", "swimmer", "climber", "hiker",
    "lawyer", "judge", "police_officer", "firefighter", "paramedic", "pilot", "captain", "sailor",
    "soldier", "officer", "general", "sergeant", "detective", "investigator", "scientist", "researcher",
    "professor", "principal", "librarian", "counselor", "therapist", "psychologist", "psychiatrist", "dentist",
    "surgeon", "physician", "veterinarian", "pharmacist", "optometrist", "chiropractor", "radiologist", "cardiologist",
    "engineer", "architect", "designer", "developer", "programmer", "analyst", "consultant", "manager",
    "director", "executive", "president", "supervisor", "coordinator", "administrator", "secretary", "assistant",
    "clerk", "cashier", "salesperson", "merchant", "vendor", "retailer", "wholesaler", "distributor",
    "manufacturer", "producer", "creator", "inventor", "innovator", "entrepreneur", "founder", "owner",
    "chef", "cook", "baker", "waiter", "waitress", "server", "bartender", "barista",
    "mechanic", "technician", "electrician", "plumber", "carpenter", "painter", "roofer", "mason",
    "gardener", "landscaper", "farmer", "rancher", "fisherman", "hunter", "forester", "miner",
    "driver", "chauffeur", "taxi_driver", "bus_driver", "truck_driver", "delivery_person", "courier", "messenger",
    "writer", "author", "journalist", "reporter", "editor", "publisher", "blogger", "columnist",
    "artist", "painter", "sculptor", "musician", "singer", "dancer", "actor", "actress",
    "photographer", "filmmaker", "director", "producer", "screenwriter", "composer", "conductor", "performer",
    "athlete", "footballer", "basketball_player", "tennis_player", "golfer", "boxer", "wrestler", "gymnast",
    "neighbor", "friend", "colleague", "classmate", "roommate", "housemate", "tenant", "landlord",
    "customer", "client", "patron", "guest", "visitor", "stranger", "acquaintance", "companion",
    "leader", "follower", "member", "participant", "volunteer", "activist", "supporter", "opponent",
    "expert", "specialist", "professional", "amateur", "beginner", "veteran", "rookie", "trainee"
]

objects = [
    "cake", "apple", "cherry",
    "banana", "orange", "grape", "strawberry", "blueberry", "raspberry", "peach", "pear",
    "pineapple", "mango", "kiwi", "lemon", "lime", "coconut", "avocado", "tomato",
    "bread", "sandwich", "pizza", "pasta", "rice", "noodles", "soup", "salad",
    "cookie", "pie", "donut", "muffin", "cupcake", "ice_cream", "chocolate", "candy",
    "cheese", "milk", "yogurt", "butter", "eggs", "meat", "chicken", "beef",
    "table", "chair", "sofa", "bed", "lamp", "mirror", "clock", "television",
    "computer", "phone", "tablet", "camera", "book", "magazine", "newspaper", "pen",
    "pencil", "paper", "notebook", "folder", "envelope", "stamp", "key", "wallet",
    "purse", "bag", "backpack", "suitcase", "umbrella", "hat", "coat", "jacket",
    "shirt", "pants", "dress", "shoes", "socks", "gloves", "scarf", "belt",
    "hammer", "screwdriver", "wrench", "drill", "saw", "knife", "spoon", "fork",
    "plate", "bowl", "cup", "glass", "bottle", "jar", "can", "box",
    "car", "bicycle", "motorcycle", "bus", "train", "airplane", "boat", "ship",
    "ball", "toy", "game", "puzzle", "doll", "robot", "kite", "balloon",
    "tree", "flower", "leaf", "branch", "rock", "stone", "shell", "feather",
    "mountain", "hill", "river", "lake", "ocean", "beach", "forest", "field",
    "desk", "keyboard", "mouse", "monitor", "printer", "calculator", "stapler", "ruler",
    "eraser", "marker", "crayon", "paintbrush", "canvas", "easel", "sculpture", "vase",
    "guitar", "piano", "violin", "drum", "flute", "trumpet", "saxophone", "clarinet",
    "bat", "racket", "club", "helmet", "glove", "uniform", "trophy", "medal"
]
# det = ["the", "a", "every", "some", "any", "all", "each", "both", "either", "neither", "this",
#        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
#        "many", "few", "several", "most", "much", "little", "enough", "more", "less", "other", "another",
#        "such", "what", "which", "whose", "no"]
det = ['the']
aux = ["can", "would", "may", "should"]



verb_surface = ['talk', 'see', 'devour', 'want', 'run']

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
    #what::d[in] -subj3 -q -wh",
    "what::d[in] -acc -wh",
    #"who::d[an] -subj3 -q -wh",
    "who::d[an] -acc -wh",
    "which::N[in]= d[in] -acc -wh",   # which + noun, inanimate
    "which::N[an]= d[an] -acc -wh",   # which + noun, animate
    "whom::d[an] -acc -wh",
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
        "that::C= +r +rel[in] d[in] -acc",
        "that::C= +r +rel[in] d[in] -subj3",
        "who::C= +r +rel[an] d[an] -acc",
        "who::C= +r +rel[an] d[an] -subj3",
        "::=>v =d[an] V",
        # "::C= c[comp]",      # complement clause for bridge verbs
        # "that::C= c[comp]"
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

    regulars = [("-PAST", "ed"), ("-3PRES", "s"), ("-2PRES", ""), ("-1PRES", "")]

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
        grammar.append(f"{d}[OBJ_REL]::N[{sub_cat}]= d[{sub_cat}] -acc -rel[{sub_cat}]")
        # grammar.append(
        #     f"{d}[SUB_REL]::N[{sub_cat}]= d[{sub_cat}] -subj3 -rel[{sub_cat}]"
        # )

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
    _ = f.write(grammar)

lexicon = Lexicon(grammar)

# strings: list[SyntacticStructure] = []
with open("orc.txt", "w") as f:
    for p in lexicon.generate_grammar(
        "C", max_strings=1000000, n_beams=None, min_log_prob=None, max_steps=48
    ):
        # strings.append(p)
        sentence = irregular_map(str(p))
        if "[OBJ_REL]" not in sentence:
            continue
        sentence = sentence.replace("[OBJ_REL]", "")
        
        if any(sentence.count(noun) > 1 for noun in agentive_nouns):
            continue
        if any(sentence.count(verb)>1 for verb in verb_surface):
            continue
        f.write(sentence + '\n')

# for s in random.choices(strings, k=20):
#     print(str(s))
#     print(irregular_map(str(s)))
    
#save to file relative clause
# with open("orc.txt", "w") as f:
#     for string in strings : 
#         sentence = irregular_map(str(string))
#         if "[OBJ_REL]" not in sentence:
#             continue
#         sentence = sentence.replace("[OBJ_REL]", "")
        
#         if any(sentence.count(noun) > 1 for noun in agentive_nouns):
#             continue
#         if any(sentence.count(verb)>1 for verb in verb_surface):
#             continue
#         f.write(sentence + '\n')
    

print("_" * 10)
# strings = []
with open("q_obj.txt", "w") as f:
    for p in lexicon.generate_grammar(
        "Q", max_strings=20000000, n_beams=None, min_log_prob=None, max_steps=48
    ):
        sentence = irregular_map(str(p))
        if "[OBJ_REL]" in sentence:
            continue
        # sentence = sentence.replace("[SUBJ_REL]", "")
        if any(sentence.count(noun) > 1 for noun in agentive_nouns):
            continue
        all_verbs = verb_surface + [v.split("::")[0] for v in verb_surface]
        if any(sentence.count(verb) > 1 for verb in all_verbs):
            continue
        f.write(sentence + '?\n')
    # strings.append(p)

# for s in random.choices(strings, k=20):
#     print(irregular_map(str(s)))
    
# with open("q_obj.txt", "w") as f:
    # # for string in strings : 
    #     sentence = irregular_map(str(string))
    #     if "[OBJ_REL]" in sentence:
    #         continue
    #     # sentence = sentence.replace("[SUBJ_REL]", "")
    #     if any(sentence.count(noun) > 1 for noun in agentive_nouns):
    #         continue
    #     all_verbs = verb_surface + [v.split("::")[0] for v in verb_surface]
    #     if any(sentence.count(verb) > 1 for verb in all_verbs):
    #         continue
    #     f.write(sentence + '?\n')
       
            
