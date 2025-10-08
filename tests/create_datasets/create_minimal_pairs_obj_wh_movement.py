import csv
import json

# Dataset of 100 matched pairs: declarative sentences and object wh-questions
# Each pair tests the same verb in both contexts

dataset = [
    # Basic transitive verbs
    {"declarative": "The boy kicked the", "wh_question": "What did the boy kick", "verb": "kick"},
    {"declarative": "The girl threw the", "wh_question": "What did the girl throw", "verb": "throw"},
    {"declarative": "The man ate the", "wh_question": "What did the man eat", "verb": "eat"},
    {"declarative": "The woman bought the", "wh_question": "What did the woman buy", "verb": "buy"},
    {"declarative": "The child found the", "wh_question": "What did the child find", "verb": "find"},
    
    {"declarative": "The dog chased the", "wh_question": "What did the dog chase", "verb": "chase"},
    {"declarative": "The cat caught the", "wh_question": "What did the cat catch", "verb": "catch"},
    {"declarative": "The teacher read the", "wh_question": "What did the teacher read", "verb": "read"},
    {"declarative": "The student wrote the", "wh_question": "What did the student write", "verb": "write"},
    {"declarative": "The artist painted the", "wh_question": "What did the artist paint", "verb": "paint"},
    
    {"declarative": "The chef cooked the", "wh_question": "What did the chef cook", "verb": "cook"},
    {"declarative": "The baby dropped the", "wh_question": "What did the baby drop", "verb": "drop"},
    {"declarative": "The player hit the", "wh_question": "What did the player hit", "verb": "hit"},
    {"declarative": "The driver saw the", "wh_question": "What did the driver see", "verb": "see"},
    {"declarative": "The worker fixed the", "wh_question": "What did the worker fix", "verb": "fix"},
    
    {"declarative": "The mother made the", "wh_question": "What did the mother make", "verb": "make"},
    {"declarative": "The father built the", "wh_question": "What did the father build", "verb": "build"},
    {"declarative": "The doctor examined the", "wh_question": "What did the doctor examine", "verb": "examine"},
    {"declarative": "The nurse checked the", "wh_question": "What did the nurse check", "verb": "check"},
    {"declarative": "The lawyer reviewed the", "wh_question": "What did the lawyer review", "verb": "review"},
    
    # Different subjects
    {"declarative": "The scientist discovered the", "wh_question": "What did the scientist discover", "verb": "discover"},
    {"declarative": "The musician played the", "wh_question": "What did the musician play", "verb": "play"},
    {"declarative": "The athlete won the", "wh_question": "What did the athlete win", "verb": "win"},
    {"declarative": "The manager approved the", "wh_question": "What did the manager approve", "verb": "approve"},
    {"declarative": "The director filmed the", "wh_question": "What did the director film", "verb": "film"},
    
    {"declarative": "The pilot flew the", "wh_question": "What did the pilot fly", "verb": "fly"},
    {"declarative": "The sailor steered the", "wh_question": "What did the sailor steer", "verb": "steer"},
    {"declarative": "The farmer grew the", "wh_question": "What did the farmer grow", "verb": "grow"},
    {"declarative": "The baker baked the", "wh_question": "What did the baker bake", "verb": "bake"},
    {"declarative": "The tailor sewed the", "wh_question": "What did the tailor sew", "verb": "sew"},
    
    {"declarative": "The carpenter built the", "wh_question": "What did the carpenter build", "verb": "build"},
    {"declarative": "The plumber repaired the", "wh_question": "What did the plumber repair", "verb": "repair"},
    {"declarative": "The mechanic fixed the", "wh_question": "What did the mechanic fix", "verb": "fix"},
    {"declarative": "The engineer designed the", "wh_question": "What did the engineer design", "verb": "design"},
    {"declarative": "The architect drew the", "wh_question": "What did the architect draw", "verb": "draw"},
    
    # Varied verbs
    {"declarative": "The police arrested the", "wh_question": "What did the police arrest", "verb": "arrest"},
    {"declarative": "The company launched the", "wh_question": "What did the company launch", "verb": "launch"},
    {"declarative": "The team developed the", "wh_question": "What did the team develop", "verb": "develop"},
    {"declarative": "The committee discussed the", "wh_question": "What did the committee discuss", "verb": "discuss"},
    {"declarative": "The group organized the", "wh_question": "What did the group organize", "verb": "organize"},
    
    {"declarative": "The boy broke the", "wh_question": "What did the boy break", "verb": "break"},
    {"declarative": "The girl opened the", "wh_question": "What did the girl open", "verb": "open"},
    {"declarative": "The man closed the", "wh_question": "What did the man close", "verb": "close"},
    {"declarative": "The woman cleaned the", "wh_question": "What did the woman clean", "verb": "clean"},
    {"declarative": "The child touched the", "wh_question": "What did the child touch", "verb": "touch"},
    
    {"declarative": "The dog buried the", "wh_question": "What did the dog bury", "verb": "bury"},
    {"declarative": "The cat scratched the", "wh_question": "What did the cat scratch", "verb": "scratch"},
    {"declarative": "The bird carried the", "wh_question": "What did the bird carry", "verb": "carry"},
    {"declarative": "The horse pulled the", "wh_question": "What did the horse pull", "verb": "pull"},
    {"declarative": "The monkey grabbed the", "wh_question": "What did the monkey grab", "verb": "grab"},
    
    # More complex subjects
    {"declarative": "The student answered the", "wh_question": "What did the student answer", "verb": "answer"},
    {"declarative": "The teacher explained the", "wh_question": "What did the teacher explain", "verb": "explain"},
    {"declarative": "The professor taught the", "wh_question": "What did the professor teach", "verb": "teach"},
    {"declarative": "The researcher analyzed the", "wh_question": "What did the researcher analyze", "verb": "analyze"},
    {"declarative": "The author published the", "wh_question": "What did the author publish", "verb": "publish"},
    
    {"declarative": "The editor revised the", "wh_question": "What did the editor revise", "verb": "revise"},
    {"declarative": "The journalist reported the", "wh_question": "What did the journalist report", "verb": "report"},
    {"declarative": "The photographer captured the", "wh_question": "What did the photographer capture", "verb": "capture"},
    {"declarative": "The designer created the", "wh_question": "What did the designer create", "verb": "create"},
    {"declarative": "The programmer coded the", "wh_question": "What did the programmer code", "verb": "code"},
    
    {"declarative": "The gardener watered the", "wh_question": "What did the gardener water", "verb": "water"},
    {"declarative": "The cleaner mopped the", "wh_question": "What did the cleaner mop", "verb": "mop"},
    {"declarative": "The painter colored the", "wh_question": "What did the painter color", "verb": "color"},
    {"declarative": "The builder constructed the", "wh_question": "What did the builder construct", "verb": "construct"},
    {"declarative": "The investor bought the", "wh_question": "What did the investor buy", "verb": "buy"},
    
    # Action verbs
    {"declarative": "The boy lifted the", "wh_question": "What did the boy lift", "verb": "lift"},
    {"declarative": "The girl pushed the", "wh_question": "What did the girl push", "verb": "push"},
    {"declarative": "The man pulled the", "wh_question": "What did the man pull", "verb": "pull"},
    {"declarative": "The woman carried the", "wh_question": "What did the woman carry", "verb": "carry"},
    {"declarative": "The child held the", "wh_question": "What did the child hold", "verb": "hold"},
    
    {"declarative": "The athlete trained the", "wh_question": "What did the athlete train", "verb": "train"},
    {"declarative": "The coach taught the", "wh_question": "What did the coach teach", "verb": "teach"},
    {"declarative": "The captain led the", "wh_question": "What did the captain lead", "verb": "lead"},
    {"declarative": "The guard protected the", "wh_question": "What did the guard protect", "verb": "protect"},
    {"declarative": "The soldier defended the", "wh_question": "What did the soldier defend", "verb": "defend"},
    
    # Perception/cognitive verbs
    {"declarative": "The boy heard the", "wh_question": "What did the boy hear", "verb": "hear"},
    {"declarative": "The girl noticed the", "wh_question": "What did the girl notice", "verb": "notice"},
    {"declarative": "The man remembered the", "wh_question": "What did the man remember", "verb": "remember"},
    {"declarative": "The woman forgot the", "wh_question": "What did the woman forget", "verb": "forget"},
    {"declarative": "The child learned the", "wh_question": "What did the child learn", "verb": "learn"},
    
    {"declarative": "The detective solved the", "wh_question": "What did the detective solve", "verb": "solve"},
    {"declarative": "The jury decided the", "wh_question": "What did the jury decide", "verb": "decide"},
    {"declarative": "The judge ruled the", "wh_question": "What did the judge rule", "verb": "rule"},
    {"declarative": "The witness saw the", "wh_question": "What did the witness see", "verb": "see"},
    {"declarative": "The victim reported the", "wh_question": "What did the victim report", "verb": "report"},
    
    # Communication verbs
    {"declarative": "The boy said the", "wh_question": "What did the boy say", "verb": "say"},
    {"declarative": "The girl told the", "wh_question": "What did the girl tell", "verb": "tell"},
    {"declarative": "The man announced the", "wh_question": "What did the man announce", "verb": "announce"},
    {"declarative": "The woman mentioned the", "wh_question": "What did the woman mention", "verb": "mention"},
    {"declarative": "The child asked the", "wh_question": "What did the child ask", "verb": "ask"},
    
    {"declarative": "The speaker presented the", "wh_question": "What did the speaker present", "verb": "present"},
    {"declarative": "The host introduced the", "wh_question": "What did the host introduce", "verb": "introduce"},
    {"declarative": "The guide showed the", "wh_question": "What did the guide show", "verb": "show"},
    {"declarative": "The instructor demonstrated the", "wh_question": "What did the instructor demonstrate", "verb": "demonstrate"},
    {"declarative": "The expert recommended the", "wh_question": "What did the expert recommend", "verb": "recommend"},
    
    # Transfer/exchange verbs
    {"declarative": "The boy gave the", "wh_question": "What did the boy give", "verb": "give"},
    {"declarative": "The girl sent the", "wh_question": "What did the girl send", "verb": "send"},
    {"declarative": "The man sold the", "wh_question": "What did the man sell", "verb": "sell"},
    {"declarative": "The woman received the", "wh_question": "What did the woman receive", "verb": "receive"},
    {"declarative": "The child shared the", "wh_question": "What did the child share", "verb": "share"},
    
    {"declarative": "The store offered the", "wh_question": "What did the store offer", "verb": "offer"},
    {"declarative": "The customer ordered the", "wh_question": "What did the customer order", "verb": "order"},
    {"declarative": "The waiter served the", "wh_question": "What did the waiter serve", "verb": "serve"},
    {"declarative": "The cashier scanned the", "wh_question": "What did the cashier scan", "verb": "scan"},
    {"declarative": "The clerk packaged the", "wh_question": "What did the clerk package", "verb": "package"},
]

# Export as CSV
csv_filename = "wh_question_dataset.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['declarative', 'wh_question', 'verb'])
    writer.writeheader()
    writer.writerows(dataset)

# Export as JSON
# json_filename = "wh_question_dataset.json"
# with open(json_filename, 'w', encoding='utf-8') as f:
#     json.dump(dataset, f, indent=2)

print(f"Dataset created with {len(dataset)} matched pairs")
print(f"Saved to: {csv_filename} and {json_filename}")
print("\nFirst 3 examples:")
for i, item in enumerate(dataset[:3]):
    print(f"\n{i+1}. Declarative: {item['declarative']}")
    print(f"   Wh-question: {item['wh_question']}")
    print(f"   Verb: {item['verb']}")