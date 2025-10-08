import csv
import json

# Dataset of 100 matched pairs: matrix clauses and object relative clauses
# Each pair tests the same verb in both contexts

dataset = [
    # Basic transitive verbs
    {"matrix": "The boy kicked the", "object_relative": "The ball that the boy kicked", "verb": "kick"},
    {"matrix": "The girl threw the", "object_relative": "The ball that the girl threw", "verb": "throw"},
    {"matrix": "The man ate the", "object_relative": "The food that the man ate", "verb": "eat"},
    {"matrix": "The woman bought the", "object_relative": "The car that the woman bought", "verb": "buy"},
    {"matrix": "The child found the", "object_relative": "The toy that the child found", "verb": "find"},
    
    {"matrix": "The dog chased the", "object_relative": "The cat that the dog chased", "verb": "chase"},
    {"matrix": "The cat caught the", "object_relative": "The mouse that the cat caught", "verb": "catch"},
    {"matrix": "The teacher read the", "object_relative": "The book that the teacher read", "verb": "read"},
    {"matrix": "The student wrote the", "object_relative": "The essay that the student wrote", "verb": "write"},
    {"matrix": "The artist painted the", "object_relative": "The picture that the artist painted", "verb": "paint"},
    
    {"matrix": "The chef cooked the", "object_relative": "The meal that the chef cooked", "verb": "cook"},
    {"matrix": "The baby dropped the", "object_relative": "The cup that the baby dropped", "verb": "drop"},
    {"matrix": "The player hit the", "object_relative": "The ball that the player hit", "verb": "hit"},
    {"matrix": "The driver saw the", "object_relative": "The sign that the driver saw", "verb": "see"},
    {"matrix": "The worker fixed the", "object_relative": "The machine that the worker fixed", "verb": "fix"},
    
    {"matrix": "The mother made the", "object_relative": "The cake that the mother made", "verb": "make"},
    {"matrix": "The father built the", "object_relative": "The house that the father built", "verb": "build"},
    {"matrix": "The doctor examined the", "object_relative": "The patient that the doctor examined", "verb": "examine"},
    {"matrix": "The nurse checked the", "object_relative": "The chart that the nurse checked", "verb": "check"},
    {"matrix": "The lawyer reviewed the", "object_relative": "The contract that the lawyer reviewed", "verb": "review"},
    
    # Different head nouns
    {"matrix": "The scientist discovered the", "object_relative": "The element that the scientist discovered", "verb": "discover"},
    {"matrix": "The musician played the", "object_relative": "The song that the musician played", "verb": "play"},
    {"matrix": "The athlete won the", "object_relative": "The medal that the athlete won", "verb": "win"},
    {"matrix": "The manager approved the", "object_relative": "The plan that the manager approved", "verb": "approve"},
    {"matrix": "The director filmed the", "object_relative": "The scene that the director filmed", "verb": "film"},
    
    {"matrix": "The pilot flew the", "object_relative": "The plane that the pilot flew", "verb": "fly"},
    {"matrix": "The sailor steered the", "object_relative": "The ship that the sailor steered", "verb": "steer"},
    {"matrix": "The farmer grew the", "object_relative": "The crops that the farmer grew", "verb": "grow"},
    {"matrix": "The baker baked the", "object_relative": "The bread that the baker baked", "verb": "bake"},
    {"matrix": "The tailor sewed the", "object_relative": "The dress that the tailor sewed", "verb": "sew"},
    
    {"matrix": "The carpenter built the", "object_relative": "The table that the carpenter built", "verb": "build"},
    {"matrix": "The plumber repaired the", "object_relative": "The pipe that the plumber repaired", "verb": "repair"},
    {"matrix": "The mechanic fixed the", "object_relative": "The engine that the mechanic fixed", "verb": "fix"},
    {"matrix": "The engineer designed the", "object_relative": "The bridge that the engineer designed", "verb": "design"},
    {"matrix": "The architect drew the", "object_relative": "The blueprint that the architect drew", "verb": "draw"},
    
    # Varied verbs
    {"matrix": "The police arrested the", "object_relative": "The suspect that the police arrested", "verb": "arrest"},
    {"matrix": "The company launched the", "object_relative": "The product that the company launched", "verb": "launch"},
    {"matrix": "The team developed the", "object_relative": "The software that the team developed", "verb": "develop"},
    {"matrix": "The committee discussed the", "object_relative": "The proposal that the committee discussed", "verb": "discuss"},
    {"matrix": "The group organized the", "object_relative": "The event that the group organized", "verb": "organize"},
    
    {"matrix": "The boy broke the", "object_relative": "The window that the boy broke", "verb": "break"},
    {"matrix": "The girl opened the", "object_relative": "The door that the girl opened", "verb": "open"},
    {"matrix": "The man closed the", "object_relative": "The gate that the man closed", "verb": "close"},
    {"matrix": "The woman cleaned the", "object_relative": "The room that the woman cleaned", "verb": "clean"},
    {"matrix": "The child touched the", "object_relative": "The flower that the child touched", "verb": "touch"},
    
    {"matrix": "The dog buried the", "object_relative": "The bone that the dog buried", "verb": "bury"},
    {"matrix": "The cat scratched the", "object_relative": "The furniture that the cat scratched", "verb": "scratch"},
    {"matrix": "The bird carried the", "object_relative": "The twig that the bird carried", "verb": "carry"},
    {"matrix": "The horse pulled the", "object_relative": "The cart that the horse pulled", "verb": "pull"},
    {"matrix": "The monkey grabbed the", "object_relative": "The banana that the monkey grabbed", "verb": "grab"},
    
    # Complex subjects
    {"matrix": "The student answered the", "object_relative": "The question that the student answered", "verb": "answer"},
    {"matrix": "The teacher explained the", "object_relative": "The lesson that the teacher explained", "verb": "explain"},
    {"matrix": "The professor taught the", "object_relative": "The course that the professor taught", "verb": "teach"},
    {"matrix": "The researcher analyzed the", "object_relative": "The data that the researcher analyzed", "verb": "analyze"},
    {"matrix": "The author published the", "object_relative": "The book that the author published", "verb": "publish"},
    
    {"matrix": "The editor revised the", "object_relative": "The manuscript that the editor revised", "verb": "revise"},
    {"matrix": "The journalist reported the", "object_relative": "The story that the journalist reported", "verb": "report"},
    {"matrix": "The photographer captured the", "object_relative": "The moment that the photographer captured", "verb": "capture"},
    {"matrix": "The designer created the", "object_relative": "The logo that the designer created", "verb": "create"},
    {"matrix": "The programmer coded the", "object_relative": "The application that the programmer coded", "verb": "code"},
    
    {"matrix": "The gardener watered the", "object_relative": "The plants that the gardener watered", "verb": "water"},
    {"matrix": "The cleaner mopped the", "object_relative": "The floor that the cleaner mopped", "verb": "mop"},
    {"matrix": "The painter colored the", "object_relative": "The wall that the painter colored", "verb": "color"},
    {"matrix": "The builder constructed the", "object_relative": "The building that the builder constructed", "verb": "construct"},
    {"matrix": "The investor bought the", "object_relative": "The stock that the investor bought", "verb": "buy"},
    
    # Action verbs
    {"matrix": "The boy lifted the", "object_relative": "The box that the boy lifted", "verb": "lift"},
    {"matrix": "The girl pushed the", "object_relative": "The button that the girl pushed", "verb": "push"},
    {"matrix": "The man pulled the", "object_relative": "The rope that the man pulled", "verb": "pull"},
    {"matrix": "The woman carried the", "object_relative": "The bag that the woman carried", "verb": "carry"},
    {"matrix": "The child held the", "object_relative": "The balloon that the child held", "verb": "hold"},
    
    {"matrix": "The athlete trained the", "object_relative": "The rookie that the athlete trained", "verb": "train"},
    {"matrix": "The coach taught the", "object_relative": "The technique that the coach taught", "verb": "teach"},
    {"matrix": "The captain led the", "object_relative": "The team that the captain led", "verb": "lead"},
    {"matrix": "The guard protected the", "object_relative": "The treasure that the guard protected", "verb": "protect"},
    {"matrix": "The soldier defended the", "object_relative": "The fort that the soldier defended", "verb": "defend"},
    
    # Perception/cognitive verbs
    {"matrix": "The boy heard the", "object_relative": "The noise that the boy heard", "verb": "hear"},
    {"matrix": "The girl noticed the", "object_relative": "The mistake that the girl noticed", "verb": "notice"},
    {"matrix": "The man remembered the", "object_relative": "The password that the man remembered", "verb": "remember"},
    {"matrix": "The woman forgot the", "object_relative": "The appointment that the woman forgot", "verb": "forget"},
    {"matrix": "The child learned the", "object_relative": "The skill that the child learned", "verb": "learn"},
    
    {"matrix": "The detective solved the", "object_relative": "The case that the detective solved", "verb": "solve"},
    {"matrix": "The jury decided the", "object_relative": "The verdict that the jury decided", "verb": "decide"},
    {"matrix": "The judge ruled the", "object_relative": "The motion that the judge ruled", "verb": "rule"},
    {"matrix": "The witness saw the", "object_relative": "The crime that the witness saw", "verb": "see"},
    {"matrix": "The victim reported the", "object_relative": "The theft that the victim reported", "verb": "report"},
    
    # Communication verbs
    {"matrix": "The boy said the", "object_relative": "The word that the boy said", "verb": "say"},
    {"matrix": "The girl told the", "object_relative": "The story that the girl told", "verb": "tell"},
    {"matrix": "The man announced the", "object_relative": "The news that the man announced", "verb": "announce"},
    {"matrix": "The woman mentioned the", "object_relative": "The fact that the woman mentioned", "verb": "mention"},
    {"matrix": "The child asked the", "object_relative": "The question that the child asked", "verb": "ask"},
    
    {"matrix": "The speaker presented the", "object_relative": "The idea that the speaker presented", "verb": "present"},
    {"matrix": "The host introduced the", "object_relative": "The guest that the host introduced", "verb": "introduce"},
    {"matrix": "The guide showed the", "object_relative": "The path that the guide showed", "verb": "show"},
    {"matrix": "The instructor demonstrated the", "object_relative": "The method that the instructor demonstrated", "verb": "demonstrate"},
    {"matrix": "The expert recommended the", "object_relative": "The solution that the expert recommended", "verb": "recommend"},
    
    # Transfer/exchange verbs
    {"matrix": "The boy gave the", "object_relative": "The gift that the boy gave", "verb": "give"},
    {"matrix": "The girl sent the", "object_relative": "The letter that the girl sent", "verb": "send"},
    {"matrix": "The man sold the", "object_relative": "The car that the man sold", "verb": "sell"},
    {"matrix": "The woman received the", "object_relative": "The package that the woman received", "verb": "receive"},
    {"matrix": "The child shared the", "object_relative": "The candy that the child shared", "verb": "share"},
    
    {"matrix": "The store offered the", "object_relative": "The discount that the store offered", "verb": "offer"},
    {"matrix": "The customer ordered the", "object_relative": "The pizza that the customer ordered", "verb": "order"},
    {"matrix": "The waiter served the", "object_relative": "The dinner that the waiter served", "verb": "serve"},
    {"matrix": "The cashier scanned the", "object_relative": "The item that the cashier scanned", "verb": "scan"},
    {"matrix": "The clerk packaged the", "object_relative": "The order that the clerk packaged", "verb": "package"},
]

# Export as CSV
csv_filename = "object_relative_dataset.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['matrix', 'object_relative', 'verb'])
    writer.writeheader()
    writer.writerows(dataset)

# Export as JSON
json_filename = "object_relative_dataset.json"
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset created with {len(dataset)} matched pairs")
print(f"Saved to: {csv_filename} and {json_filename}")
print("\nFirst 3 examples:")
for i, item in enumerate(dataset[:3]):
    print(f"\n{i+1}. Matrix clause: {item['matrix']}")
    print(f"   Object relative: {item['object_relative']}")
    print(f"   Verb: {item['verb']}")
    
print("\n" + "="*60)
print("TESTING PATTERN:")
print("="*60)
print("Matrix clause: 'The boy kicked the ___'")
print("  → HIGH noun mass (object position unfilled)")
print("\nObject relative: 'The ball that the boy kicked ___'")
print("  → LOW noun mass (object position filled by 'ball')")
print("\nDifference = Matrix noun mass - Object relative noun mass")
print("  → Should be LARGE and POSITIVE if model understands")