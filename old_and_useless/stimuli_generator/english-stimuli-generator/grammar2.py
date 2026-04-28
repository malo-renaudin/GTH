from python_mg import Lexicon, SyntacticStructure
import random
import os
import sys
import gc
import time
from collections import deque

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

# Configuration
class Config:
    # File paths
    ORC_OUTPUT = "orc_sentences.txt"
    QUESTIONS_OUTPUT = "questions.txt"
    CHECKPOINT_DIR = "checkpoints"
    
    # Generation limits
    MAX_ORC_SENTENCES = 1000000  # 1 million
    MAX_QUESTION_SENTENCES = 5000000  # 5 million
    
    # Memory management
    BUFFER_SIZE = 10000  # Write to file every N sentences
    CHECKPOINT_INTERVAL = 50000  # Save progress every N sentences
    
    # Performance tuning
    MAX_STEPS = 30  # Reduce from 48 to limit complexity
    BEAM_SIZE = 1000  # Limit beam search
    
    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

def create_grammar():
    """Build the complete grammar (your existing code)"""
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
    
    polar_questions = ["::T<= +q Q"]
    
    wh_words = [
        "what::d[in] -acc -wh",
        "who::d[an] -acc -wh", 
        "which::N[in]= d[in] -acc -wh",
        "which::N[an]= d[an] -acc -wh",
        "whom::d[an] -acc -wh",
        "::T<= +q +wh Q",
        "::q -q",
    ]
    
    grammar = (
        polar_questions + wh_words + questions + verbs + [
            "you::d[an] -subj2", "you::d[an] -acc",
            "I::d[an] -subj1", "me::d[an] -acc", 
            "he::d[an] -subj3", "him::d[an] -acc",
            "she::d[an] -subj3", "her::d[an] -acc",
            "::d[an]= +theme theme[an]",
            "that::C= +r +rel[in] d[in] -acc",
            "that::C= +r +rel[in] d[in] -subj3", 
            "who::C= +r +rel[an] d[an] -acc",
            "who::C= +r +rel[an] d[an] -subj3",
            "::=>v =d[an] V",
        ]
    )
    
    # Add nouns
    for noun in agentive_nouns:
        grammar.append(f"{noun}::N[an]")
    for noun in objects:
        grammar.append(f"{noun}::N[in]")
        
    # Add proper names
    for name in proper_names:
        grammar.append(f"{name}::d[an] -subj3")
        grammar.append(f"{name}::d[an] -acc")
        
    # Add determiners
    for d in det:
        for sub_cat in ["in", "an"]:
            grammar.append(f"{d}::N[{sub_cat}]= d[{sub_cat}] -theme")
            grammar.append(f"{d}::N[{sub_cat}]= d[{sub_cat}] -subj3")
            grammar.append(f"{d}::N[{sub_cat}]= d[{sub_cat}] -acc")
            grammar.append(f"{d}[OBJ_REL]::N[{sub_cat}]= d[{sub_cat}] -acc -rel[{sub_cat}]")
            
    # Add auxiliaries
    for a in aux:
        for person in ["subj3", "subj2", "subj1"]:
            grammar.append(f"{a}::V= +{person} T")
            grammar.append(f"{a}::V= q= +{person} T") 
            grammar.append(f"{a}::V= r= +{person} T")
            
    # Add progressive and past
    progressive = [
        "am::prog= +subj1 T", "are::prog= +subj2 T", "is::prog= +subj3 T",
        "am::prog= q= +subj1 T", "are::prog= q= +subj2 T", "is::prog= q= +subj3 T",
        "am::prog= r= +subj1 T", "are::prog= r= +subj2 T", "is::prog= r= +subj3 T",
        "ing::=>V prog",
    ]
    
    past = ["PAST::=>V +subj3 t", "PAST::=>V +subj2 t", "PAST::=>V +subj1 t"]
    
    grammar = grammar + progressive + past
    
    grammar = "\n".join(grammar) + """
::T= C
::t= T  
::t= r= T
::r -r
3PRES::=>V +subj3 t
2PRES::=>V +subj2 t
1PRES::=>V +subj1 t
"""
    return grammar

def irregular_map(s: str) -> str:
    """Apply morphological rules"""
    exceptions = [
        ("run-PAST", "ran"), ("run-ing", "running"),
        ("see-PAST", "saw"), ("try-3PRES", "tries"), ("try-PAST", "tried"),
    ]
    regulars = [("-PAST", "ed"), ("-3PRES", "s"), ("-2PRES", ""), ("-1PRES", "")]
    
    for old, new in exceptions + regulars:
        s = s.replace(old, new)
    return s.replace("-", "")

class SentenceFilter:
    """Efficient sentence filtering"""
    def __init__(self):
        self.agentive_set = set(agentive_nouns)
        self.verb_set = set(['talk', 'see', 'devour', 'want', 'run'])
        
    def is_valid_orc(self, sentence: str) -> bool:
        """Filter for object relative clauses"""
        if "[OBJ_REL]" not in sentence:
            return False
            
        # Check for repeated nouns or verbs
        words = sentence.split()
        noun_counts = sum(1 for word in words if word in self.agentive_set)
        verb_counts = sum(1 for word in words if word in self.verb_set)
        
        return noun_counts <= len(set(word for word in words if word in self.agentive_set)) and \
               verb_counts <= len(set(word for word in words if word in self.verb_set))
    
    def is_valid_question(self, sentence: str) -> bool:
        """Filter for questions"""
        if "[OBJ_REL]" in sentence:
            return False
            
        words = sentence.split()
        noun_counts = sum(1 for word in words if word in self.agentive_set) 
        verb_counts = sum(1 for word in words if word in self.verb_set)
        
        return noun_counts <= len(set(word for word in words if word in self.agentive_set)) and \
               verb_counts <= len(set(word for word in words if word in self.verb_set))

class HighVolumeGenerator:
    """Memory-efficient high-volume sentence generator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.filter = SentenceFilter()
        
        # Build lexicon
        grammar = create_grammar()
        with open("grammar.txt", "w") as f:
            f.write(grammar)
        self.lexicon = Lexicon(grammar)
        
    def load_checkpoint(self, checkpoint_file: str) -> int:
        """Load generation progress from checkpoint"""
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return int(f.read().strip())
        return 0
    
    def save_checkpoint(self, checkpoint_file: str, count: int):
        """Save generation progress"""
        with open(checkpoint_file, 'w') as f:
            f.write(str(count))
    
    def generate_orc_sentences(self):
        """Generate object relative clauses with checkpointing"""
        print("Starting ORC generation...")
        
        checkpoint_file = os.path.join(self.config.CHECKPOINT_DIR, "orc_progress.txt")
        start_count = self.load_checkpoint(checkpoint_file)
        
        sentence_buffer = deque()
        generated_count = start_count
        
        # Open file in append mode if resuming
        mode = 'a' if start_count > 0 else 'w'
        
        with open(self.config.ORC_OUTPUT, mode) as f:
            generator = self.lexicon.generate_grammar(
                "C", 
                max_strings=self.config.MAX_ORC_SENTENCES * 10,  # Generate extra to account for filtering
                n_beams=self.config.BEAM_SIZE,
                min_log_prob=None,
                max_steps=self.config.MAX_STEPS
            )
            
            skipped = 0
            for p in generator:
                if generated_count >= self.config.MAX_ORC_SENTENCES:
                    break
                
                # Skip already generated sentences when resuming
                if skipped < start_count:
                    skipped += 1
                    continue
                
                sentence = irregular_map(str(p))
                
                if self.filter.is_valid_orc(sentence):
                    clean_sentence = sentence.replace("[OBJ_REL]", "")
                    sentence_buffer.append(clean_sentence)
                    generated_count += 1
                    
                    # Batch write to file
                    if len(sentence_buffer) >= self.config.BUFFER_SIZE:
                        f.write('\n'.join(sentence_buffer) + '\n')
                        f.flush()
                        sentence_buffer.clear()
                        gc.collect()  # Force garbage collection
                    
                    # Save checkpoint
                    if generated_count % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(checkpoint_file, generated_count)
                        print(f"Generated {generated_count:,} ORC sentences")
            
            # Write remaining buffer
            if sentence_buffer:
                f.write('\n'.join(sentence_buffer) + '\n')
        
        print(f"Completed ORC generation: {generated_count:,} sentences")
        return generated_count
    
    def generate_questions(self):
        """Generate questions with checkpointing"""
        print("Starting question generation...")
        
        checkpoint_file = os.path.join(self.config.CHECKPOINT_DIR, "questions_progress.txt")
        start_count = self.load_checkpoint(checkpoint_file)
        
        sentence_buffer = deque()
        generated_count = start_count
        
        mode = 'a' if start_count > 0 else 'w'
        
        with open(self.config.QUESTIONS_OUTPUT, mode) as f:
            generator = self.lexicon.generate_grammar(
                "Q",
                max_strings=self.config.MAX_QUESTION_SENTENCES * 10,
                n_beams=self.config.BEAM_SIZE,
                min_log_prob=None,
                max_steps=self.config.MAX_STEPS
            )
            
            skipped = 0
            for p in generator:
                if generated_count >= self.config.MAX_QUESTION_SENTENCES:
                    break
                
                if skipped < start_count:
                    skipped += 1
                    continue
                
                sentence = irregular_map(str(p))
                
                if self.filter.is_valid_question(sentence):
                    sentence_buffer.append(sentence + '?')
                    generated_count += 1
                    
                    if len(sentence_buffer) >= self.config.BUFFER_SIZE:
                        f.write('\n'.join(sentence_buffer) + '\n')
                        f.flush()
                        sentence_buffer.clear()
                        gc.collect()
                    
                    if generated_count % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(checkpoint_file, generated_count)
                        print(f"Generated {generated_count:,} questions")
            
            if sentence_buffer:
                f.write('\n'.join(sentence_buffer) + '\n')
        
        print(f"Completed question generation: {generated_count:,} sentences")
        return generated_count

def main():
    """Main execution function"""
    random.seed(100)
    
    config = Config()
    generator = HighVolumeGenerator(config)
    
    start_time = time.time()
    
    try:
        # Generate ORC sentences
        orc_count = generator.generate_orc_sentences()
        
        print("=" * 50)
        
        # Generate questions  
        q_count = generator.generate_questions()
        
        total_time = time.time() - start_time
        total_sentences = orc_count + q_count
        
        print(f"\nGeneration Complete!")
        print(f"ORC sentences: {orc_count:,}")
        print(f"Questions: {q_count:,}")
        print(f"Total sentences: {total_sentences:,}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Rate: {total_sentences/total_time:.1f} sentences/second")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted. Progress saved in checkpoints.")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise

if __name__ == "__main__":
    main()