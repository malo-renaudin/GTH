####
# The aim of this script is to create new training set by enriching data/english_data/train.txt with data from 
# generate_simple_datasets/generated_simple_datasets. Across the new datasets, we should add the same amount of sentences each time. 
# Since the goal is to add sentences with a specific syntactic structures, mixed with declarative sentences, so that the model avoids 
# using lexical heuristics to compute syntactic dependencies, the number of sentences to have is the maximum 
# amount of declaratives with the right vocabulary. This is generate_simple_datasets/generated_simple_datasets/declaratives_from_orc7.txt
# + generate_simple_datasets/generated_simple_datasets/declaratives_from_wh5.txt. Then, as a first step, 
# we create a dataset with a lot of orc, and one with a lot of wh. We'll follow the formula to augment the datasets that is detailed in the overleaf,.
####
import random
import argparse

dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument("--structure", type=str, default="orc", help="syntactic structure to add to the dataset (orc or wh)")
dataset_parser.add_argument("--target_freq", type=float, default=0.1, help="target frequency of the target structure in the dataset")
dataset_parser.add_argument("--name", type=str, default="default_name", help="name of the dataset")

parser = argparse.ArgumentParser(
    parents=[dataset_parser], description="Sampling of source datasets to create proper training sets."
)
args = parser.parse_args()

base_train_file = "data/english_data/train.txt"
orc = "generate_simple_datasets/generated_simple_datasets/orc7.txt"
wh  = "generate_simple_datasets/generated_simple_datasets/wh5.txt"
declaratives_from_orc = "generate_simple_datasets/generated_simple_datasets/declaratives_from_orc7.txt"
declaratives_from_wh = "generate_simple_datasets/generated_simple_datasets/declaratives_from_wh5.txt"

def count_lines(path):
    with open(path, "r") as f:
        return sum(1 for _ in f)


# N is the number of lines in base_train_file
N = count_lines(base_train_file)
# n is the number of sentences we add to the dataset
n = count_lines(declaratives_from_orc) + count_lines(declaratives_from_wh)

start_orc_freq = 0.84/100
start_wh_freq= 0.9/100

def preprocess_sentence(line):
    line = line.strip()
    line = line.replace("<unk>", "")
    line = line.replace("<eos>", "<|endoftext|>")
    # ensure <|endoftext|> at the end
    if not line.endswith("<|endoftext|>"):
        line = line + " <|endoftext|>"
    return line + "\n"

#function to augment 1 target frequency (enough for now, but will need improvement fast enough)
def augment_dataset(N, n, start_freq, target_freq, base_dataset, target_dataset, declaratives_from_wh, declaratives_from_orc):
    raw_n_target_to_add = int(round((target_freq - start_freq) * N + target_freq * n))

    # We build exactly n additional lines: n_target_to_add from target structure,
    # and the remaining lines from declaratives.
    with open(declaratives_from_wh, "r") as f:
        wh_sentences = f.readlines()
    with open(declaratives_from_orc, "r") as f:
        orc_sentences = f.readlines()
    declaratives = wh_sentences + orc_sentences

    with open(target_dataset, "r") as f:
        target = f.readlines()

    max_target_available = min(len(target), n)
    n_target_to_add = max(0, min(raw_n_target_to_add, max_target_available))
    n_declaratives_to_add = n - n_target_to_add

    if n_declaratives_to_add > len(declaratives):
        raise ValueError(
            f"Not enough declaratives to build dataset: need {n_declaratives_to_add}, "
            f"have {len(declaratives)}."
        )

    random.shuffle(declaratives)
    random.shuffle(target)

    sentences_to_add = [preprocess_sentence(s) for s in declaratives[:n_declaratives_to_add] + target[:n_target_to_add]]
    print(
        f"Requested target additions={raw_n_target_to_add}; using {n_target_to_add} target + "
        f"{n_declaratives_to_add} declaratives (total added={len(sentences_to_add)})."
    )

    with open(base_dataset, "r") as f:
        base = [preprocess_sentence(line) for line in f]
    final = base + sentences_to_add
    random.shuffle(final)
    return final

if __name__ == "__main__":
    if args.structure == "orc":
        start_freq = start_orc_freq
        target_dataset = orc
    elif args.structure == "wh":
        start_freq = start_wh_freq
        target_dataset = wh
    else:
        raise ValueError("Invalid structure. Choose 'orc' or 'wh'.")

    augmented_train = augment_dataset(N, n, start_freq, args.target_freq, base_train_file, target_dataset, declaratives_from_wh, declaratives_from_orc)
    with open(f"data/english_data/train_{args.name}_augmented.txt", "w") as f:
        f.writelines(augmented_train)
    