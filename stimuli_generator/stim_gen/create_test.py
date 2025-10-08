import random

# File paths
relative_clauses_file = "/scratch2/mrenaudin/GTH/stimuli_generator/relative_clauses.txt"
test_relative_clauses_file = "/scratch2/mrenaudin/GTH/stimuli_generator/test_relative_clauses.txt"

# Set seed for reproducibility
random.seed(42)

# Read all relative_clauses
with open(relative_clauses_file, 'r', encoding='utf-8') as f:
    relative_clauses = [line.strip() for line in f if line.strip()]

# Check we have enough sentences
num_to_sample = 1000
if len(relative_clauses) < num_to_sample:
    raise ValueError(f"Not enough sentences in relative_clauses.txt. Available: {len(relative_clauses)}")

# Sample 1000 sentences
sampled_relative_clauses = random.sample(relative_clauses, num_to_sample)

# Write sampled sentences to test_relative_clauses.txt
with open(test_relative_clauses_file, 'w', encoding='utf-8') as f:
    for sentence in sampled_relative_clauses:
        f.write(sentence + '\n')

# Remove sampled sentences from original relative_clauses list
remaining_relative_clauses = [q for q in relative_clauses if q not in sampled_relative_clauses]

# Overwrite relative_clauses.txt with remaining sentences
with open(relative_clauses_file, 'w', encoding='utf-8') as f:
    for sentence in remaining_relative_clauses:
        f.write(sentence + '\n')

print(f"✅ Created {test_relative_clauses_file} with 1000 random sentences")
print(f"✅ Updated {relative_clauses_file} with remaining {len(remaining_relative_clauses)} sentences")
import random

# File paths
relative_clauses_file = "/scratch2/mrenaudin/GTH/stimuli_generator/relative_clauses.txt"
test_relative_clauses_file = "/scratch2/mrenaudin/GTH/stimuli_generator/test_relative_clauses.txt"

# Set seed for reproducibility
random.seed(42)

# Read all relative_clauses
with open(relative_clauses_file, 'r', encoding='utf-8') as f:
    relative_clauses = [line.strip() for line in f if line.strip()]

# Check we have enough sentences
num_to_sample = 1000
if len(relative_clauses) < num_to_sample:
    raise ValueError(f"Not enough sentences in relative_clauses.txt. Available: {len(relative_clauses)}")

# Sample 1000 sentences
sampled_relative_clauses = random.sample(relative_clauses, num_to_sample)

# Write sampled sentences to test_relative_clauses.txt
with open(test_relative_clauses_file, 'w', encoding='utf-8') as f:
    for sentence in sampled_relative_clauses:
        f.write(sentence + '\n')

# Remove sampled sentences from original relative_clauses list
remaining_relative_clauses = [q for q in relative_clauses if q not in sampled_relative_clauses]

# Overwrite relative_clauses.txt with remaining sentences
with open(relative_clauses_file, 'w', encoding='utf-8') as f:
    for sentence in remaining_relative_clauses:
        f.write(sentence + '\n')

print(f"✅ Created {test_relative_clauses_file} with 1000 random sentences")
print(f"✅ Updated {relative_clauses_file} with remaining {len(remaining_relative_clauses)} sentences")
