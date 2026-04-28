import random

filename = 'wh3.txt'
output_file = 'wh3_sample.txt'
num_samples = 3000

with open(filename, 'r') as f:
    sentences = f.readlines()

sampled = random.sample(sentences, min(num_samples, len(sentences)))

with open(output_file, 'w') as f:
    f.writelines(sampled)

print(f"Sampled {len(sampled)} sentences to {output_file}")