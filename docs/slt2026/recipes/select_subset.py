import re
from collections import Counter

with open('sentences.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

DIGRAPHS = {'lh', 'nh', 'ch', 'rr', 'ss', 'qu', 'gu'}

def get_features(sentence):
    s = sentence.lower()
    s = re.sub(r'[^a-záéíóúãõâêôàüç]', ' ', s)
    feats = Counter()
    pos = 0
    while pos < len(s):
        if s[pos] == ' ':
            pos += 1
            continue
        bigram = s[pos:pos+2] if pos + 1 < len(s) else ''
        if bigram in DIGRAPHS:
            feats[bigram] += 1
            pos += 2
        else:
            feats[s[pos]] += 1
            pos += 1
    return feats

all_features = [get_features(s) for s in sentences]

# Corpus-level phoneme distribution
total_counts = Counter()
for f in all_features:
    total_counts.update(f)
total = sum(total_counts.values())
target_dist = {k: v / total for k, v in total_counts.items()}

# Greedy selection: minimize L1 distance to target distribution at each step
n_select = 50
selected_indices = []
selected_set = set()
current_counts = Counter()
current_total = 0

for step in range(n_select):
    best_score = float('inf')
    best_idx = None
    all_keys = set(target_dist.keys())

    for idx, feats in enumerate(all_features):
        if idx in selected_set:
            continue
        feat_total = sum(feats.values())
        if feat_total == 0:
            continue
        new_counts = current_counts + feats
        new_total = current_total + feat_total
        dist = sum(abs(new_counts.get(k, 0) / new_total - target_dist.get(k, 0))
                   for k in all_keys | set(new_counts.keys()))
        if dist < best_score:
            best_score = dist
            best_idx = idx

    if best_idx is None:
        print(f"Warning: no candidate found at step {step}")
        break

    selected_indices.append(best_idx)
    selected_set.add(best_idx)
    current_counts.update(all_features[best_idx])
    current_total += sum(all_features[best_idx].values())

# Stats
subset_counts = Counter()
for si in selected_indices:
    subset_counts.update(all_features[si])
subset_total = sum(subset_counts.values())

coverage = len(subset_counts) / len(total_counts) * 100
l1 = sum(abs(subset_counts.get(k, 0) / subset_total - target_dist.get(k, 0))
         for k in set(total_counts) | set(subset_counts))
missing = set(total_counts.keys()) - set(subset_counts.keys())

print(f"Phoneme/digraph units in corpus: {len(total_counts)}")
print(f"Units covered by subset: {len(subset_counts)} ({coverage:.1f}%)")
print(f"L1 distance from full corpus distribution: {l1:.4f}")
print(f"Missing units: {sorted(missing)}")

# Save subset
output_lines = [sentences[si] for si in sorted(selected_indices)]
with open('sentences_50.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines) + '\n')

print(f"\nSaved {len(output_lines)} sentences to sentences_50.txt")
