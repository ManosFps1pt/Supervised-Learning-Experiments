# Cell 1
import os
import numpy as np
import pandas as pd
from gdown import download

# Cell 3
def load_embeddings(id, filename="embeddings.csv.gz"):
    if not os.path.exists(filename):
        download(id=id, output=filename, quiet=False)

    df = pd.read_csv(filename, header=None, engine='pyarrow')
    words = df.iloc[:, 0].tolist()
    embeddings = df.iloc[:, 1:].to_numpy(dtype=float)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return words, np.array(embeddings)

words, embeddings = load_embeddings('1biezzvCn3TkxRLy-7t6LA6M_bNFV8xl_')
word_to_index = {word: i for i, word in enumerate(words)}
index_to_word = {i: word for i, word in enumerate(words)}

print( "Πρώτες 3 λέξεις:", words[:3] )
print( "Το index της λέξης 'ρωγμή':", word_to_index['ρωγμή'])
print( "Οι διαστάσεις του πίνακα των embeddings:", embeddings.shape )

# Cell 5
import torch

embeddings_tensor = torch.FloatTensor(embeddings)
print( "Οι διαστάσεις του τανιστή των embeddings:", embeddings_tensor.shape )

# Cell 7
def similarity(i, j):
    return torch.dot(embeddings_tensor[i], embeddings_tensor[j])

word1 = 'κόβω'
word2 = 'ψαλίδι'
word1index = word_to_index[word1]
word2index = word_to_index[word2]
print( "Ομοιότητα('{}', '{}') = {:0.3f}".format( word1, word2, similarity(word1index, word2index) ) )

# Cell 9
def distance_matrix(similarity_matrix: torch.FloatTensor) -> torch.FloatTensor:
    result = torch.full_like(similarity_matrix, float('inf'))  # Default to infinity
    result[(similarity_matrix >= 0.3)] = 3  # Dashed line
    result[(similarity_matrix >= 0.4)] = 2  # Faint line
    result[similarity_matrix >= 0.5] = 1  # Dark line
    return result

sim = torch.FloatTensor(similarity(word1index, word2index))
dist = distance_matrix(sim)

print( "Απόσταση('{}', '{}') =  {}".format( word1, word2, dist.item() ) )

# Cell 11
emb = embeddings_tensor[word_to_index['λέξη']]
sim = torch.matmul(emb, torch.t(embeddings_tensor)) # ανάστροφος πίνακας για να γίνεται ο πολλαπλασιασμός
dist = distance_matrix(sim)

top10 = torch.topk(sim, k=10) # Top 10 λέξεις
for i in top10.indices:
    print( '{:0.3f}'.format(sim[i]), words[i], dist[i].item() )

# Cell 13
sim_full = torch.matmul(embeddings_tensor, torch.t(embeddings_tensor))
dist_full = distance_matrix(sim_full)
top10 = torch.topk(sim_full, k=10, axis=1)

j = word_to_index['λέξη']
for i in top10.indices[j]:
    print( '{:0.3f}'.format(sim[i]), words[i], dist[i].item() )

# Cell 16
goal_reached = False
current_word_index = word_to_index['λέξη']
greedy_best = torch.topk(sim_full, k=2, axis=1)
word_chain = ['λέξη']
max_steps = 10
step = 0

while not goal_reached and step < max_steps:
  current_word_index = greedy_best.indices[current_word_index][1]
  word_chain.append(words[current_word_index])
  if current_word_index == word_to_index['αλυσίδα']:
    goal_reached = True
  step += 1

print(word_chain)

# Cell 18
goal_reached = False
current_word_index = word_to_index['λέξη']
greedy_best = torch.topk(sim_full, k=32057, axis=1)
word_chain = ['λέξη']
max_steps = 100
step = 0

while not goal_reached and step < max_steps:
  neig = 0
  selected = False
  while not selected:
    current_word_index = greedy_best.indices[current_word_index][neig]
    if not index_to_word[int(current_word_index.item())] in word_chain:
      word_chain.append(words[current_word_index])
      selected = True
    neig += 1
  if current_word_index == word_to_index['αλυσίδα']:
    goal_reached = True
  step += 1

print(word_chain)

# Cell 20
# TODO
def beam_search(start_word, end_word, beam_width=300, max_steps=300):
    start_index = word_to_index[start_word]
    end_index = word_to_index[end_word]

    beam = [([start_index], 0)]  # (path, total_distance)

    for step in range(max_steps):
        new_beam = []
        for path, total_distance in beam:
            current_index = path[-1]

            # Get similarities to all other words
            similarities = torch.matmul(embeddings_tensor[current_index], torch.t(embeddings_tensor))

            # Get distances based on similarities
            distances = distance_matrix(similarities)

            # Sort by distance and get top candidates
            top_indices = torch.topk(similarities, k=beam_width).indices

            for neighbor_index in top_indices:
                neighbor_index = neighbor_index.item()

                # Avoid cycles
                if neighbor_index not in path:
                    new_path = path + [neighbor_index]
                    new_distance = total_distance + distances[neighbor_index].item()
                    new_beam.append((new_path, new_distance))

        # Sort by total distance and keep top candidates
        new_beam.sort(key=lambda x: x[1])
        beam = new_beam[:beam_width]

        # Check if goal is reached
        if any(path[-1] == end_index for path, _ in beam):
            best_path, _ = next(x for x in beam if x[path][-1] == end_index)
            return [words[index] for index in best_path]

    # If goal not reached within max_steps, return best path so far
    best_path, _ = beam[0]
    return [words[index] for index in best_path]

word_chain = beam_search('αλυσίδα', 'λέξη')
print(word_chain)

