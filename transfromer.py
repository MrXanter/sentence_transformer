from ast import arg
from numpy import argpartition, argsort, inf
from sentence_transformers import SentenceTransformer
from scipy.special import softmax
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
"The sun was bright enough to warm the entire valley.",
"A gentle breeze moved through the trees as the day grew warmer.", 
"Clouds gathered slowly, hinting at the possibility of rain later.",
"The new AI system can analyze data faster than previous models.",
"Machine learning algorithms continue to improve with more training examples.",
"The company released an update that enhances the model reasoning abilities.",
"She boarded the early train heading for the capital city.",
"After a long drive, he finally arrived at the coastal village.",
"Tourists wandered through the narrow streets looking for local food.",
"The cat slept peacefully on the windowsill all afternoon.",
]
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
similarities = softmax(similarities)

len_sentenses = len(sentences)
k = 0
row = similarities[k]
row[k] = -inf
top3_idx = argpartition(row, -3)[-3:]
top3_idx_sorted = top3_idx[argsort(row[top3_idx])[::-1]]
similar_sentences = [sentences[i] for i in top3_idx_sorted]
print (f"Top 3 similar sentences to: '{sentences[k]}'")
for sentence in similar_sentences:
    print(f"  {sentence}")