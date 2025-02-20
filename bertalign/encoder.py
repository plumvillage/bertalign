import numpy as np

from dotenv import load_dotenv
load_dotenv()

import openai

# from sentence_transformers import SentenceTransformer
from utils import yield_overlaps

# class Encoder:
#     def __init__(self, model_name):
#         self.model = SentenceTransformer(model_name)
#         self.model_name = model_name

#     def transform(self, sents, num_overlaps):
#         overlaps = []
#         for line in yield_overlaps(sents, num_overlaps):
#             overlaps.append(line)

#         sent_vecs = self.model.encode(overlaps)
#         embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
#         sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

#         len_vecs = [len(line.encode("utf-8")) for line in overlaps]
#         len_vecs = np.array(len_vecs)
#         len_vecs.resize(num_overlaps, len(sents))

#         return sent_vecs, len_vecs
    
class EncoderOpenAIEmbeddings:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name

    def get_embeddings(self, texts):
        print(f"Getting embeddings for {len(texts)} texts using model {self.model_name}.")
        response = openai.Embedding.create(input=texts, model=self.model_name)
        return [item["embedding"] for item in response["data"]]

    def transform(self, sents, num_overlaps):
        overlaps = list(yield_overlaps(sents, num_overlaps))
        sent_vecs = np.array(self.get_embeddings(overlaps))
        
        embedding_dim = sent_vecs.shape[1]
        sent_vecs.resize((num_overlaps, len(sents), embedding_dim))

        len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps])
        len_vecs.resize((num_overlaps, len(sents)))

        return sent_vecs, len_vecs