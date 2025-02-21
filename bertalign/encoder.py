import numpy as np

from dotenv import load_dotenv
load_dotenv()

import openai

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps
import tiktoken

class Encoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(overlaps)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs

def batch_embeddings(texts, model_name="text-embedding-3-small"):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens_per_text = 8191

    text_lists = []
    current_text_list = []
    current_token_count = 0

    for text in texts:
        text_token_count = len(tokenizer.encode(text))
        if current_token_count + text_token_count > max_tokens_per_text:
            text_lists.append(current_text_list)
            current_text_list = []
            current_token_count = 0
        current_text_list.append(text)
        current_token_count += text_token_count
    text_lists.append(current_text_list)

    # Get embeddings for each list of texts
    embeddings = []
    print(f"Getting embeddings for {len(texts)} texts, split into {len(text_lists)} batches")
    for text_list in text_lists:
        response = openai.Embedding.create(input=text_list, model=model_name)
        embeddings += [item["embedding"] for item in response["data"]]

    return embeddings
class EncoderOpenAIEmbeddings:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = list(yield_overlaps(sents, num_overlaps))
        sent_vecs = np.array(batch_embeddings(overlaps))
        
        embedding_dim = sent_vecs.shape[1]
        sent_vecs.resize((num_overlaps, len(sents), embedding_dim))

        len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps])
        len_vecs.resize((num_overlaps, len(sents)))

        return sent_vecs, len_vecs