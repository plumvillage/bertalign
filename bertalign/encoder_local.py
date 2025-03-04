import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

import openai

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps
import tiktoken

from diskcache import Cache
cache = Cache(os.path.join(os.getenv("GEMS_DATA_CACHE_PATH"), "data_LaBSE_embeddings_cache_timestamp_sync"))

def store_embedding_to_cache(text, embedding):
    text = replace_excluded_from_embedding(text)
    cache.set(text, embedding, expire=604800)  # 1 week

def get_embedding_from_cache(text):
    text = replace_excluded_from_embedding(text)
    return cache[text]

def has_embedding_in_cache(text):
    text = replace_excluded_from_embedding(text)
    return text in cache

def replace_excluded_from_embedding(text):
    text = text.replace(" [CURSOR_POSITION] ", " ") #ingnore cursor_position for embedding
    text = text.replace("[CURSOR_POSITION] ", " ") #start of sentence
    text = text.replace(" [CURSOR_POSITION]", " ") #end of sentence
    text = text.replace("_", " ") #ignore special characters inserted by chapter parsing
    text = text.strip()
    return text

class EncoderLocal:
    def __init__(self, model_name):
        #self.model = SentenceTransformer(model_name, device="cpu")

        print(f"SentenceTransformer using model: {model_name} (may require some time for download)")
        self.model = SentenceTransformer(model_name, cache_folder=os.path.join(os.getenv("GEMS_DATA_CACHE_PATH"), "data_SentenceTransformers_model_cache")) #autoselect device
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = list(yield_overlaps(sents, num_overlaps))
        
        # Check cache for existing embeddings
        cached_embeddings = []
        missing_texts = []
        for text in overlaps:
            if has_embedding_in_cache(text):
                cached_embeddings.append(get_embedding_from_cache(text))
            else:
                missing_texts.append(text)

        print(f"Found {len(cached_embeddings)} cached embeddings, missing {len(missing_texts)} (generating, may take a while)")
        
        # Get embeddings for missing texts
        if missing_texts:
            for text in missing_texts:
                print(f"Missing: {replace_excluded_from_embedding(text)}")
            missing_embeddings = self.model.encode(missing_texts)
            for text, embedding in zip(missing_texts, missing_embeddings):
                store_embedding_to_cache(text, embedding)
                cached_embeddings.append(embedding)
        
        # Ensure the order of embeddings matches the order of overlaps
        embeddings = []
        cache_index = 0
        for text in overlaps:
            if has_embedding_in_cache(text):
                embeddings.append(cached_embeddings[cache_index])
                cache_index += 1
            else:
                embeddings.append(get_embedding_from_cache(text))
        
        sent_vecs = np.array(embeddings)
        embedding_dim = sent_vecs.shape[1]
        sent_vecs.resize((num_overlaps, len(sents), embedding_dim))

        len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps])
        len_vecs.resize((num_overlaps, len(sents)))

        return sent_vecs, len_vecs