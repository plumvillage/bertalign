import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

from bertalign.utils import yield_overlaps


from huggingface_hub import InferenceClient

from diskcache import Cache
cache = Cache(os.path.join(os.getenv("GEMS_TOOLS_DATA_PATH"), "data_LaBSE_embeddings_cache_timestamp_sync"))

def store_embedding_to_cache(text, embedding):
    text = text.replace(" [CURSOR_POSITION] ", " ") #ingnore cursor_position for embedding
    cache.set(text, embedding, expire=604800)  # 1 week

def get_embedding_from_cache(text):
    text = text.replace(" [CURSOR_POSITION] ", " ")
    return cache[text]

def has_embedding_in_cache(text):
    text = text.replace(" [CURSOR_POSITION] ", " ")
    return text in cache



class EncoderLocal:
    def __init__(self, model_name):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenve("HUGGINGFACE_API_KEY")
        )

    def get_embedding(self, texts):
        response = self.client.predict(
            model_id="sentence-transformers/LaBSE",
            inputs=texts
        )
        return response[0]["embedding"]
    
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

        print(f"Found {len(cached_embeddings)} cached embeddings, missing {len(missing_texts)}")
        
        # Get embeddings for missing texts
        if missing_texts:
            missing_embeddings = self.get_embedding(missing_texts)
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