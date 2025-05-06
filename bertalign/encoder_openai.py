import numpy as np

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

from bertalign.utils import yield_overlaps
import tiktoken

import concurrent.futures

import os

from diskcache import Cache
cache = Cache(os.path.join(os.getenv("GEMS_TOOLS_DATA_PATH"), "data_openai_embeddings_cache_timestamp_sync"))

def store_embedding_to_cache(text, embedding):
    text = text.replace(" [CURSOR_POSITION] ", " ") #ingnore cursor_position for embedding
    cache.set(text, embedding, expire=604800)  # 1 week

def get_embedding_from_cache(text):
    text = text.replace(" [CURSOR_POSITION] ", " ")
    return cache[text]

def has_embedding_in_cache(text):
    text = text.replace(" [CURSOR_POSITION] ", " ")
    return text in cache

def batch_embeddings(texts, model_name="text-embedding-3-small"):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens_per_text = 8191

    text_lists = []
    current_text_list = []
    current_token_count = 0

    for text in texts:
        text = text.replace(" [CURSOR_POSITION] ", " ")
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

    def get_embeddings_for_text_list(text_list):
        response = client.embeddings.create(input=text_list, model=model_name)
        return [item["embedding"] for item in response.data]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_embeddings_for_text_list, text_list) for text_list in text_lists]
        for future in concurrent.futures.as_completed(futures):
            embeddings += future.result()

    return embeddings

class EncoderOpenAIEmbeddings:
    def __init__(self, model_name="text-embedding-3-small"):
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

        print(f"Found {len(cached_embeddings)} cached embeddings, missing {len(missing_texts)}")

        # Get embeddings for missing texts
        if missing_texts:
            print(missing_texts)
            missing_embeddings = batch_embeddings(missing_texts, self.model_name)
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