import os
import re
from typing import List

import numpy as np
import requests
from diskcache import Cache
from dotenv import load_dotenv

try:
    from bertalign.utils import yield_overlaps
except ModuleNotFoundError:
    from utils import yield_overlaps

load_dotenv()

cache = Cache(
    os.path.join(
        os.getenv("GEMS_DATA_CACHE_PATH"),
        "data_LaBSE_embeddings_cache_timestamp_sync",
    )
)


def store_embedding_to_cache(text, embedding):
    text = replace_excluded_from_embedding(text)
    embedding = np.array(embedding, dtype=np.float32)
    cache.set(text, embedding, expire=604800)  # 1 week


def get_embedding_from_cache(text):
    text = replace_excluded_from_embedding(text)
    embedding = cache[text]
    return np.array(embedding, dtype=np.float32)


def get_embedding_from_cache_or_none(text):
    text = replace_excluded_from_embedding(text)
    embedding = cache.get(text, default=None)
    if embedding is None:
        return None
    return np.array(embedding, dtype=np.float32)


def has_embedding_in_cache(text):
    text = replace_excluded_from_embedding(text)
    return text in cache


def replace_excluded_from_embedding(text):
    text = text.replace("[CURSOR_POSITION]", " ")
    text = text.replace("[PARAGRAPH_BREAK]", " ")
    text = text.replace("_", " ")
    text = replace_nbsp_with_space(text)
    text = reduce_whitespace_to_single_space(text)
    text = remove_unicode_control_chars(text)
    text = normalize_line_endings(text)
    text = text.strip()
    return text


def replace_nbsp_with_space(s):
    if "\xa0" in s:
        print(
            f"{os.path.basename(__file__)}: Replacing non-breaking spaces with regular spaces."
        )
    return s.replace("\xa0", " ")


def reduce_whitespace_to_single_space(s):
    if "\u3000" in s:
        print(
            f"{os.path.basename(__file__)}: Replacing IDEOGRAPHIC SPACE with regular spaces."
        )
    s = s.replace("\u3000", " ")
    return re.sub(r"\s+", " ", s)


def normalize_line_endings(s):
    return s.replace("\r\n", "\n").replace("\r", "\n")


def remove_unicode_control_chars(s):
    control_chars_regex = r"[\u0000-\u0009\u000B-\u000C\u000E-\u001F\u007F-\u009F]"
    return re.sub(control_chars_regex, " ", s)


class EncoderHfApi:
    def __init__(self, model_name):
        self.model_name = model_name
        self.hf_api_key = os.getenv("HF_API_KEY")
        if not self.hf_api_key:
            raise Exception("HF_API_KEY must be set")
        self.api_url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model_name}/pipeline/feature-extraction"
        )
        print(f"Hugging Face Inference API using model: {self.model_name}")

    def _query_embeddings(self, sentences: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {"inputs": {"sentences": sentences}}
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise Exception(f"Unexpected HF response format: {data}")
        return data

    def transform(self, sents, num_overlaps):
        overlaps = list(yield_overlaps(sents, num_overlaps))

        embeddings_by_index = {}
        missing_texts = []
        missing_indices = []
        for idx, text in enumerate(overlaps):
            cached_embedding = get_embedding_from_cache_or_none(text)
            if cached_embedding is not None:
                embeddings_by_index[idx] = cached_embedding
            else:
                missing_texts.append(text)
                missing_indices.append(idx)

        print(
            f"Found {len(embeddings_by_index)} cached embeddings, missing {len(missing_texts)}"
        )

        if missing_texts:
            print(f"Getting embeddings for {len(missing_texts)} texts from HF API")
            missing_embeddings = self._query_embeddings(missing_texts)
            missing_embeddings = np.array(missing_embeddings, dtype=np.float32)
            for idx, text, embedding in zip(
                missing_indices, missing_texts, missing_embeddings
            ):
                store_embedding_to_cache(text, embedding)
                embeddings_by_index[idx] = np.array(embedding, dtype=np.float32)

        embeddings = [embeddings_by_index.get(i) for i in range(len(overlaps))]
        if any(embedding is None for embedding in embeddings):
            missing_indices_after_fill = [
                i for i, embedding in enumerate(embeddings) if embedding is None
            ]
            for idx in missing_indices_after_fill:
                text = overlaps[idx]
                raise Exception(
                    f"Missing embedding after cache filling, this should not happen, for text: {text}"
                )

        sent_vecs = np.array(embeddings, dtype=np.float32)
        embedding_dim = sent_vecs.shape[1]
        sent_vecs.resize((num_overlaps, len(sents), embedding_dim))

        len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps])
        len_vecs.resize((num_overlaps, len(sents)))

        return sent_vecs, len_vecs
