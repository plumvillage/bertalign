import os
import requests
import json
from typing import List, Dict, Any, Union

EMBEDDING_API_BASIC_AUTH_USERNAME = os.getenv("EMBEDDING_API_BASIC_AUTH_USERNAME")
EMBEDDING_API_BASIC_AUTH_PASSWORD = os.getenv("EMBEDDING_API_BASIC_AUTH_PASSWORD")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")

def get_embeddings(texts: Union[List[str], str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts by calling the embedding API.
    
    Args:
        texts: A list of strings or a single string to get embeddings for
        
    Returns:
        A list of embeddings (each embedding is a list of floats)
        
    Raises:
        Exception: If the API call fails
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    # Prepare the request
    headers = {"Content-Type": "application/json"}
    data = {
        "input": texts,
        "model": "sentence-transformers/LaBSE"  # Default model used by the server
    }
    
    # Set up authentication if credentials are provided
    auth = None
    if EMBEDDING_API_BASIC_AUTH_USERNAME and EMBEDDING_API_BASIC_AUTH_PASSWORD:
        auth = (EMBEDDING_API_BASIC_AUTH_USERNAME, EMBEDDING_API_BASIC_AUTH_PASSWORD)
    else:
        raise Exception("EMBEDDING_API_BASIC_AUTH_USERNAME and EMBEDDING_API_BASIC_AUTH_PASSWORD must be set")
    
    # Make the API call
    try:
        response = requests.post(
            f"{EMBEDDING_API_URL}/v1/embeddings",
            headers=headers,
            json=data,
            auth=auth,
            timeout=30
        )
        
        # Raise an exception for HTTP errors
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract embeddings from the response
        embeddings = [item["embedding"] for item in result["data"]]
        
        return embeddings
    except requests.exceptions.RequestException as e:
        if response is not None and hasattr(response, "content"):
            print(response.content)
        raise Exception(f"Failed to get embeddings: {str(e)}")
    