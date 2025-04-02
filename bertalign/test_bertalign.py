import sys
import os
from aligner import Bertalign
from utils import split_sents
import random

from dotenv import load_dotenv
load_dotenv()

aligner_path = os.path.abspath('../../RETAS-aligner')
print(aligner_path)
sys.path.append(aligner_path)

from generate_embeddings_align_debug_visualization import generate_beads_visualization_html

src_text_path = "C:/Users/Jonas/Downloads/Old Path White Clouds-Walking in the Footsteps of the Buddha -- Thich Nhat Hanh -- 1st, 1991 -- b1e49eb2673a8a2349d9ede542253926 -- Anna’s Archive.txt"
with open(src_text_path, 'r', encoding='utf-8') as file:
    src_text = file.read()

tgt_text_path = "H:/whisper_finetune_2/whisper-finetune-audiobooks/[done 2] Đường Xưa Mây Trắng/Đường-Xưa-Mây-Trắng-Thích-Nhất-Hạnh_textonly.txt"
with open(tgt_text_path, 'r', encoding='utf-8') as file:
    tgt_text = file.read()

aligner = Bertalign(
    src=src_text,
    tgt=tgt_text,
    src_lang="english",
    tgt_lang="vietnamese",
    is_split=False
)

aligner.align_sents()

beads = aligner.result

html = generate_beads_visualization_html(beads, aligner.src_sents, aligner.tgt_sents)
with open("beads.html", "w", encoding="utf-8") as file:
    file.write(html)

def generate_translation_jsonl(beads, src_sents, tgt_sents, batch_size=10, output_file="translation_data.jsonl"):
    """
    Generate a JSONL file for translation data from alignment beads.
    
    Args:
        beads: List of alignment beads (src_idx, tgt_idx)
        src_sents: Source sentences (English)
        tgt_sents: Target sentences (Vietnamese)
        batch_size: Number of sentences to batch together
        output_file: Output JSONL file path
    """
    import json
    
    # Fixed system prompt
    system_prompt = """
You will be provided the text of a Dharma Talk or lecture given by Thich Nhat Hanh in Vietnamese. 
The task is to translate it into English. You should translate it in the style of Thich Nhat Hanh, and use appropriate terms, words, and metaphors found in Thich Nhat Hanh's writings, teachings and books, where appropriate.
Avoid old-school buddhist terms if you know there is a term that Thich Nhat Hanh would have used, and allow yourself to be creative in your translation, imagining how Thich Nhat Hanh would choose to express the concept in English. Do your best to convey the meaning of the text in English, as if you were Thich Nhat Hanh speaking, instead of always translating literally. Some examples:

1. Thich Nhat Hanh uses Thầy to refer to himself in the third person, you should preserve this in your translation; for example by translating "Thầy nói" -> "Thầy said", not "The teacher said".

Only translate the provided text, do not add any continuation, clarification, or comments. 

Please try to translate the entire text that is presented."""
    
    batches = []

    current_src_batch = []
    current_tgt_batch = []
    
    for bead in beads:
        src_indexes, tgt_indexes = bead
        if len(src_indexes) == 0 or len(tgt_indexes) == 0:
            continue

        for index in src_indexes:
                current_src_batch.append(src_sents[index])
            
        for index in tgt_indexes:
                current_tgt_batch.append(tgt_sents[index])
        
        # When we reach batch_size, create a new entry, only at a point where both sides have content
        if len(current_src_batch) >= batch_size:
            # Only add if both sides have content
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": " ".join(current_tgt_batch).replace("\n", " ")
                    },
                    {
                        "role": "assistant",
                        "content": " ".join(current_src_batch).replace("\n", " ")
                    }
                ]
            }
            batches.append(entry)
            
            # Reset batches
            current_src_batch = []
            current_tgt_batch = []
    
    # Add the final batch if it's not empty
    if current_src_batch and current_tgt_batch:
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": " ".join(current_tgt_batch).replace("\n", " ")
                },
                {
                    "role": "assistant",
                    "content": " ".join(current_src_batch).replace("\n", " ")
                }
            ]
        }
        batches.append(entry)

    random.shuffle(batches)
    train_batches = batches[:int(len(batches) * 0.8)]
    test_batches = batches[int(len(batches) * 0.8):]
    
    # Write to JSONL file
    with open("train_batches.jsonl", 'w', encoding='utf-8') as f:
        for entry in train_batches:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    with open("test_batches.jsonl", 'w', encoding='utf-8') as f:
        for entry in test_batches:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(batches)} translation pairs in {output_file}")

# Call the function to generate the JSONL file
generate_translation_jsonl(beads, aligner.src_sents, aligner.tgt_sents)

