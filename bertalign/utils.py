import re

import pickle
import string
from os.path import join, dirname

from dotenv import load_dotenv
load_dotenv()

import nltk
import nltk.data

from nltk import PunktSentenceTokenizer
vi_sentence_tokenizer = None

nltk.download('punkt') #directory specified by env NLTK_DATA=
nltk.download('punkt_tab') 
en_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fr_sentence_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

def add_non_splitting_tokens_to_text(text):
    punctuation = {
		'!': '<exclamation>', '?': '<question>', ',': '<comma>', ';': '<semicolon>',
		':': '<colon>', '_': '<underscore>', '-': '<dash>',
		'(': '<open_paren>', ')': '<close_paren>',
		'[': '<open_bracket>', ']': '<close_bracket>'
	}

    pattern = re.compile(r'([' + re.escape(''.join(punctuation.keys())) + r'])(?=[' + re.escape(''.join(punctuation.keys())) + r'])')
    
    def replace_match(match):
        return punctuation[match.group(0)]
    
    replaced_text = pattern.sub(replace_match, text)
    return replaced_text

def remove_splitting_tokens_from_list_of_sentences(sentences):
	return [remove_splitting_tokens_from_text(sentence) for sentence in sentences]

def remove_splitting_tokens_from_text(text):
    punctuation = {
        '<exclamation>': '!', '<question>': '?', '<comma>': ',', '<semicolon>': ';',
        '<colon>': ':', '<underscore>': '_', '<dash>': '-',
        '<open_paren>': '(', '<close_paren>': ')',
        '<open_bracket>': '[', '<close_bracket>': ']'
    }
    pattern = re.compile('|'.join(map(re.escape, punctuation.keys())))
    
    def replace_match(match):
        return punctuation[match.group(0)]
    replaced_text = pattern.sub(replace_match, text)
    return replaced_text

def clean_text(text):
    clean_text = []
    text = text.strip()
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r'\s+', ' ', line)
            clean_text.append(line)
    return "\n".join(clean_text)

def split_sents(text, lang):
    lang = lang.replace('summarized_', '').lower()
    text = add_non_splitting_tokens_to_text(text)
    if lang == 'vietnamese':
        split_sentences = split_into_sentences_vi(text)
        split_sentences = remove_splitting_tokens_from_list_of_sentences(split_sentences)
        return split_sentences
    elif lang == 'english':
        split_sentences = split_into_sentences_en(text)
        split_sentences = remove_splitting_tokens_from_list_of_sentences(split_sentences)
        return split_sentences
    elif lang == 'french':
        split_sentences = split_into_sentences_fr(text)
        split_sentences = remove_splitting_tokens_from_list_of_sentences(split_sentences)
        return split_sentences
    else:
        raise Exception(f'The language {lang} is not suppored yet.')
    
#from https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
def split_into_sentences_en(text: str) -> list[str]:
    return en_sentence_tokenizer.tokenize(text)

#todo: see if there is a better way - chatgpt made this for me
def split_into_sentences_fr(text: str) -> list[str]:
    return fr_sentence_tokenizer.tokenize(text)

#from https://github.com/undertheseanlp/underthesea/blob/main/underthesea/pipeline/sent_tokenize/__init__.py, we don't need all of underthesea, just this. underthesea has a lot of dependencies
def _load_model():  
	global vi_sentence_tokenizer
	if vi_sentence_tokenizer is not None:
		return
	model_path = join(dirname(__file__), 'st_kiss-strunk-2006_2019_01_13.pkl')
	with open(model_path, 'rb') as fs:
		punkt_param = pickle.load(fs)

	punkt_param.sent_starters = {}
	abbrev_types = ['g.m.t', 'e.g', 'dr', 'dr', 'vs', "000", 'mr', 'mrs', 'prof', 'inc', 'tp', 'ts', 'ths',
				'th', 'vs', 'tp', 'k.l', 'a.w.a.k.e', 't', 'a.i', '</i', 'g.w',
				'ass',
				'u.n.c.l.e', 't.e.s.t', 'ths', 'd.c', 've…', 'ts', 'f.t', 'b.b', 'z.e', 's.g', 'm.p',
				'g.u.y',
				'l.c', 'g.i', 'j.f', 'r.r', 'v.i', 'm.h', 'a.s', 'bs', 'c.k', 'aug', 't.d.q', 'b…', 'ph',
				'j.k', 'e.l', 'o.t', 's.a']
	abbrev_types.extend(string.ascii_uppercase)
	for abbrev_type in abbrev_types:
		punkt_param.abbrev_types.add(abbrev_type)
	for abbrev_type in string.ascii_lowercase:
		punkt_param.abbrev_types.add(abbrev_type)
	vi_sentence_tokenizer = PunktSentenceTokenizer(punkt_param)


def split_into_sentences_vi(text: str) -> list[str]:
    global sent_tokenizer
    _load_model()
    sentences = vi_sentence_tokenizer.sentences_from_text(text)
    return sentences

def yield_overlaps(lines, num_overlaps):
    lines = [_preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in _layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2

def _layer(lines, num_overlaps, comb=' '):
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii:ii + num_overlaps]))
    return out
    
def _preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line

if __name__ == "__main__":
	text = "\"Test end with quote.\" Test 2"
	print(split_into_sentences_en(text))
    