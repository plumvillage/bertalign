import re

from nltk import PunktSentenceTokenizer
vi_sentence_tokenizer = None

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
    if lang in LANG.SPLITTER:
        if lang == 'vi':
            return split_into_sentences_vi(text)
        elif lang == 'en':
            return split_into_sentences_en(text)
        elif lang == 'fr':
            return split_into_sentences_fr(text)
        else:
            raise Exception('The language {} is not suppored yet.'.format(LANG.ISO[lang]))
    	
    else:
        raise Exception('The language {} is not suppored yet.'.format(LANG.ISO[lang]))
    
#from https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
def split_into_sentences_en(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    alphabets= "([A-Za-z])"
    prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
    starters = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov|edu|me)"
    digits = r"([0-9])"
    multiple_dots = r'\.{2,}'
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

#todo: see if there is a better way - chatgpt made this for me
def split_into_sentences_fr(text: str) -> list[str]:
    """
    Split the text into sentences in French.

    This function accounts for common French abbreviations, honorifics, 
    and sentence-ending punctuation to ensure proper sentence splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    alphabets = "([A-Za-zÀ-ÿ])"
    prefixes = r"(M|Mme|Mlle|Dr|Pr|St)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
    starters = r"(Il|Elle|Ils|Elles|On|Nous|Vous|Ce|Cela|Mais|Cependant|Toutefois|Car|Puis|Donc|Or|Ainsi|D'ailleurs)"
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov|edu|me|fr)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, r"\1<prd>", text)
    text = re.sub(websites, r"<prd>\1", text)
    text = re.sub(digits + r"[.]" + digits, r"\1<prd>\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    
    text = re.sub(r"\s" + alphabets + r"[.] ", r" \1<prd> ", text)
    text = re.sub(acronyms + r" " + starters, r"\1<stop> \2", text)
    text = re.sub(alphabets + r"[.]" + alphabets + r"[.]" + alphabets + r"[.]", r"\1<prd>\2<prd>\3<prd>", text)
    text = re.sub(alphabets + r"[.]" + alphabets + r"[.]", r"\1<prd>\2<prd>", text)
    text = re.sub(r" " + suffixes + r"[.] " + starters, r" \1<stop> \2", text)
    text = re.sub(r" " + suffixes + r"[.]", r" \1<prd>", text)
    text = re.sub(r" " + alphabets + r"[.]", r" \1<prd>", text)
    
    # Handle quotes and punctuation
    text = text.replace(".”", "”.")
    text = text.replace(".\"", "\".")
    text = text.replace("!\"", "\"!")
    text = text.replace("?\"", "\"?")
    
    # Mark sentence endings
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


#from https://github.com/undertheseanlp/underthesea/blob/main/underthesea/pipeline/sent_tokenize/__init__.py, we don't need all of underthesea, just this. underthesea has a lot of dependencies
def _load_model():  
	global sentence_tokenizer
	if sentence_tokenizer is not None:
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
	sentence_tokenizer = PunktSentenceTokenizer(punkt_param)


def split_into_sentences_vi(text: str) -> list[str]:
    global sent_tokenizer
    _load_model()
    sentences = sentence_tokenizer.sentences_from_text(text)
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
    
class LANG:
    SPLITTER = {
        'en': 'English',
        'fr': 'French',
        'vi': 'Vietnamese',
    }
    ISO = {
		'aa': 'Afar',
		'ab': 'Abkhaz',
		'af': 'Afrikaans',
		'ak': 'Akan',
		'am': 'Amharic',
		'an': 'Aragonese',
		'ar': 'Arabic',
		'as': 'Assamese',
		'av': 'Avaric',
		'ay': 'Aymara',
		'az': 'Azerbaijani',
		'ba': 'Bashkir',
		'be': 'Belarusian',
		'bg': 'Bulgarian',
		'bh': 'Bihari',
		'bi': 'Bislama',
		'bm': 'Bambara',
		'bn': 'Bengali',
		'bo': 'Tibetan',
		'br': 'Breton',
		'bs': 'Bosnian',
		'ca': 'Catalan',
		'ce': 'Chechen',
		'ch': 'Chamorro',
		'co': 'Corsican',
		'cr': 'Cree',
		'cs': 'Czech',
		'cv': 'Chuvash',
		'cy': 'Welsh',
		'da': 'Danish',
		'de': 'German',
		'dv': 'Divehi',
		'dz': 'Dzongkha',
		'ee': 'Ewe',
		'el': 'Greek',
		'en': 'English',
		'es': 'Spanish',
		'et': 'Estonian',
		'eu': 'Basque',
		'fa': 'Persian',
		'ff': 'Fula',
		'fi': 'Finnish',
		'fj': 'Fijian',
		'fo': 'Faroese',
		'fr': 'French',
		'fy': 'Western Frisian',
		'ga': 'Irish',
		'gd': 'Scottish Gaelic',
		'gl': 'Galician',
		'gn': 'Guaraní',
		'gu': 'Gujarati',
		'gv': 'Manx',
		'ha': 'Hausa',
		'he': 'Hebrew',
		'hi': 'Hindi',
		'ho': 'Hiri Motu',
		'hr': 'Croatian',
		'ht': 'Haitian',
		'hu': 'Hungarian',
		'hy': 'Armenian',
		'hz': 'Herero',
		'id': 'Indonesian',
		'ig': 'Igbo',
		'ii': 'Nuosu',
		'ik': 'Inupiaq',
		'io': 'Ido',
		'is': 'Icelandic',
		'it': 'Italian',
		'iu': 'Inuktitut',
		'ja': 'Japanese',
		'jv': 'Javanese',
		'ka': 'Georgian',
		'kg': 'Kongo',
		'ki': 'Kikuyu',
		'kj': 'Kwanyama',
		'kk': 'Kazakh',
		'kl': 'Kalaallisut',
		'km': 'Khmer',
		'kn': 'Kannada',
		'ko': 'Korean',
		'kr': 'Kanuri',
		'ks': 'Kashmiri',
		'ku': 'Kurdish',
		'kv': 'Komi',
		'kw': 'Cornish',
		'ky': 'Kyrgyz',
		'lb': 'Luxembourgish',
		'lg': 'Ganda',
		'li': 'Limburgish',
		'ln': 'Lingala',
		'lo': 'Lao',
		'lt': 'Lithuanian',
		'lu': 'Luba-Katanga',
		'lv': 'Latvian',
		'mg': 'Malagasy',
		'mh': 'Marshallese',
		'mi': 'Māori',
		'mk': 'Macedonian',
		'ml': 'Malayalam',
		'mn': 'Mongolian',
		'mr': 'Marathi',
		'ms': 'Malay',
		'mt': 'Maltese',
		'my': 'Burmese',
		'na': 'Nauru',
		'nb': 'Norwegian Bokmål',
		'nd': 'North Ndebele',
		'ne': 'Nepali',
		'ng': 'Ndonga',
		'nl': 'Dutch',
		'nn': 'Norwegian Nynorsk',
		'no': 'Norwegian',
		'nr': 'South Ndebele',
		'nv': 'Navajo',
		'ny': 'Chichewa',
		'oc': 'Occitan',
		'oj': 'Ojibwe',
		'om': 'Oromo',
		'or': 'Oriya',
		'os': 'Ossetian',
		'pa': 'Panjabi',
		'pl': 'Polish',
		'ps': 'Pashto',
		'pt': 'Portuguese',
		'qu': 'Quechua',
		'rm': 'Romansh',
		'rn': 'Kirundi',
		'ro': 'Romanian',
		'ru': 'Russian',
		'rw': 'Kinyarwanda',
		'sa': 'Sanskrit',
		'sc': 'Sardinian',
		'sd': 'Sindhi',
		'se': 'Northern Sami',
		'sg': 'Sango',
		'si': 'Sinhala',
		'sk': 'Slovak',
		'sl': 'Slovenian',
		'sm': 'Samoan',
		'sn': 'Shona',
		'so': 'Somali',
		'sq': 'Albanian',
		'sr': 'Serbian',
		'ss': 'Swati',
		'st': 'Southern Sotho',
		'su': 'Sundanese',
		'sv': 'Swedish',
		'sw': 'Swahili',
		'ta': 'Tamil',
		'te': 'Telugu',
		'tg': 'Tajik',
		'th': 'Thai',
		'ti': 'Tigrinya',
		'tk': 'Turkmen',
		'tl': 'Tagalog',
		'tn': 'Tswana',
		'to': 'Tonga',
		'tr': 'Turkish',
		'ts': 'Tsonga',
		'tt': 'Tatar',
		'tw': 'Twi',
		'ty': 'Tahitian',
		'ug': 'Uighur',
		'uk': 'Ukrainian',
		'ur': 'Urdu',
		'uz': 'Uzbek',
		've': 'Venda',
		'vi': 'Vietnamese',
		'wa': 'Walloon',
		'wo': 'Wolof',
		'xh': 'Xhosa',
		'yi': 'Yiddish',
		'yo': 'Yoruba',
		'za': 'Zhuang',
		'zh': 'Chinese',
		'zu': 'Zulu',
    }
