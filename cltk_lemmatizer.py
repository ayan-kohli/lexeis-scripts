from cltk.lemmatize.grc import GreekBackoffLemmatizer
from cltk.alphabet.text_normalization import cltk_normalize
from lemmatizers.ensemble import *
import json
import re
from lemmatizers.html_parser import sections
import sqlite3

GREEK_OLD_MODEL = open_pickle("cltk_data/grc/model/grc_models_cltk/lemmata/backoff/greek_lemmata_cltk.pickle")
GREEK_NEW_MODEL = open_pickle("cltk_data/grc/model/grc_models_cltk/lemmata/backoff/greek_model.pickle")

dict_old = EnsembleDictLemmatizer(lemmas=GREEK_OLD_MODEL, source="Morpheus Lemmas")
dict_new = EnsembleDictLemmatizer(lemmas=GREEK_NEW_MODEL, source="Greek Model")

GREEK_SUB_PATTERNS = [("(ων)(ος|ι|να)$", r"ων")]

regex_lem = EnsembleRegexpLemmatizer(regexps=GREEK_SUB_PATTERNS, source="CLTK Greek Regex Patterns")

TRAIN_SENTS = open_pickle("cltk_data/grc/model/grc_models_cltk/lemmata/backoff/greek_lemmatized_sents.pickle")

unigram_lem = EnsembleUnigramLemmatizer(train=TRAIN_SENTS, source="CLTK Sentence Training Data")

class EnsembleWrapper(SequentialEnsembleLemmatizer):
    def __init__(self, backoff, verbose = False):
        self._taggers = backoff
        self.VERBOSE = verbose
    
    def choose_tag(self, tokens, index, history):
        return super().choose_tag(tokens, index, history)


lemmatizers = [dict_new, unigram_lem, regex_lem, dict_old]

ensemble = EnsembleWrapper(lemmatizers)

spuria_headers = ["onlawsocrates", "onvirtuesocrates", "dimodokos", "sisyphus", "eryxias", "axiochus"]
spuria_texts = sections

connection = sqlite3.connect("GreekLexicon20250422.sqlite")
cur = connection.cursor()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)

for i in range(len(spuria_headers)):
    header = spuria_headers[i]
    sec = spuria_texts[i]
    sec_toks = remove_punctuation(cltk_normalize(sec)).split()
    lemmata = ensemble.lemmatize(sec_toks)
    json_list = []
    comparison_list = []

    for i, (token, ensemble_results) in enumerate(lemmata):
        if token == " ":
            continue
        returned = {}

        for lemmatizer in lemmatizers:
            returned[id(lemmatizer)] = ""

        for output in ensemble_results:
            for lemmatizer, lemmas in output.items():
                out_string = f"{lemmatizer}: "
                for lemma, score in lemmas:
                    out_string += f"\n--- {lemma} (Confidence: {score})"
                returned[id(lemmatizer)] = out_string
        
        for i in range(len(lemmatizers)):
            lem = lemmatizers[i]
            if returned[id(lem)] == "":
                returned[id(lem)] = f"{i + 1}. " + f"{lem}: No output"
            else:
                returned[id(lem)] = f"{i + 1}. " + returned[id(lem)] 
        
        # morpheus lookup for the token to compare

        cur.execute(f'SELECT lemma FROM Lexicon WHERE token="{token}"')
        morph_lemmas = [row[0] for row in cur.fetchall()]

        cltk_output = {}
        for output in ensemble_results:
            for lemmatizer, lemmas in output.items():
                name = str(lemmatizer)
                cltk_output.setdefault(name, [])
                for lemma, score in lemmas:
                    cltk_output[name].append(f"{lemma} ({score:.2f})")

        comparison_list.append([
            token,
            {
                "morpheus": morph_lemmas,
                "cltk": cltk_output
            }
        ])

        json_list.append((token, list(returned.values())))

    filename = "lemmatizer_outputs/" + header + "_cltk" + ".json"

    with open(filename, "w") as f:
        f.write("[\n")
        for i in range(len(json_list)):
            kv = json_list[i]
            f.write(json.dumps(kv, ensure_ascii=False, indent=0))
            if i == len(json_list) - 1:
                f.write("\n")
            else:
                f.write(",\n")
        f.write("]")

    with open("lemmatizer_outputs/" + header + "_comparison.json", "w") as f:
        json.dump(comparison_list, f, ensure_ascii=False, indent=2)
        
