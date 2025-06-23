import sqlite3
from cltk.alphabet.text_normalization import cltk_normalize
import re
import json
from lemmatizers.html_parser import sections

# future : remove duplicates

connection = sqlite3.connect("morpheus_data/GreekLexicon20250422.sqlite")
cur = connection.cursor()

spuria_headers = ["onlawsocrates", "onvirtuesocrates", "dimodokos", "sisyphus", "eryxias", "axiochus"]
spuria_texts = sections

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)

for i in range(len(spuria_headers)):
    header = spuria_headers[i]
    sec = spuria_texts[i]
    sec_toks = remove_punctuation(cltk_normalize(sec)).split()
    json_list = []

    for token in sec_toks:
        cur.execute(f'SELECT lemma FROM Lexicon WHERE token="{token}"')
        lemmas = cur.fetchall()
        lemmata = []
        for lemma in lemmas:
            lemmata.append(lemma[0])
        json_list.append((token, lemmata))

    filename = "lemmatizer_outputs/" + header + "_morph" + ".json"

    with open(filename, "w") as f:
        f.write("[\n")
        for i in range(len(json_list)):
            kv = json_list[i]
            f.write(json.dumps(kv, ensure_ascii=False))
            if i == len(json_list) - 1:
                f.write("\n")
            else:
                f.write(",\n")
        f.write("]")

