import os
import unicodedata
from cltk.alphabet.text_normalization import cltk_normalize
from time import time
from cltk.stops.grc import STOPS as stops_list
import pickle 
import json
import random

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from database_analysis.config import *
from lemmatizers.ensemble import *

CLTK_APOSTROPHE = '’'
stops_tuple = tuple(stops_list)

def normalize_for_cltk_input(text: str) -> str:
    text = text.replace("ʼ", CLTK_APOSTROPHE) 
    text = text.replace("'", CLTK_APOSTROPHE)  
    text = text.replace("`", CLTK_APOSTROPHE)  
    text = text.replace("‘", CLTK_APOSTROPHE)  
    return cltk_normalize(text)

def normalize_for_morph(text):
    text = text.replace("'", "ʼ")
    
    oxia_to_tonos = {
        'ά': 'ά',  
        'έ': 'έ',
        'ή': 'ή',
        'ί': 'ί',
        'ό': 'ό',
        'ύ': 'ύ',
        'ώ': 'ώ',
        'ά': 'ά',  
        'έ': 'έ',
        'ή': 'ή',
        'ί': 'ί',
        'ό': 'ό',
        'ύ': 'ύ',
        'ώ': 'ώ',
    }
    for oxia, tonos in oxia_to_tonos.items():
        text = text.replace(oxia, tonos)
    
    return unicodedata.normalize('NFC', text).strip()
    

def normalize_for_comparison(text: str) -> str:
    if text is None:
        return None
    
    text = text.replace("ʼ", "'")
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("‘", "'")
    text = text.replace("ϑ", "θ")
    
    return text

# TODO: some of these no parses are genuinely not in the current Morpheus database. However, if we WEBSCRAPE from https://anastrophe.uchicago.edu/morpheus?word=%CF%86%CE%AC%CE%B3%CE%BF%CF%82, 
# a lemma seems to ALWAYS be returned. Is that a better method than consulting this database for as much accuracy as possible?
# the database just has Plato
# spreadsheet has "Spuria only" lemmas, produce a count of how many tokens that appear in the text file sent by Prof Rusten are also NOT parsed
def morpheus_lemmatizer(token):  
    query = "SELECT lemma FROM Lexicon WHERE token = ?"
    norm_token = normalize_for_morph(token)
    morph_cur.execute(query, (norm_token,))
    lemmas = morph_cur.fetchall()
    lemmata = []
    for lemma in lemmas:
        lemmata.append(lemma[0])
    return lemmata

# if the distinct query returns multiple probabilities and multiple lemmas
# for any one of those lemmas, if you have multiple sets of probabilities
# use the probability value pair 

# TODO: something of interest is how often a word has multiple lemmas in the first place
"""
Before solving below, just count how many times it happens.

How the full code should work
I have my token
Old morph outputs:
(
    (Lemma1, 0.83), (Lemma2, 0.17)
    (Lemma1, 0.2), (Lemma2, 0.8)
)

^ WHERE they collectively add up, e.g. lemma1 consists of lemma1.p1 at 0.4, lemma2.p2 at 0.33, lemma3.p3 at p1

Easiest case: if the first pair is the only one we get, just pick the highest probability
Case 2: Lemma1 is the max in all cases, just pick Lemma1 (i.e. lemma1 wins n times in n pairs)
Case 3: In cases where there are multiple probability pairs AND the winner of that comparison differs (i.e. a lemma DOES NOT win n times), select the most frequently
occurring pair. If opt1 appears 30 times and opt2 appears 10 times, go with option 1. If they appear the same amount of times, mark for human review

"""

"""
But a different case of σχῆμα, where they're all the same
Eventually all the pairs sum to 1 anyways, e.g. 0.98737 and 0.11723 or whatever.
The difference is in teh parse probabilities, which DOES NOT matter
So its just 1 for that lemma b/c we don't care what the actual parses are
"""

thuceurplat_db_multiple_lemmas = 0
def thuceurplat_db_lemmatizer(token):
    global uni_multiple_lemmas, thuceurplat_db_multiple_lemmas, morph_multiple_lemmas, unigram_prob_not_one, uni_thuc_disagreements, fallback_to_thuceurplat_db, fallback_to_morph, total_tokens
    token_query = """
    SELECT DISTINCT L.lemma, P.prob
    FROM Lexicon L JOIN parses P 
    ON L.lexid = P.lex
    WHERE L.token = ?
    """
    norm_token = normalize_for_morph(token)
    old_morph_cur.execute(token_query, (norm_token,))
    thuceurplat_db_out = old_morph_cur.fetchall()
    
    if not thuceurplat_db_out:
        return None, None 
    
    if any(prob != 1.0 for _, prob in thuceurplat_db_out):
        thuceurplat_db_out = [(lemma, prob) for lemma, prob in thuceurplat_db_out if prob != 1.0]
        if not thuceurplat_db_out:
            return None, None
    
    lemma_probs = defaultdict(float)
    for lemma, prob in thuceurplat_db_out:
        lemma_probs[lemma] += prob

    unique_lemmas = list(lemma_probs.keys())
    if len(unique_lemmas) > 1:
        thuceurplat_db_multiple_lemmas += 1

    total = sum(lemma_probs.values())
    normalized_probs = {}
    for lemma, prob in lemma_probs.items():
        normalized_probs[lemma] = prob / total 
    
    max_prob = max(normalized_probs.values())
    winners = []
    for lemma, prob in normalized_probs.items():
        if prob == max_prob:
            winners.append(lemma) 

    if len(winners) == 1:
        return winners[0], max_prob
    else:
        return None, None

TRAIN_SENTS = open_pickle("cltk_data/grc/model/grc_models_cltk/lemmata/backoff/greek_lemmatized_sents.pickle")
unigram_lem = EnsembleUnigramLemmatizer(
    TRAIN_SENTS, source="CLTK Sentence Training Data"
)
def unigram_lemmatizer(token, unigram_lem):
    token = normalize_for_cltk_input(token)
    lemma = unigram_lem.lemmatize([token])
    return lemma[0][1]

# TODO: get a count of which ones are detected by the ensemble
# also, produce the list of no parse tokens
def load_spuria_lemmas():
    with open("spuria_lemmatization/spuria_only.txt", "r") as f:
        lines = [line.rstrip() for line in f if line.rstrip() != ""]
        return lines
    
uni_multiple_lemmas = 0
thuceurplat_db_multiple_lemmas = 0
morph_multiple_lemmas = 0
unigram_prob_not_one = 0
uni_thuc_disagreements = 0
fallback_to_thuceurplat_db = 0
fallback_to_morph = 0
no_parse_lemmas = 0
total_tokens = 0
spuria_lemma_count = 0
morph_fallback_multiple = 0
SPURIA_LEMMAS = load_spuria_lemmas()
CURRENT_TEXT = None

list_all_no_parse = []

def corrected_np_cache():
    CACHE_FILE = "spuria_lemmatization/corrected_np.json"
    
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            return cache
    else:
        np_cache = {}
        with open("spuria_lemmatization/corrected_np.csv", "r") as f:
            for line in f:
                lst = line.strip().split(",")
                if lst:
                    token, parse, parser, notes, text, page, section = lst 
                    np_cache[f"{text}.{page}.{section}.{token}"] = parse 
                        
        
        with open(CACHE_FILE, "w") as f:
            json.dump(np_cache, f)
        
        return np_cache

def corrected_multiple_cache():
    CACHE_FILE = "spuria_lemmatization/corrected_multiples.json"
    
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            return cache
    else:
        np_cache = {}
        with open("spuria_lemmatization/spuria_multipleparse.csv", "r") as f:
            for line in f:
                lst = line.strip().split(",")
                if lst:
                    token, lemma = lst 
                    np_cache[token] = lemma 
                        
        
        with open(CACHE_FILE, "w") as f:
            json.dump(np_cache, f)
        
        return np_cache
    
def ensemble_lemmatizer(token, text, page, section):
    global uni_multiple_lemmas, thuceurplat_db_multiple_lemmas, morph_multiple_lemmas, unigram_prob_not_one, uni_thuc_disagreements
    global fallback_to_thuceurplat_db, fallback_to_morph, total_tokens, no_parse_lemmas, spuria_lemma_count, SPURIA_LEMMAS, CURRENT_TEXT, list_all_no_parse, list_ensemble_results
        
    results = {}
    total_tokens += 1  
    
    morph_lemmas = []
    
    unigram_out = unigram_lemmatizer(token, unigram_lem)
    results["unigram"] = unigram_out
    
    manual_cache = corrected_np_cache()
    multiple_cache = corrected_multiple_cache()
    
    if unigram_out and isinstance(unigram_out, list) and unigram_out[0]:
        lemma_dict = unigram_out[0]
        lemma_probs = list(lemma_dict.values())[0]
        for lemma, prob in lemma_probs:
            if lemma in SPURIA_LEMMAS:
                spuria_lemma_count += 1
        
        lemma_probs.sort(key=lambda x: x[1], reverse=True)
        top_uni_lemma, top_prob = lemma_probs[0]
        
        if len(lemma_probs) > 1:
            uni_multiple_lemmas += 1
        
        if top_prob != 1.0:
            unigram_prob_not_one += 1
        
    else:
        top_uni_lemma, top_prob = None, 0.0

    if top_uni_lemma is None:
        thuceurplat_db_lemma, thuceurplat_db_prob = thuceurplat_db_lemmatizer(token)
        results["thuceurplat_db"] = (thuceurplat_db_lemma, thuceurplat_db_prob)
        if thuceurplat_db_lemma in SPURIA_LEMMAS:
            spuria_lemma_count += 1

        morph_lemmas = list(set(morpheus_lemmatizer(token)))
        results["morpheus"] = morph_lemmas
        for lemma in morph_lemmas:
            if lemma in SPURIA_LEMMAS:
                spuria_lemma_count += 1
            

        if thuceurplat_db_lemma is None:
            if morph_lemmas == []:
                try:
                    manual_lemma = manual_cache[f"{text}.{page}.{section}.{token}"]
                    results["manual"] = (manual_lemma, 1.0)
                except KeyError:         
                    results["manual"] = None
                    list_all_no_parse.append((token, text, page, section))
                    no_parse_lemmas += 1
            else:
                results["manual"] = None
                if len(morph_lemmas) > 1:
                    if token in multiple_cache:
                        results["morpheus"] = (multiple_cache[token], 1.0)
                fallback_to_morph += 1
        else:
            fallback_to_thuceurplat_db += 1
    else:
        thuceurplat_db_lemma, thuceurplat_db_prob = thuceurplat_db_lemmatizer(token)
        results["thuceurplat_db"] = (thuceurplat_db_lemma, thuceurplat_db_prob)
        morph_lemmas = list(set(morpheus_lemmatizer(token)))
        results["morpheus"] = morph_lemmas
        results["manual"] = None

        top_uni_lemma = normalize_for_comparison(top_uni_lemma)
        thuceurplat_db_lemma = normalize_for_comparison(thuceurplat_db_lemma)
        if thuceurplat_db_lemma is not None and top_uni_lemma != thuceurplat_db_lemma:
            if not (re.search(r'\d+$', str(thuceurplat_db_lemma))):
                uni_thuc_disagreements += 1

    
    if isinstance(morph_lemmas, list) and len(morph_lemmas) > 1:
        morph_multiple_lemmas += 1
    
    if (
        unigram_out and isinstance(unigram_out, list) and len(unigram_out) > 0 and unigram_out[0]
    ) or (
        thuceurplat_db_lemma is not None
    ) or (
        isinstance(morph_lemmas, list) and len(morph_lemmas) > 0
    ):
        lemma_dict = unigram_out[0] if unigram_out and isinstance(unigram_out, list) and len(unigram_out) > 0 else {}
        lemma_probs = list(lemma_dict.values())[0] if lemma_dict else []

        list_ensemble_results.append([
            token,
            json.dumps(lemma_probs, ensure_ascii=False),
            json.dumps(results["thuceurplat_db"], ensure_ascii=False),
            json.dumps(results["morpheus"], ensure_ascii=False)
        ])

    return results

def top_lemma_from_results(ensemble_results):
    top_lemmas = {}
    
    unigram_out = ensemble_results.get("unigram", None)
    if unigram_out and isinstance(unigram_out, list) and unigram_out[0]:
        lemma_dict = unigram_out[0]
        lemma_probs = list(lemma_dict.values())[0]
        lemma_probs.sort(key=lambda x: x[1], reverse=True)
        top_uni_lemma, top_prob = lemma_probs[0]
        top_lemmas["Unigram"] = (top_uni_lemma, top_prob)
    else:
        top_lemmas["Unigram"] = (None, 0.0)
    
    thuceurplat_db_out = ensemble_results.get("thuceurplat_db", (None, None))
    top_lemmas["ThucEurPlat"] = thuceurplat_db_out
    
    morph_out = ensemble_results.get("morpheus", (None, None))
    top_lemmas["Morpheus"] = morph_out
    
    manual_lemma = ensemble_results.get("manual", (None, None))
    top_lemmas["Manual"] = manual_lemma
    
    return top_lemmas

def backoff_chain(top_lemmas):
    global morph_fallback_multiple
    if top_lemmas["Unigram"][0]:
        lemma, prob = top_lemmas["Unigram"]
        return lemma, prob, "unigram"
    else:
        if top_lemmas["ThucEurPlat"][0]:
            lemma, prob = top_lemmas["ThucEurPlat"]
            return lemma, prob, "thuceurplat"
        else: 
            # TODO: how to handle multiple Morpheus lemmas?
            if top_lemmas["Morpheus"]:
                # default: first one 
                lemmata = top_lemmas["Morpheus"]
                if len(lemmata) > 1:
                    morph_fallback_multiple += 1
                return lemmata[0], 1.0, "morpheus"
            else:
                if top_lemmas["Manual"]:
                    lemma, prob = top_lemmas["Manual"]
                    return lemma, prob, "manual"
                else:
                    return "unknown", 0.0, "none"
        
    

SPURIA_TEXTS = ['Ax', 'Def', 'DeIusto', 'Dem', 'Eryx', 'DeVirt', 'Sis']
# SPURIA_TEXTS = ['Ax']
list_ensemble_results = []
CURR_SEQ = 593456

def load_lemma_into_db(token_idx, token, text, lemma, prob, lemma_meaning="X"):
    global CURR_SEQ
    conn = sqlite3.connect("/Users/ayan/Desktop/Lexeis-Aristophanes/utilities/plato/input/CopyGreekMorphologyThucEurPlato.db")
    cur = conn.cursor()

    # insert into tokens if not already present
    cur.execute("SELECT tokenid FROM tokens WHERE tokenid = ?", (token_idx,))
    if not cur.fetchone():
        cur.execute("""
            INSERT INTO tokens (tokenid, content, seq, type, file)
            VALUES (?, ?, ?, ?, ?)""", (token_idx, token, CURR_SEQ, "undetermined", f"Plato{text}Gr.xml"))
        # TODO: correct way to handle seq: starts at 1 for the first token in the text, and increments by 1 for each token
        # TODO: handling type
        CURR_SEQ += 1
        
    # insert into Lexicon table if not already present
    cur.execute("SELECT lexid FROM Lexicon WHERE token = ? AND lemma = ?", (token, lemma))
    row = cur.fetchone()
    if row:
        # TODO: lemma matches but the token isnt the same as db
        lexid = row[0]
    else:
        cur.execute("INSERT INTO Lexicon (token, lemma, blesslemma, blesslex) VALUES (?, ?, 0, 0)", (token, lemma))
        # could bless later if needed
        lexid = cur.lastrowid

    # insert into parses table
    cur.execute("INSERT INTO parses (tokenid, lex, prob) VALUES (?, ?, ?)", (token_idx, lexid, prob))

    # update frequencies
    cur.execute("SELECT count FROM frequencies WHERE lemma = ?", (lemma,))
    freq = cur.fetchone()
    if freq:
        cur.execute("UPDATE frequencies SET count = count + 1 WHERE lemma = ?", (lemma,))
    else:
        cur.execute("INSERT INTO frequencies (lemma, rank, count, rate, lookupform) VALUES (?, ?, ?, ?, ?)", 
                    (lemma, -1, 1, -1.0, lemma))
        # leave rank/rate -1 for now

    #commit changes
    conn.commit()
    
def lemmatizer_eval():
    global CURRENT_TEXT, list_all_no_parse, list_ensemble_results
    
    conn = sqlite3.connect("/Users/ayan/Desktop/Lexeis-Aristophanes/utilities/plato/results/lexicon_database.db")
    cur = conn.cursor()
    
    full_tokens = []
    
    for text in SPURIA_TEXTS:
        cur.execute("""
            SELECT token_index, token, text, page, section
            FROM text_storage 
            WHERE text = ? AND token_index NOT IN (-1, -2)
        """, (text,))
        tokens_data = cur.fetchall()
        full_tokens.extend(tokens_data)
    
    random_fallback_old = []
    random_fallback_morph = []
    random_fallback_morph_multiple = []
    random_disagreements = []
    # TODO: we want the ENTIRE list of stuff that isn't present in the database
    # two files: one csv file with no parses, one csv file with everything that was lemmatized and what lemmas we got (just tokens and lemmas)
    # for now, put them into the DB as unknown
    no_parses = []
    total_tokens = len(full_tokens)
    
    print(f"\nBeginning evaluation...")
    print(f"Found {total_tokens} tokens to process.\n")
    
    ops = 0
    time_spent = 0
    
    for token_idx, token, text, page, section in full_tokens:
        start_time = time()
        
        ensemble_results = ensemble_lemmatizer(token, text, page, section)
        
        if fallback_to_thuceurplat_db > 0 and len(random_fallback_old) < 5 and not top_lemma_from_results(ensemble_results)["Unigram"][0]:
            if random.random() < 0.1:  
                thuceurplat_db_out = ensemble_results.get("thuceurplat_db", (None, None))
                random_fallback_old.append((token, thuceurplat_db_out))

        morph_out = ensemble_results.get("morpheus", [])
        
        if fallback_to_morph > 0 and len(random_fallback_morph) < 5 and not top_lemma_from_results(ensemble_results)["Unigram"][0] and not top_lemma_from_results(ensemble_results)["ThucEurPlat"][0]:
            if random.random() < 0.1:
                random_fallback_morph.append((token, morph_out))
        
        # if len(morph_out) > 1 and not top_lemma_from_results(ensemble_results)["Unigram"][0] and not top_lemma_from_results(ensemble_results)["ThucEurPlat"][0]:
        #     random_fallback_morph_multiple.append((token, morph_out))
                    

        unigram_out = ensemble_results.get("unigram", None)
        thuceurplat_db_out = ensemble_results.get("thuceurplat_db", (None, None))
        if unigram_out and thuceurplat_db_out[0] is not None:
            unigram_lemma = unigram_out[0][list(unigram_out[0].keys())[0]][0][0]
            if unigram_lemma != thuceurplat_db_out[0] and len(random_disagreements) < 5:
                if random.random() < 0.1 and not (re.search(r'\d+$', str(thuceurplat_db_out[0]))):
                    random_disagreements.append(
                        (token, f"Unigram: {unigram_lemma}", f"ThucEurPlat: {thuceurplat_db_out[0]}")
                    )
        
        # if no_parse_lemmas > 0:
        #     if ensemble_results.get("manual") == :
        #         # TODO: dashes in Sisyphus are asides, right? Treat as punctuation? ids: 4348464 4348480 4348722 4348731 4349114
        #         no_parses.append((normalize_for_morph(token), []))
                
        ops += 1
        time_spent += (time() - start_time)
        
        if ops % 500 == 0:
            avg_time = time_spent / ops
            eta = round(avg_time * (total_tokens - ops), 2)
            print(f"Evaluation currently on operation {ops} out of {total_tokens} with an estimated {eta} seconds left")
        
        top_lemmas = top_lemma_from_results(ensemble_results)
        # TODO: prob isnt used right now
        lemma, prob, source = backoff_chain(top_lemmas)
        load_lemma_into_db(token_idx, token, text, lemma, prob)
    
    total_time = round(time_spent, 2)
    print(f"\n--- Finished evaluation in {total_time} seconds. ---")
    
    print("----- METRICS SUMMARY -----")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Fallbacks to ThucEurPlat from the unigram lemmatizer: {fallback_to_thuceurplat_db}")
    print(f"Fallbacks to morpheus DB from the ThucEurPlat DB: {fallback_to_morph}")
    print(f"Unigram and ThucEurPlat disagreements: {uni_thuc_disagreements}")
    print(f"Unigram output contains multiple unique lemmas: {uni_multiple_lemmas}")
    print(f"ThucEurPlat output contains multiple unique lemmas: {thuceurplat_db_multiple_lemmas}")
    print(f"Morpheus output contains multiple unique lemmas: {morph_multiple_lemmas}")
    print(f"Fallback to a Morpheus output with multiple unique lemmas: {morph_fallback_multiple}")
    print(f"No parse detected by ensemble: {no_parse_lemmas}")
    print(f"Spuria lemma count: {spuria_lemma_count}")
    
    print("\n--- RANDOM EXAMPLES ---")
    
    print("\nFallback to Old Morph Examples:")
    for ex in random_fallback_old:
        print(ex)
    
    print("\nFallback to Morpheus Examples:")
    for ex in random_fallback_morph:
        print(ex)
    
    print("\nFallback to Morpheus Examples for which multiple unique lemmas are produced:")
    for ex in random_fallback_morph_multiple:
        print(ex)
    
    print("\nDisagreements (Unigram vs Old Morph):")
    for ex in random_disagreements:
        print(ex)
    
    print("\n No Parse Examples:")
    for ex in list_all_no_parse:
        print(ex)
        
    # write all no parses to csv file
    with open("spuria_lemmatization/spuria_no_parse.csv", "w") as f:
        f.write("token,text,page,section\n")
        for tok, text, page, section in list_all_no_parse:
            f.write(f"{tok}, {text}, {page}, {section}\n")
        f.close()
    
    # write all lemmatized to csv file
    with open("spuria_lemmatization/spuria_token_lemmas.csv", "w", encoding="utf-8") as f:
        f.write("token,unigram_out,thuceurplat_out,morpheus_out\n")
        for token, unigram_out, thuceurplat_out, morph_out in list_ensemble_results:
            f.write(f"{token},{unigram_out},{thuceurplat_out},{morph_out}\n")
        f.close()
    conn.close()


if __name__ == "__main__":
    lemmatizer_eval()
    # res = ensemble_lemmatizer("ἀποδέω", None, None, None)
    # top_lemmas = top_lemma_from_results(res)
    # print(res)
    # print(backoff_chain(top_lemmas))


