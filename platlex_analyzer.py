import random
import sqlite3
import unicodedata
from cltk.lemmatize.grc import GreekBackoffLemmatizer
from cltk.alphabet.text_normalization import cltk_normalize
from time import time
from cltk.stops.grc import STOPS as stops_list
import pickle 

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from lemmatizers.ensemble import *
from config import *

CLTK_APOSTROPHE = '’' 

def normalize_for_cltk_input(text: str) -> str:
    text = text.replace("ʼ", CLTK_APOSTROPHE) 
    text = text.replace("'", CLTK_APOSTROPHE)  
    text = text.replace("`", CLTK_APOSTROPHE)  
    text = text.replace("‘", CLTK_APOSTROPHE)  
    
    return cltk_normalize(text)

def normalize_for_comparison(text: str) -> str:
    text = text.replace("ʼ", "'")
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("‘", "'")
    
    return unicodedata.normalize('NFC', text).lower().strip()
    
def morpheus_lemmatizer(token):  
    morph_cur.execute(f'SELECT lemma FROM Lexicon WHERE token="{token}"')
    lemmas = morph_cur.fetchall()
    lemmata = []
    for lemma in lemmas:
        lemmata.append(lemma[0])
    return lemmata

backoff_lemmatizer = GreekBackoffLemmatizer(verbose=True)

def cltk_lemmatizer(token):
    token = cltk_normalize(token).replace('ʼ', '’')
    lemma = backoff_lemmatizer.lemmatize([token])
    return lemma[0][1]

def init_ensemble():
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
    
    lemmatizer_list = [dict_new, unigram_lem, regex_lem, dict_old]
    ensemble = EnsembleWrapper(lemmatizer_list)
    return ensemble, lemmatizer_list

ensemble, lemmatizer_list = init_ensemble()

def top_lemma_per_ensemble(token, lemmatizers):
    lemmatizer_names = {}
    
    for lemmatizer in lemmatizers:
        if "Greek Model" in str(lemmatizer):
            lemmatizer_names[str(lemmatizer)] = "Dict New"
        elif "CLTK Sentence Training Data" in str(lemmatizer):
            lemmatizer_names[str(lemmatizer)] = "Unigram"
        elif "CLTK Greek Regex Patterns" in str(lemmatizer):
            lemmatizer_names[str(lemmatizer)] = "Regexp"
        elif "Morpheus Lemmas" in str(lemmatizer):
            lemmatizer_names[str(lemmatizer)] = "Dict Old"
    
    results = {}
    for lemmatizer_obj in lemmatizers:
        lemmatizer_str = str(lemmatizer_obj)
        if lemmatizer_str in lemmatizer_names.keys():
            simplified_name = lemmatizer_names[lemmatizer_str]
            results[simplified_name] = ""
    
    lemmata_output = ensemble.lemmatize([token])

    if lemmata_output:
        _, ensemble_results_for_token = lemmata_output[0]

        for output_item in ensemble_results_for_token:
            for lemmatizer_instance, lemmas_list in output_item.items():
                simplified_name = lemmatizer_names.get(str(lemmatizer_instance), str(lemmatizer_instance))
                if lemmas_list:
                    sorted_lemmas = sorted(lemmas_list, key=lambda x: x[1], reverse=True)
                    top_lemma = sorted_lemmas[0][0]
                    results[simplified_name] = top_lemma
    return results
    
def check_lemmatizer(lemmatizer_id, token, db_lemma):
    if lemmatizer_id == "Backoff":
        cltk_input_token = normalize_for_cltk_input(token)
        
        raw_cltk_output = cltk_lemmatizer(cltk_input_token) 
        
        comparison_output = normalize_for_comparison(raw_cltk_output)
        comparison_db_lemma = normalize_for_comparison(db_lemma)
        
        if comparison_output == "":
            return raw_cltk_output, False, "NP"
        elif comparison_output == comparison_db_lemma:
            return raw_cltk_output, True, "C"
        else:
            return raw_cltk_output, False, "I"

    elif lemmatizer_id == "Morpheus":
        morph_outputs = morpheus_lemmatizer(token) 
        
        comparison_db_lemma = normalize_for_comparison(db_lemma)
        
        comparison_outputs = [normalize_for_comparison(lemma) for lemma in morph_outputs]

        if not comparison_outputs:
            return [], False, "NP"
        elif comparison_db_lemma in comparison_outputs:
            return morph_outputs, True, "C"
        else:
            return morph_outputs, False, "I"
        
def check_ensemble(token, db_lemma):
    out = {}
    cltk_input_token = normalize_for_cltk_input(token)
    comparison_db_lemma = normalize_for_comparison(db_lemma)
    
    ensemble_results = top_lemma_per_ensemble(cltk_input_token, lemmatizer_list)
    for k, v in ensemble_results.items():
        comparison_v = normalize_for_comparison(v)
        if comparison_v == "":
            out[k] = v, False, "NP"
        elif comparison_v == comparison_db_lemma:
            out[k] = v, True, "C"
        else:
            out[k] = v, False, "I"
    
    return out
            
def get_tokens():
    CACHE_FILE = "platlex_query_cache.pkl"

    if os.path.exists(CACHE_FILE):
        print(f"Found cache file '{CACHE_FILE}'. Loading data directly...")
        with open(CACHE_FILE, 'rb') as f:
            tokens = pickle.load(f)
        print("Data loaded from cache.")
    else:
        print("No cache file found. Running database query...")
        stops_tuple = tuple(stops_list)
        query = f"""
        SELECT
            I.token_index,
            I.lemma,
            T.token
        FROM instance_information AS I
        JOIN text_storage AS T ON I.token_index = T.token_index
        WHERE 
            I.lemma NOT REGEXP '[0-9]'
            AND T.token NOT IN {stops_tuple};
        """
        cursor.execute(query)
        tokens = cursor.fetchall()
        
        print("Query finished. Saving results to cache file for future runs...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(tokens, f)
        print(f"Results saved to '{CACHE_FILE}'.")

    return tokens

def id_to_token(id):
    cursor.execute(f'SELECT token FROM text_storage WHERE token_index={id}')
    token = cursor.fetchall()
    return token[0][0]
        
def calc_error(correct, incorrect, np, total):
    adjusted_total = total - np
    if adjusted_total == 0:
        hit_rate = 0.0
        miss_rate = 0.0
    else:
        hit_rate = 100 * round(correct / adjusted_total, 4)
        miss_rate = 100 * round(incorrect / adjusted_total, 4)
        
    if total == 0:
        fail_rate = 0.0
    else:
        fail_rate = 100 * round(np / total, 4)
    
    return hit_rate, miss_rate, fail_rate

def calc_ensemble_error(ensemble_stats, ensemble_np, total):
    ensemble_errors = {}
    for k, v in ensemble_stats.items():
        ensemble_errors[k] = calc_error(v[0], v[1], v[2], total)
    ensemble_fail = 100 * round(ensemble_np / total, 4)
    return ensemble_errors, ensemble_fail

randomly_incorr = []
randomly_np = []

def lemmatizer_eval(tokens, lemmatizer_type):
    correct = 0
    incorrect = 0
    np = 0
    ensemble_np = 0
    ensemble_stats = {}
    total = len(tokens)
          
    for lemmatizer in lemmatizer_list:
        if "Greek Model" in str(lemmatizer):
            ensemble_stats["Dict New"] = [0, 0, 0]
        elif "CLTK Sentence Training Data" in str(lemmatizer):
            ensemble_stats["Unigram"] = [0, 0, 0]
        elif "CLTK Greek Regex Patterns" in str(lemmatizer):
            ensemble_stats["Regexp"] = [0, 0, 0]
        elif "Morpheus Lemmas" in str(lemmatizer):
            ensemble_stats["Dict Old"] = [0, 0, 0]
            
    
    print(f"Beginning {lemmatizer_type} evaluation...")
    ops = 0
    time_spent = 0
    
    for (idx, lemma, token) in tokens:
        begin = time()
        random_prob = random.random()
        
        if lemmatizer_type == "Ensemble":
            res = check_ensemble(token, lemma)
            np_total = 0
            for k, v in res.items():
                out_data, found, status = v
            
                if found and status == "C":
                    ensemble_stats[k][0] += 1 
                else:
                    if status == "I":
                        ensemble_stats[k][1] += 1 
                    else: 
                        ensemble_stats[k][2] += 1  
                        np_total += 1
            if np_total == 4:
                ensemble_np += 1
                if random_prob < 0.0008:
                    randomly_np.append([token, lemma, res])
                
        else:  
            out_data, found, status = check_lemmatizer(lemmatizer_type, token, lemma)
            
            if found and status == "C":
                correct += 1
            else:
                if status == "I":
                    if lemmatizer_type == "Backoff" and random_prob < 0.0002:
                        randomly_incorr.append([token, lemma, out_data])
                    incorrect += 1
                else: 
                    np += 1
            
        ops += 1
        
        end = time()
        op_time = end - begin
        time_spent += op_time
        
        if ops > 0:
            avg_op_time = time_spent / ops
            eta = round(avg_op_time * (total - ops), 2)
        else:
            eta = 0
        
        if ops % 25000 == 0:
            print(f"{lemmatizer_type} evaluation currently on operation {ops} out of {total} with an estimated {eta} seconds left")
    
    end = time()
    print(f"Finished {lemmatizer_type} Eval in {end - begin:.2f} seconds!\n\n")
    
    if lemmatizer_type != "Ensemble":
        return calc_error(correct, incorrect, np, total)
    else:
        return calc_ensemble_error(ensemble_stats, ensemble_np, total)

if __name__ == "__main__":
    tokens = get_tokens()

    morph_hit, morph_miss, morph_fail = lemmatizer_eval(tokens, "Morpheus")
    cltk_hit, cltk_miss, cltk_fail = lemmatizer_eval(tokens, "Backoff")
    ensemble_rates, ensemble_fail = lemmatizer_eval(tokens, "Ensemble")


    print("----- MORPHEUS STATS -----")
    print(f"Correct rate over parseable tokens: {morph_hit}%")
    print(f"Incorrect rate over parseable tokens: {morph_miss}%")
    print(f"No parse rate over all tokens: {morph_fail}%")

    print("\n----- BACKOFF STATS -----")
    print(f"Correct rate over parseable tokens: {cltk_hit}%")
    print(f"Incorrect rate over parseable tokens: {cltk_miss}%")
    print(f"No parse rate over all tokens: {cltk_fail}%")
    print("Randomly incorrect sample:")
    print(randomly_incorr)

    for k, v in ensemble_rates.items():
        print(f"\n----- {k.upper()} STATS -----")
        print(f"Correct rate over parseable tokens: {v[0]}%")
        print(f"Incorrect rate over parseable tokens: {v[1]}%")
        print(f"No parse rate over all tokens: {v[2]}%")
        
    print ("\n----- ENSEMBLE FAILURE -----")
    print(f"No parse rate of all four lemmatizers over all tokens: {ensemble_fail}%")
    print("Randomly unparseable sample:")
    print(randomly_np)
    
    connection.close()
    morph_connect.close()

