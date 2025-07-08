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

def normalize_for_comparison(text: str) -> str:
    if text is None:
        return None
    
    text = text.replace("ʼ", "'")
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("‘", "'")
    text = text.replace("ϑ", "θ")
    
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
    
    return unicodedata.normalize('NFC', text).lower().strip()

def morpheus_lemmatizer(token):  
    query = "SELECT lemma FROM Lexicon WHERE token = ?"
    token = normalize_for_comparison(token)
    morph_cur.execute(query, (token,))
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

old_morph_multiple_lemmas = 0
def old_morpheus_lemmatizer(token):
    global uni_multiple_lemmas, old_morph_multiple_lemmas, morph_multiple_lemmas, unigram_prob_not_one, uni_thuc_disagreements, fallback_to_old_morph, fallback_to_morph, total_tokens
    token_query = """
    SELECT DISTINCT L.lemma, P.prob
    FROM Lexicon L JOIN parses P 
    ON L.lexid = P.lex
    WHERE L.token = ?
    """
    token = normalize_for_comparison(token)
    old_morph_cur.execute(token_query, (token,))
    
    if not old_morph_out:
        return None, None 
    
    if any(prob != 1.0 for _, prob in old_morph_out):
        old_morph_out = [(lemma, prob) for lemma, prob in old_morph_out if prob != 1.0]
        if not old_morph_out:
            return None, None
    
    lemma_probs = defaultdict(float)
    for lemma, prob in old_morph_out:
        lemma_probs[lemma] += prob

    unique_lemmas = list(lemma_probs.keys())
    if len(unique_lemmas) > 1:
        old_morph_multiple_lemmas += 1

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
        print(winners)
        return None, None

TRAIN_SENTS = open_pickle("cltk_data/grc/model/grc_models_cltk/lemmata/backoff/greek_lemmatized_sents.pickle")
unigram_lem = EnsembleUnigramLemmatizer(
    TRAIN_SENTS, source="CLTK Sentence Training Data"
)
def unigram_lemmatizer(token, unigram_lem):
    token = normalize_for_cltk_input(token)
    lemma = unigram_lem.lemmatize([token])
    return lemma[0][1]

uni_multiple_lemmas = 0
old_morph_multiple_lemmas = 0
morph_multiple_lemmas = 0
unigram_prob_not_one = 0
uni_thuc_disagreements = 0
fallback_to_old_morph = 0
fallback_to_morph = 0
total_tokens = 0


def ensemble_lemmatizer(token):
    global uni_multiple_lemmas, old_morph_multiple_lemmas, morph_multiple_lemmas, unigram_prob_not_one, uni_thuc_disagreements, fallback_to_old_morph, fallback_to_morph, total_tokens
    
    results = {}
    total_tokens += 1  
    
    morph_lemmas = []
    
    unigram_out = unigram_lemmatizer(token, unigram_lem)
    results["unigram"] = unigram_out
    
    if unigram_out and isinstance(unigram_out, list) and unigram_out[0]:
        lemma_dict = unigram_out[0]
        lemma_probs = list(lemma_dict.values())[0]
        
        lemma_probs.sort(key=lambda x: x[1], reverse=True)
        top_uni_lemma, top_prob = lemma_probs[0]
        
        if len(lemma_probs) > 1:
            uni_multiple_lemmas += 1
        
        if top_prob != 1.0:
            unigram_prob_not_one += 1
        
    else:
        top_uni_lemma, top_prob = None, 0.0

    if top_uni_lemma is None:
        fallback_to_old_morph += 1
        old_morph_lemma, old_morph_prob = old_morpheus_lemmatizer(token)
        results["old_morpheus"] = (old_morph_lemma, old_morph_prob)
        
        if old_morph_lemma is None:
            fallback_to_morph += 1
            morph_lemmas = morpheus_lemmatizer(token)
            results["morpheus"] = morph_lemmas
        else:
            results["morpheus"] = []
    else:
        old_morph_lemma, old_morph_prob = old_morpheus_lemmatizer(token)
        results["old_morpheus"] = (old_morph_lemma, old_morph_prob)
        morph_lemmas = morpheus_lemmatizer(token)
        results["morpheus"] = morph_lemmas
        
        top_uni_lemma = normalize_for_comparison(top_uni_lemma)
        old_morph_lemma = normalize_for_comparison(old_morph_lemma)
        if old_morph_lemma is not None and top_uni_lemma != old_morph_lemma:
            if not (re.search(r'\d+$', str(old_morph_lemma))):
                uni_thuc_disagreements += 1
    
    if isinstance(morph_lemmas, list) and len(set(morph_lemmas)) > 1:
        morph_multiple_lemmas += 1
    
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
    
    old_morph_out = ensemble_results.get("old_morpheus", (None, None))
    top_lemmas["Old Morph"] = old_morph_out
    
    morph_out = ensemble_results.get("morpheus", (None, None))
    top_lemmas["Morpheus"] = old_morph_out
    
    return top_lemmas

SPURIA_TEXTS = ['Ax', 'Def', 'DeIusto', 'Dem', 'Eryx', 'DeVirt', 'Sis']

def lemmatizer_eval():
    conn = sqlite3.connect("/Users/ayan/Desktop/Lexeis-Aristophanes/utilities/plato/results/lexicon_database.db")
    cur = conn.cursor()
    
    full_tokens = []
    
    for text in SPURIA_TEXTS:
        cur.execute("""
            SELECT token_index, token 
            FROM text_storage 
            WHERE text = ? AND token_index NOT IN (-1, -2)
        """, (text,))
        tokens_data = cur.fetchall()
        full_tokens.extend(tokens_data)
    
    random_fallback_old = []
    random_fallback_morph = []
    random_disagreements = []
    total_tokens = len(full_tokens)
    
    print(f"\nBeginning evaluation...")
    print(f"Found {total_tokens} tokens to process.\n")
    
    ops = 0
    time_spent = 0
    
    for token_idx, token in full_tokens:
        if ops > 50:
            break
        
        start_time = time()
        
        ensemble_results = ensemble_lemmatizer(token)
        
        if fallback_to_old_morph > 0 and len(random_fallback_old) < 5:
            if random.random() < 0.01:  
                old_morph_out = ensemble_results.get("old_morpheus", (None, None))
                random_fallback_old.append((token, old_morph_out))

        if fallback_to_morph > 0 and len(random_fallback_morph) < 5:
            if random.random() < 0.01:
                morph_out = ensemble_results.get("morpheus", [])
                random_fallback_morph.append((token, morph_out))

        unigram_out = ensemble_results.get("unigram", None)
        old_morph_out = ensemble_results.get("old_morpheus", (None, None))
        if unigram_out and old_morph_out[0] is not None:
            unigram_lemma = unigram_out[0][list(unigram_out[0].keys())[0]][0][0]
            if unigram_lemma != old_morph_out[0] and len(random_disagreements) < 5:
                if random.random() < 0.1 and not (re.search(r'\d+$', str(old_morph_out[0]))):
                    random_disagreements.append(
                        (token, f"Unigram: {unigram_lemma}", f"Old Morph: {old_morph_out[0]}")
                    )
        
        ops += 1
        time_spent += (time() - start_time)
        
        if ops % 500 == 0:
            avg_time = time_spent / ops
            eta = round(avg_time * (total_tokens - ops), 2)
            print(f"Evaluation currently on operation {ops} out of {total_tokens} with an estimated {eta} seconds left")
    
    total_time = round(time_spent, 2)
    print(f"\n--- Finished evaluation in {total_time} seconds. ---")
    
    print("----- METRICS SUMMARY -----")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Fallbacks to old morpheus: {fallback_to_old_morph}")
    print(f"Fallbacks to morpheus DB: {fallback_to_morph}")
    print(f"Unigram and old morph disagreements: {uni_thuc_disagreements}")
    print(f"Unigram probability not 1: {unigram_prob_not_one}")
    print(f"Unigram multiple lemmas: {uni_multiple_lemmas}")
    print(f"Old morph multiple lemmas: {old_morph_multiple_lemmas}")
    print(f"Morpheus multiple lemmas: {morph_multiple_lemmas}")
    
    print("\n--- RANDOM EXAMPLES ---")
    
    print("\nFallback to Old Morph Examples:")
    for ex in random_fallback_old:
        print(ex)
    
    print("\nFallback to Morpheus Examples:")
    for ex in random_fallback_morph:
        print(ex)
    
    print("\nDisagreements (Unigram vs Old Morph):")
    for ex in random_disagreements:
        print(ex)
    
    conn.close()


if __name__ == "__main__":
    old_morpheus_lemmatizer("κατά")
    # lemmatizer_eval()


