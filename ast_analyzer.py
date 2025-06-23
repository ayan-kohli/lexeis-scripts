import os
import dotenv
import json
import re
import unicodedata
from config import *
from cltk.lemmatize.grc import GreekBackoffLemmatizer
from cltk.alphabet.text_normalization import cltk_normalize
from time import time
from collections import Counter
import random
import pickle

# TODO: for Ast: Words where his only citations are the Spurious texts shld be highlighted

dotenv.load_dotenv()
DB_PASS = os.getenv("DB_PASS")
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
    query = "SELECT lemma FROM Lexicon WHERE token = ?"
    morph_cur.execute(query, (token,))
    lemmas = morph_cur.fetchall()
    lemmata = []
    for lemma in lemmas:
        lemmata.append(lemma[0])
    return lemmata

backoff_lemmatizer = GreekBackoffLemmatizer(verbose=False)

def cltk_lemmatizer(token):
    token = normalize_for_cltk_input(token)
    lemma = backoff_lemmatizer.lemmatize([token])
    return lemma[0][1]

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

SHORT_NAMES = {
    "Alcibiades1": "Alc1", "Alcibiades2": "Alc2", "Apology": "Ap",
    "Charmides": "Charm", "Cleitophon": "Clit", "Cratylus": "Cra",
    "Critias": "Criti", "Crito": "Cri", "Epinomis": "Epin",
    "Euthydemus": "Euthd", "Euthyphro": "Euthphr", "Gorgias": "Grg",
    "Hipparchus": "Hipparch", "HippiasMajor": "HpMai", "HippiasMinor": "HpMi",
    "Ion": "Ion", "Laches": "La", "Laws": "Leg",
    "Letters": "Ep", "Lovers": "Amat", "Lysis": "Ly",
    "Menexenus": "Menex", "Meno": "Men", "Minos": "Min",
    "Parmenides": "Prm", "Phaedo": "Phd", "Phaedrus": "Phdr",
    "Philebus": "Phlb", "Protagoras": "Prt", "Republic": "Resp",
    "Sophist": "Soph", "Statesman": "Plt", "Symposium": "Symp",
    "Theaetetus": "Tht", "Theages": "Thg", "Timaeus": "Ti",
}

def parse_citation_string(citation):
    match = re.match(r'^([A-Za-z]+[0-9]?)\.?\s*(\d+[a-z]?)\.?\s*([a-z])$', citation.strip())
    if match:
        return match.groups()
    return None

def build_ast_citation_cache(cursor):
    CACHE_FILE = "ast_cache.pkl"

    if os.path.exists(CACHE_FILE):
        print(f"Found cache file {CACHE_FILE}. Loading citations directly...")
        with open(CACHE_FILE, 'rb') as f:
            ast_cache = pickle.load(f)
        print("Data loaded from cache.")
    else:
        print("No cache file found. Building citation cache...")
        ast_cache = {}
        cursor.execute("""
                    SELECT L.lemmaid, LD.long_def 
                    FROM lemmata L JOIN long_definitions LD 
                    ON L.lemmaid = LD.lemmaid
                    """)
        
        for ast_lemma, long_def_json in cursor.fetchall():
            if not long_def_json:
                continue
            
            all_refs = []
            try:
                data = json.loads(long_def_json)
                if not isinstance(data, list): 
                    continue
                
                for sub_entry in data:
                    def find_refs(element):
                        if isinstance(element, dict):
                            if "refList" in element:
                                for ref_item in element.get("refList", []):
                                    if "ref" in ref_item:
                                        all_refs.append(ref_item["ref"])
                            for value in element.values():
                                find_refs(value)
                        elif isinstance(element, list):
                            for item in element:
                                find_refs(item)
                                
                    find_refs(sub_entry)
            except (json.JSONDecodeError, TypeError):
                continue

            if all_refs:
                ast_cache[ast_lemma] = Counter(all_refs)
        print(f"Cache built with citation data for {len(ast_cache)} lemmas.")
        
        print("Cache finished. Saving results to cache file for future runs...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(ast_cache, f)
        print(f"Results saved to {CACHE_FILE}.")
        
    return ast_cache

def extract_quotes_from_json(json_str):
    quotes_by_ref = {}
    try:
        data = json.loads(json_str)
        if not isinstance(data, list): 
            return {}

        def find_all_refs_and_quotes(element):
            if isinstance(element, dict):
                if "refList" in element:
                    for ref_item in element.get("refList", []):
                        ref = ref_item.get("ref")
                        note = ref_item.get("note")
                        if ref and note:
                            clean_quote = ''
                            for i, char in enumerate(note):
                                if '\u0370' <= char <= '\u03FF':
                                    clean_quote = note[i:].strip()
                                    break
                            if clean_quote:
                                quotes_by_ref[ref] = clean_quote
                for value in element.values():
                    find_all_refs_and_quotes(value)
            elif isinstance(element, list):
                for item in element:
                    find_all_refs_and_quotes(item)
        
        find_all_refs_and_quotes(data)
            
    except (json.JSONDecodeError, TypeError):
        pass
    return quotes_by_ref

def get_token_context(cursor, token_index, window=5):
    query_initial = "SELECT dialogue, page, section, sequence_index FROM text_storage WHERE token_index = %s"
    cursor.execute(query_initial, (token_index,))
    result = cursor.fetchone()
    if not result:
        return None

    dialogue, page, section, seq_idx = result
    
    start_seq = max(0, seq_idx - window)
    end_seq = seq_idx + window

    query_context = """
        SELECT token FROM text_storage 
        WHERE dialogue = %s AND page = %s AND section = %s 
        AND sequence_index BETWEEN %s AND %s 
        ORDER BY sequence_index
    """
    cursor.execute(query_context, (dialogue, page, section, start_seq, end_seq))
    
    context_words = [row[0] for row in cursor.fetchall()]
    return " ".join(context_words)

def quote_match(quote, context):
    if not quote or not context:
        return False
        
    norm_quote = normalize_for_comparison(quote)
    norm_context = normalize_for_comparison(context)

    fragments = [frag.strip() for frag in norm_quote.split('...') if frag.strip()]

    if not fragments:
        return False

    last_idx = -1
    for fragment in fragments:
        curr_idx = norm_context.find(fragment, last_idx + 1)
        if curr_idx == -1:
            return False 
        last_idx = curr_idx
        
    return True 

def lemmatizer_evaluation_runner():
    print("Establishing connections...")
    print("Connections established!")

    ast_citation_cache = build_ast_citation_cache(cursor)
    
    CACHE_FILE = "query_cache.pkl"

    if os.path.exists(CACHE_FILE):
        print(f"Found cache file '{CACHE_FILE}'. Loading data directly...")
        with open(CACHE_FILE, 'rb') as f:
            locations_to_check = pickle.load(f)
        print("Data loaded from cache.")
    else:
        print("No cache file found. Running database query...")
        query = """
            SELECT
                I.lemma,
                L.lemmaid,
                T.dialogue,
                T.page,
                T.section
            FROM instance_information AS I
            JOIN text_storage AS T ON I.token_index = T.token_index
            JOIN lemmata AS L ON I.lemma = L.lemma
            WHERE LENGTH(T.token) > 1
            GROUP BY I.lemma, L.lemmaid, T.dialogue, T.page, T.section;
        """
        cursor.execute(query)
        locations_to_check = cursor.fetchall()
        
        print("Query finished. Saving results to cache file for future runs...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(locations_to_check, f)
        print(f"Results saved to '{CACHE_FILE}'.")
    
    total = len(locations_to_check)
    print(f"Found {total} unique lemma/location pairs to check.")
    print(f"Beginning combined evaluation...")
    
    stats = { "Morpheus": {"C": 0, "I": 0, "NP": 0}, "Backoff": {"C": 0, "I": 0, "NP": 0} }
    quote_verified_examples = []
    verified_tokens_evaluated = 0                           
    begin = time()
    ops = 0

    for i, (db_lemma, lemmaid, dialogue, page, section) in enumerate(locations_to_check):
        ops += 1
        if ops % 25000 == 0:
            elapsed_time = time() - begin
            avg_op_time = elapsed_time / ops
            eta = round(avg_op_time * (total - ops), 2)
            print(f"Evaluation currently on operation {ops} out of {total} with an estimated {eta} seconds left")
        
        lemma_count_query = """
            SELECT COUNT(*) FROM instance_information I 
            JOIN text_storage T ON I.token_index = T.token_index 
            WHERE I.lemma = %s AND T.dialogue = %s AND T.page = %s AND T.section = %s
        """
        cursor.execute(lemma_count_query, (db_lemma, dialogue, page, section))
        lemma_occurrence_count = cursor.fetchone()[0]
        
        ast_lemma_citations = ast_citation_cache.get(lemmaid)
        if not ast_lemma_citations: 
            continue
            
        short_name = SHORT_NAMES.get(dialogue, dialogue)
        cite_key = f"{short_name}.{page}.{section}"
        ast_citation_count = ast_lemma_citations.get(cite_key, 0)
        
        is_verified = lemma_occurrence_count == ast_citation_count and ast_citation_count > 0
        
        if is_verified:
            all_tokens_query = """
            SELECT T.token FROM instance_information I JOIN text_storage T 
            ON I.token_index = T.token_index
            WHERE I.lemma = %s AND T.dialogue = %s AND T.page = %s AND T.section = %s
            """
            cursor.execute(all_tokens_query, (db_lemma, dialogue, page, section))
            
            tokens_in_sec = Counter([row[0] for row in cursor.fetchall()])
            
            for token_form, count in tokens_in_sec.items():
                verified_tokens_evaluated += count
                morph_out, morph_found, status = check_lemmatizer("Morpheus", token_form, db_lemma)
                stats["Morpheus"][status] += count
                cltk_out, cltk_found, status = check_lemmatizer("Backoff", token_form, db_lemma)
                stats["Backoff"][status] += count
        
        else: 
            cursor.execute("SELECT long_def FROM long_definitions WHERE lemmaid = %s", (lemmaid,))
            result = cursor.fetchone()
            if not result or not result[0]: continue
            
            quotes_by_ref = extract_quotes_from_json(result[0])
            quote_text = quotes_by_ref.get(cite_key)

            if quote_text:
                all_lemma_instances_query = """
                    SELECT T.token, T.token_index FROM instance_information I
                    JOIN text_storage T ON I.token_index = T.token_index
                    WHERE I.lemma = %s AND T.dialogue = %s AND T.page = %s AND T.section = %s
                """
                cursor.execute(all_lemma_instances_query, (db_lemma, dialogue, page, section))
                candidate_tokens = cursor.fetchall()
                
                for token_form, specific_token_index in candidate_tokens:
                    context = get_token_context(cursor, specific_token_index)
                    if quote_match(quote_text, context):
                        verified_tokens_evaluated += 1
                        morph_out, morph_found, status = check_lemmatizer("Morpheus", token_form, db_lemma)
                        stats["Morpheus"][status] += 1
                        cltk_out, cltk_found, status = check_lemmatizer("Backoff", token_form, db_lemma)
                        stats["Backoff"][status] += 1
                        
                        example = { 
                                   "Lemma": db_lemma, 
                                   "Token": token_form, 
                                   "Location": f"{dialogue} {page}{section}", 
                                   "Counts": f"Lemma Count: {lemma_occurrence_count}, Ast: {ast_citation_count})", 
                                   "Quote": quote_text, 
                                   "Context": context, 
                                   "Morph": status, 
                                   "Backoff": status 
                        }
                        quote_verified_examples.append(example)
                        break
                    
            # example of this edge case: 
            # ei with circumflex and eis with circumflex
            # εἷ vs. εἷζ
            # BUT both map to the EIMI to be?
            """
            If the lemma appears twice and Ast cites fewer times than the lemma appears, there's gonna be options:
            1. No quote: toss
            2. Is a quote: Keep
            
            If 1 to 1 match with lemma occurrence count, keep
            If he cites more times than the lemma appears, don't include that section (toss it entirely) and flag it
            """
            
            """
            Quote example: for εἷμἰ, verb: to be, from Symp.600.A is his only citation
            Sentence: Socrates εἷ. εἷζ agaphos 
            Cannot differentiate which of the two he is talking about in his citation, so we toss it
            Lemma occurrence count is 2 (eimi appears twice), but RIGHT NOW our rule checker just does this:
            ei goes to eimi, don't see any other ei. it's cited once so include it
            eis goes to eimi, don't see any other eis. it's cited once so include it
            
            BUT this isn't flagged because its two different forms of the same lemma
            Summary of this edge case: two different forms of the same lemma is the specific edge case
            Reason why you want to check lemma opposed to token: if only checking token, could not catch the above edge case.
            If checking the lemma, the lemma occurrence count is 2 and so we do check this edge case
            Whether or not the tokens are the same at all does not matter for our evaluation
            
            BUT if Ast does εἷμἰ, verb: to be, from Symp.600.A AND quotes "εἷζ agaphos"
            In the quote, he's talking about this εἷζ. SO we toss εἷ 
            Would need a context window that says: these are all of the tokens that parse to this lemma
            """
            
            # eimi on page 630c. if both of the above forms are there, and he cites without a quote, we can't tell what
            # form he's tallking about (unless he cites twice). ONLY in that instance can we rely on a quote
            
            # Q:
            # quotes_by_ref = extract_quotes_from_json(long_def_json, db_lemma)

            # short_name = SHORT_NAMES.get(dialogue, dialogue)
            # cite_key = f"{short_name}.{page}.{section}"
            
            # if quotes_by_ref.get(cite_key):
            #     verified_tokens_evaluated += token_occurrence_count
                
            #     morph_output, morph_found, morph_status = check_lemmatizer("Morpheus", token, db_lemma)
            #     stats["Morpheus"][morph_status] += token_occurrence_count
                
            #     backoff_output, backoff_found, backoff_status = check_lemmatizer("Backoff", token, db_lemma)
            #     stats["Backoff"][backoff_status] += token_occurrence_count

    end = time()
    print(f"Finished Combined Eval in {end - begin:.2f} seconds!\n\n")
    print(f"Total verified tokens evaluated: {verified_tokens_evaluated}")

    for lemmatizer, results in stats.items():
        correct = results["C"]
        incorrect = results["I"]
        not_parsed = results["NP"]
        
        hit_rate, miss_rate, fail_rate = calc_error(correct, incorrect, not_parsed, verified_tokens_evaluated)
        
        print(f"\n----- {lemmatizer.upper()} STATS (on Ast-verified tokens) -----")
        print(f"Correct rate over parseable tokens: {hit_rate}%")
        print(f"Incorrect rate over parseable tokens: {miss_rate}%")
        print(f"No parse rate over all verified tokens: {fail_rate}%")
    
    print(f"\n----- QUOTE VERIFIED EXAMPLES -----")
    for ex in quote_verified_examples:
        print(ex)

if __name__ == "__main__":
    lemmatizer_evaluation_runner()
    connection.close()
    morph_connect.close()
    # json_obj = [{"subList": [{"subList": [], "text": {"identifier": "I", "keyPassageList": [], "refList": [{"note": ".  \u03c6\u03b9\u03bb\u03bf\u03c3\u03bf\u03c6\u03af\u03b1 \u03b3\u03ac\u03c1 \u1f10\u03c3\u03c4\u03b9 \u03c0\u03b1\u03bb\u03b1\u03b9\u03bf\u03c4\u03ac\u03c4\u03b7 \u03c4\u03b5 \u03ba\u03b1\u1f76 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03b7 \u03c4\u1ff6\u03bd \u1f19\u03bb\u03bb\u03ae\u03bd\u03c9\u03bd \u1f10\u03bd \u039a\u03c1\u03b7\u03c4\u1fc7 cet. ", "ref": "Prt.342.a", "refLink": "Prt/342/a"}, {"note": " (viii).  \u03b4\u03bf\u03c5\u03bb\u03b5\u03af\u03b1 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03b7. ", "ref": "Resp.564.a", "refLink": "Resp/564/a"}, {"note": " (iv).  \u1f43 \u03b4\u1f74 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03c4\u1fc6\u03c2 \u03c8\u03c5\u03c7\u1fc6\u03c2 (nect. \u1f43 \u03c4\u1fc6\u03c2 \u03c8\u03c5\u03c7\u1fc6\u03c2) \u1f10\u03bd \u1f11\u03ba\u03ac\u03c3\u03c4\u03c9 \u1f10\u03c3\u03c4\u03af. ", "ref": "Resp.442.a", "refLink": "Resp/442/a"}, {"note": ".  \u03c3\u03c9\u03c6\u03c1\u03bf\u03c3\u03c5\u03bd\u1fc6\u03c2 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03b7\u03c2 \u03bc\u03b5\u03c4\u03b5\u03c7\u03b5\u03b9. ", "ref": "Symp.196.c", "refLink": "Symp/196/c"}, {"note": " (i).  \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03bf\u03c5 \u1f00\u03be\u03af\u03b1\u03bd \u03b5\u1f36\u03bd\u03b1\u03b9. ", "ref": "Resp.331.a", "refLink": "Resp/331/a"}, {"note": " (ii).  \u03c0\u03cc\u03c4\u03b5\u03c1\u03bf\u03bd \u03bf\u1f50 \u03c0\u03b5\u03c1\u1f76 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03bf\u03c5 \u1f10\u03c3\u03c4\u1f76\u03bd \u03b5\u1f56 \u1f00\u03c0\u03b5\u03c1\u03b3\u03b1\u03c3\u03b8\u03ad\u03bd\u03c4\u03b1; ", "ref": "Resp.374.c", "refLink": "Resp/374/c"}, {"note": " (iii).  \u03c4\u1ff7 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u1ff3 \u1f44\u03c7\u03bb\u1ff3. Id. ", "ref": "Resp.397.d", "refLink": "Resp/397/d"}, {"note": " (iii). ", "ref": "Leg.700.c", "refLink": "Leg/700/c"}, {"note": ".  \u1f45\u03c4\u03b9 \u03bd\u1fe6\u03bd \u1f10\u03c3\u03bc\u1f72\u03bd \u1f10\u03bd \u1f00\u03b3\u03bd\u03bf\u03af\u1fb3 \u03c4\u1fc7 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u1fc3 \u03c0\u03b5\u03c1\u1f76 \u03b1\u1f50\u03c4\u03bf\u1fe6. ", "ref": "Soph.249.e", "refLink": "Soph/249/e"}, {"note": ".  \u1f45\u03c0\u03c9\u03c2 \u1f61\u03c2 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03c7\u03c1\u03cc\u03bd\u03bf\u03bd \u03b2\u03b9\u03ce\u03c3\u03b5\u03c4\u03b1\u03b9 \u03c4\u03bf\u03b9\u03bf\u1fe6\u03c4\u03bf\u03c2 \u1f64\u03bd. ", "ref": "Grg.481.b", "refLink": "Grg/481/b"}, {"note": " (ix).  \u1f41\u03c2 \u1f02\u03bd \u03b1\u1f50\u03c4\u03c9\u03bd \u03bc\u03ac\u03bb\u03b9\u03c3\u03c4\u03b1 ... \u03bc\u03ad\u03b3\u03b9\u03c3\u03c4\u03bf\u03bd \u03ba\u03b1\u1f76 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u1f10\u03bd \u03c4\u1fc7 \u03c8\u03bd\u03c7\u1fc7 \u03c4\u03cd\u03c1\u03b1\u03bd\u03bd\u03bf\u03bd \u1f14\u03c7\u1fc3. ", "ref": "Resp.575.c", "refLink": "Resp/575/c"}, {"note": ".  \u1f61\u03c2 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03b7\u03bd (\u1f21\u03b4\u03bf\u03bd\u03ae\u03bd). ", "ref": "Phlb.60.d", "refLink": "Phlb/60/d"}, {"note": ".  \u03c4\u03bf\u1fe6\u03c4\u03bf \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03b1\u1f50\u03c4\u1ff7 \u03b5\u1f30\u03c2 \u03c4\u1f78 \u1f44\u03bd\u03bf\u03bc\u03b1 \u1f10\u03bd\u03b5\u03ba\u03ad\u03c1\u03b1\u03c3\u03b5. ", "ref": "Cra.427.c", "refLink": "Cra/427/c"}, {"note": ".  \u1f10\u03c0\u1f76 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03c4\u03b5\u03af\u03bd\u03bf\u03bd\u03c4\u03b1\u03c2. ", "ref": "Symp.222.a", "refLink": "Symp/222/a"}, {"note": " (ii).  \u03ba\u03b1\u03b8\u2019 \u1f45\u03c3\u03bf\u03bd \u1f00\u03bd\u03b8\u03c1\u03ce\u03c0\u1ff3 \u1f10\u03c0\u1f76 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03bf\u1f37\u03cc\u03bd \u03c4\u03b5. ", "ref": "Resp.383.c", "refLink": "Resp/383/c"}, {"note": " (iv).  \u1f10\u03bd \u1f10\u03bb\u03c0\u03af\u03c3\u03b9\u03bd \u1f00\u03b3\u03b1\u03b8\u03b1\u1fd6\u03c2 \u03b4\u03b9\u03ac\u03b3\u03bf\u03bd\u03c4\u03b5\u03c2 \u03c4\u1f78 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03c4\u03bf\u1fe6 \u03b2\u03af\u03bf\u03c5. ", "ref": "Leg.718.a", "refLink": "Leg/718/a"}, {"note": ".  \u03c4\u1f78 ... \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03c0\u03c5\u03c1\u1f78\u03c2 \u1f14\u03c7\u03b5\u03b9. ", "ref": "Epin.981.d", "refLink": "Epin/981/d"}, {"note": ".  \u03c3\u03bf\u03c6\u03b9\u03c3\u03c4\u03b1\u1f76 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03b9 \u03b3\u1fc6\u03c2 \u1f10\u03ba\u03b5\u1fd6 \u03b5\u1f30\u03c3\u03af\u03bd. ", "ref": "Prt.342.b", "refLink": "Prt/342/b"}, {"note": " (v).  \u1f61\u03c2 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03b9 \u03c4\u1ff6\u03bd \u03c0\u03b1\u03af\u03b4\u03c9\u03bd. ", "ref": "Resp.460.b", "refLink": "Resp/460/b"}, {"note": ".  \u1f10\u03bd \u1fa7 \u03c4\u1f70 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03ac \u1f10\u03c3\u03c4\u03b9 \u03c4\u1ff6\u03bd \u03c0\u03ac\u03bb\u03b1\u03b9 \u1fe5\u03b7\u03b8\u03ad\u03bd\u03c4\u03c9\u03bd. ", "ref": "Plt.288.b", "refLink": "Plt/288/b"}, {"note": ".  \u03c4\u1ff6\u03bd \u03bb\u03b5\u03b3\u03bf\u03bc\u03ad\u03bd\u03c9\u03bd \u1f31\u03b1\u03c4\u03c1\u1ff6\u03bd \u1f00\u03c0\u03b1\u03c4\u1ff6\u03c3\u03b1 \u03c4\u03bf\u1f7a\u03c2 \u03c0\u03bb\u03b5\u03af\u03c3\u03c4\u03bf\u03c5\u03c2. ", "ref": "Ti.88.a", "refLink": "Ti/88/a"}, {"note": ".  \u1f43\u03c2 ... \u1f21\u03bc\u1fb6\u03c2 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03b1 \u1f40\u03bd\u03af\u03bd\u03b7\u03c3\u03b9\u03bd. ", "ref": "Symp.193.d", "refLink": "Symp/193/d"}, {"note": ".  \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03b1 \u1f00\u03b3\u03ac\u03bb\u03bc\u03b1\u03c4\u03b1 \u1f00\u03c1\u03b5\u03c4\u1fc6\u03c2 \u1f10\u03bd \u03b1\u1f51\u03c4\u03bf\u1fd6\u03c2 \u1f14\u03c7\u03bf\u03bd\u03c4\u03b1\u03c2. ", "ref": "Symp.222.a", "refLink": "Symp/222/a"}, {"note": " (vii).  \u03c4\u1f78 \u1f14\u03b8\u03bd\u03bf\u03c2 ... \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03b1 \u1f40\u03bd\u03ae\u03c3\u03b5\u03b9\u03bd. ", "ref": "Resp.541.a", "refLink": "Resp/541/a"}, {"note": ".  \u1f21 ... \u03bc\u03b5\u03b3\u03af\u03c3\u03c4\u03b7 \u03ba\u03b1\u1f76 \u03b5\u1f30\u03c2 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03b1 (ad plurima pertinens) ... \u03c3\u03c4\u03c1\u03b1\u03c4\u03b7\u03b3\u03b9\u03ba\u1f74 \u03c4\u03ad\u03c7\u03bd\u03b7. Adverbiorum partes agunt \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd, longissime (", "ref": "Epin.975.e", "refLink": "Epin/975/e"}, {"note": " (x).  \u1f03 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03c6\u03af\u03bb\u03bf\u03c3\u03bf\u03c6\u03af\u03b1\u03c2 ... \u1f00\u03c6\u03ad\u03c3\u03c4\u03b7\u03ba\u03b5. ", "ref": "Resp.587.a", "refLink": "Resp/587/a"}, {"note": " (x). ) et maxime (", "ref": "Resp.587.b", "refLink": "Resp/587/b"}, {"note": " (v).  \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03ba\u03b5\u03c7\u03c9\u03c1\u03b9\u03c3\u03bc\u03ad\u03bd\u03b7\u03bd \u03c6\u03cd\u03c3\u03b9\u03bd \u1f14\u03c7\u03bf\u03bd\u03c4\u03b1\u03c2. ", "ref": "Resp.453.e", "refLink": "Resp/453/e"}, {"note": ". ); \u03c4\u1f78 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd, maximam partem (", "ref": "Thg.130.d", "refLink": "Thg/130/d"}, {"note": ". ", "ref": "Plt.288.b", "refLink": "Plt/288/b"}, {"note": " (vii). ", "ref": "Resp.528.a", "refLink": "Resp/528/a"}, {"note": ". ", "ref": "Ti.64.c", "refLink": "Ti/64/c"}, {"note": " (iii). ", "ref": "Leg.679.a", "refLink": "Leg/679/a"}, {"note": " (iv).  \u1f61\u03c2 \u1f10\u03c0\u1f76 \u03c4\u1f78 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd); c. genit. (", "ref": "Leg.720.d", "refLink": "Leg/720/d"}, {"note": ".  \u1f10\u03bc\u03bc\u03b1\u03bd\u1f74\u03c2 \u03c4\u1f78 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03bd \u03b3\u03b9\u03b3\u03bd\u03cc\u03bc\u03b5\u03bd\u03bf\u03c2 \u03c4\u03bf\u1fe6 \u03b2\u03af\u03bf\u03c5); \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03b1 (", "ref": "Ti.86.c", "refLink": "Ti/86/c"}, {"note": ". ) et \u03c4\u1f70 \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03b1 (", "ref": "HpMai.281.b", "refLink": "HpMai/281/b"}, {"note": ". ). ", "ref": "Criti.118.c", "refLink": "Criti/118/c"}], "start": "\u03b7, \u03bf\u03bd, plurimus; maximus. "}}], "text": {"identifier": "\u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03c2", "keyPassageList": [], "refList": [], "start": "\u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03c2 "}}, {"subList": [{"subList": [], "text": {"identifier": "I", "keyPassageList": [], "refList": [{"note": ".  \u03b4\u03b9\u03cc\u03c4\u03b9 \u03b4\u03ad, \u1f41 \u03bb\u03cc\u03b3\u03bf\u03c2 \u03c0\u03bb\u03b5\u03af\u03c9\u03bd. ", "ref": "Ti.54.b", "refLink": "Ti/54/b"}, {"note": ".  \u03bf\u1f50\u03b4\u1f72\u03bd \u03c0\u03bb\u03b5\u03af\u03c9\u03bd \u1f41 \u03c0\u1fb6\u03c2 \u03c7\u03c1\u03cc\u03bd\u03bf\u03c2 cet. ", "ref": "Ap.40.e", "refLink": "Ap/40/e"}, {"note": ".  \u1f65\u03c3\u03c4\u03b5 \u03c0\u03bf\u03bb\u1f7a \u03c0\u03bb\u03b5\u03af\u03c9\u03bd \u1f10\u03bb\u03c0\u1f76\u03c2 cet. ", "ref": "Phdr.231.d", "refLink": "Phdr/231/d"}, {"note": " (iv).  \u03bc\u03b1\u03ba\u03c1\u03bf\u03c4\u03ad\u03c1\u03b1 \u03ba\u03b1\u1f76 \u03c0\u03bb\u03b5\u03af\u03c9\u03bd \u1f41\u03b4\u1f78\u03c2 \u1f21 \u1f10\u03c0\u1f76 \u03c4\u03bf\u1fe6\u03c4\u03bf \u1f04\u03b3\u03bf\u03c5\u03c3\u03b1. ", "ref": "Resp.435.d", "refLink": "Resp/435/d"}, {"note": " (viii).  \u1f00\u03c0\u03bf\u03c1\u03af\u03b1 \u03c0\u03bb\u03b5\u03af\u03c9\u03bd. ", "ref": "Leg.830.b", "refLink": "Leg/830/b"}, {"note": ".  \u1f04\u03c1\u03b1 \u03c4\u1ff7 \u1f34\u03c3\u1ff3 \u03bc\u03bf\u03c1\u03af\u1ff3 \u03b4\u03b9\u03bf\u03af\u03c3\u03b5\u03b9 \u03c4\u1f78 \u03c0\u03bb\u03ad\u03bf\u03bd \u03c4\u03bf\u1fe6 \u1f10\u03bb\u03ac\u03c4\u03c4\u03bf\u03bd\u03bf\u03c2 \u1f22 \u03c3\u03bc\u03b9\u03ba\u03c1\u1f78 \u03c4\u03ad\u03c1\u1ff3; ", "ref": "Prm.154.d", "refLink": "Prm/154/d"}, {"note": ".  \u03c4\u03cc\u03c4\u03b5 ... \u03c0\u03bb\u03ad\u03bf\u03bd \u1f40\u03bd\u03cc\u03bc\u03b1\u03c4\u03bf\u03c2 \u1f26\u03bd \u03c4\u1f78 \u03b3\u03ad\u03bd\u03bf\u03c2 cet. ", "ref": "Criti.114.e", "refLink": "Criti/114/e"}, {"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd \u03b3\u1f70\u03c1 \u1f34\u03c3\u03c9\u03c2 \u1f14\u03c1\u03b3\u03bf\u03bd. ", "ref": "La.190.c", "refLink": "La/190/c"}, {"note": ".  \u03c4\u03bf\u1fe6 \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03bf\u03c2 \u03b2\u03af\u03bf\u03c5. ", "ref": "Ti.75.c", "refLink": "Ti/75/c"}, {"note": " (vii).  \u03b4\u03b9\u03c0\u03bb\u03b1\u03c3\u03af\u03b1\u03c2 \u03c4\u03b5 \u03ba\u03b1\u1f76 \u1f14\u03c4\u03b9 \u03c0\u03bf\u03bb\u03bb\u03c9 \u03c0\u03bb\u03ad\u03bf\u03bd\u03bf\u03c2 \u1f00\u03c3\u03c7\u03bf\u03bb\u03af\u03b1\u03c2 \u1f10\u03c3\u03c4\u1f76 \u03b3\u03ad\u03bc\u03c9\u03bd \u1f41 ... \u03b2\u03af\u03bf\u03c2. ", "ref": "Leg.807.c", "refLink": "Leg/807/c"}, {"note": ".  \u1f35\u03bd\u03b1 \u03ba\u03b1\u1f76 \u1f00\u03c0\u03b1\u03bb\u03bb\u03b1\u03b3\u1ff6\u03bc\u03b5\u03bd \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03bf\u03c2 \u03b6\u03b7\u03c4\u03ae\u03c3\u03b5\u03c9\u03c2. ", "ref": "HpMai.303.c", "refLink": "HpMai/303/c"}, {"note": " (xi).  \u03bc\u1f74 \u03c0\u03bb\u03ad\u03bf\u03bd\u03bf\u03c2 \u03c4\u03b9\u03bc\u1fb6\u03bd \u03b4\u03b9\u03b1\u03c0\u03b5\u03b9\u03c1\u03ce\u03bc\u03b5\u03bd\u03bf\u03bd. ", "ref": "Leg.921.b", "refLink": "Leg/921/b"}, {"note": " (xiii).  \u03bf\u1f50\u03b4\u1f72\u03bd \u03b1\u1f56 \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03bf\u03c2 \u1f22 \u03b4\u03ad\u03ba\u03b1 \u03bc\u03bd\u1ff6\u03bd \u03b4\u03ad\u03bf\u03b9 \u1f02\u03bd cet. ", "ref": "Ep.361.e", "refLink": "Ep/361/e"}, {"note": ".  \u1f10\u03bd \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03b9 \u1f04\u03c1\u03b9\u03b8\u03bc\u1ff3. ", "ref": "Tht.196.b", "refLink": "Tht/196/b"}, {"note": ".  \u03c0\u03bb\u03b5\u03b9\u03cc\u03bd\u03b9 \u03ba\u03b1\u1f76 \u1f10\u03bb\u03ac\u03c4\u03c4\u03bf\u03bd\u03b9 \u03c7\u03c1\u03cc\u03bd\u1ff3. ", "ref": "Prm.154.d", "refLink": "Prm/154/d"}, {"note": ".  \u1f10\u03bd \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03b9 (\u1f00\u03c0\u03bf\u03c1\u03af\u1fb3) \u03c6\u03b1\u03b9\u03bd\u03cc\u03bc\u03b5\u03b8\u03b1. ", "ref": "Soph.250.e", "refLink": "Soph/250/e"}, {"note": " (ii).  \u03c0\u03bb\u03ad\u03bf\u03bd\u03b9 ... \u03ba\u03b1\u03ba\u1ff7 \u1f51\u03c0\u03b5\u03c1\u03b2\u03ac\u03bb\u03bb\u03b5\u03b9\u03bd \u03c4\u1f78 \u1f00\u03b4\u03b9\u03ba\u03b5\u1fd6\u03c3\u03b8\u03b1\u03b9 cet. ", "ref": "Resp.358.e", "refLink": "Resp/358/e"}, {"note": " (ix).  \u1f00\u03bc\u03b7\u03c7\u03ac\u03bd\u1ff3 ... \u1f45\u03c3\u03c9 \u03c0\u03bb\u03ad\u03bf\u03bd\u03b9 \u03bd\u03b9\u03ba\u03ae\u03c3\u03b5\u03b9 \u03b5\u1f50\u03c3\u03c7\u03b7\u03bc\u03bf\u03c3\u03cd\u03bd\u1fc3 \u03c4\u03b5 \u03b2\u03af\u03bf\u03c5 cet. ", "ref": "Resp.588.a", "refLink": "Resp/588/a"}, {"note": " (vi).  \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03b9 \u03c4\u1ff6\u03bd \u03bd\u03ad\u03c9\u03bd \u03b6\u03b7\u03bc\u03b9\u03bf\u03cd\u03c3\u03b8\u03c9. ", "ref": "Leg.762.d", "refLink": "Leg/762/d"}, {"note": ".  \u03c0\u03bb\u03b5\u03af\u03c9 \u03c0\u03bf\u03c5 \u03c7\u03c1\u03cc\u03bd\u03bf\u03bd \u03b3\u03ad\u03b3\u03bf\u03bd\u03b5\u03bd \u1f22 \u03c4\u1f04\u03bb\u03bb\u03b1. ", "ref": "Prm.154.c", "refLink": "Prm/154/c"}, {"note": " (vii).  \u03c6\u03b9\u03bb\u03bf\u03c3\u03bf\u03c6\u03af\u03b1\u03c2 \u1f14\u03c4\u03b9 \u03c0\u03bb\u03b5\u03af\u03c9 \u03b3\u03ad\u03bb\u03c9\u03c4\u03b1 \u03ba\u03b1\u03c4\u03b1\u03bd\u03c4\u03bb\u03b7\u03c3\u03bf\u03bc\u03b5\u03bd. ", "ref": "Resp.536.b", "refLink": "Resp/536/b"}, {"note": " (ix).  \u1f45\u03c3\u03c9 \u1f02\u03bd \u03c0\u03bb\u03b5\u03af\u03c9 \u03c7\u03c1\u03cc\u03bd\u03bf\u03bd \u1f10\u03bd \u03c4\u03c5\u03c1\u03b1\u03bd\u03bd\u03af\u03b4\u03b9 \u03b2\u03b9\u1ff3. ", "ref": "Resp.576.b", "refLink": "Resp/576/b"}, {"note": " (xii).  (\u1f10\u1f70\u03bd) \u03c4\u1f78\u03bd \u1f25\u03bc\u03b9\u03c3\u03c5\u03bd \u1f00\u03c1\u03b9\u03b8\u03bc\u1f78\u03bd \u03c0\u03bb\u03b5\u03af\u03c9 \u03c0\u03bf\u03b9\u1ff6 \u03c3\u03b9\u03bd. ", "ref": "Leg.946.a", "refLink": "Leg/946/a"}, {"note": " (i).  \u03b2\u03c1\u03b1\u03c7\u03b5\u1f76 \u03b3\u03ad \u03c4\u03b9\u03bd\u03b9 \u03c0\u03bb\u03b5\u03af\u03c9 (\u03c4\u1f74\u03bd \u03bf\u1f50\u03c3\u03af\u03b1\u03bd) \u1f22 \u03c0\u03b1\u03c1\u03ad\u03bb\u03b1\u03b2\u03bf\u03bd. ", "ref": "Resp.330.b", "refLink": "Resp/330/b"}, {"note": " (viii).  \u03bf\u1f50\u03b4\u1f72\u03bd \u03c0\u03bb\u03b5\u03af\u03c9 \u1f10\u03c0\u03b9\u03bc\u03ad\u03bb\u03b5\u03b9\u03b1\u03bd \u03c0\u03b5\u03c0\u03bf\u03b9\u03ae\u03bc\u03ad\u03bd\u03bf\u03c5\u03c2 \u1f00\u03c1\u03b5\u03c4\u1fc6\u03c2 \u1f22 \u03c4\u03bf\u1f7a\u03c2 \u03c0\u03ad\u03bd\u03b7\u03c4\u03b1\u03c2. ", "ref": "Resp.556.c", "refLink": "Resp/556/c"}, {"note": ".  \u03b5\u1f30 \u1f10\u03b3\u1f7c \u03b4\u1f72 \u03bf\u1f50\u03b4\u1f72\u03bd \u1f10\u03c0\u03af\u03c3\u03c4\u03b1\u03bc\u03b1\u03b9 \u03c0\u03bb\u03ad\u03bf\u03bd \u03c0\u03bb\u1f74\u03bd \u03b2\u03c1\u03b1\u03c7\u03ad\u03bf\u03c2, \u1f45\u03c3\u03bf\u03bd cet. ", "ref": "Tht.161.b", "refLink": "Tht/161/b"}, {"note": ".  \u1f45\u03c3\u03bf\u03b9 \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd \u1f11\u03bd\u1f78\u03c2 \u03bb\u03ad\u03b3\u03bf\u03c5\u03c3\u03b9 \u03c4\u1f78 \u03c0\u1fb6\u03bd \u03b5\u1f36\u03bd\u03b1\u03b9. ", "ref": "Soph.244.b", "refLink": "Soph/244/b"}, {"note": ".  \u03c4\u1f78 \u03c0\u03bb\u03ad\u03bf\u03bd \u03ba\u03b1\u1f76 \u03c4\u1f78 \u1f14\u03bb\u03b1\u03c4\u03c4\u03bf\u03bd \u1f00\u03c0\u03b5\u03c1\u03b3\u03ac\u03b6\u03b5\u03c3\u03b8\u03bf\u03bd. ", "ref": "Phlb.24.c", "refLink": "Phlb/24/c"}, {"note": " (v).  \u03c0\u03bb\u03ad\u03bf\u03bd \u03b5\u1f36\u03bd\u03b1\u03b9 ... \u1f25\u03bc\u03b9\u03c3\u03c5 \u03c0\u03b1\u03bd\u03c4\u03cc\u03c2. ", "ref": "Resp.466.c", "refLink": "Resp/466/c"}, {"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd \u1f22 \u03b5\u1f34\u03ba\u03bf\u03c3\u03b9 \u03bc\u03bd\u1fb6\u03c2. ", "ref": "HpMai.282.e", "refLink": "HpMai/282/e"}, {"note": ".  \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd \u03b4\u2019 \u03bf\u1f50\u03b4\u03b5\u1f76\u03c2 \u1f02\u03bd \u1f10\u03ba\u03b5\u03af\u03bd\u03c9\u03bd \u03b5\u1f50\u03be\u03b1\u03bc\u03ad\u03bd\u03c9\u03bd \u1f00\u03ba\u03bf\u03cd\u03c3\u03b5\u03b9\u03b5. ", "ref": "Alc2.148.c", "refLink": "Alc2/148/c"}], "start": "(et \u03c0\u03bb\u03ad\u03c9\u03bd), \u1f41 \u1f21, maior; longior. "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u03bc\u03b7\u03b4\u1f72\u03bd \u03c0\u03bb\u03ad\u03bf\u03bd \u03b1\u1f50\u03c4\u1ff7 \u03b3\u03ad\u03bd\u03b7\u03c4\u03b1\u03b9. ", "ref": "Symp.222.d", "refLink": "Symp/222/d"}], "start": "Etiam cum v. \u03b3\u03af\u03b3\u03bd\u03b5\u03c3\u03b8\u03b1\u03b9. "}}], "text": {"identifier": "II", "keyPassageList": [], "refList": [{"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd \u03c4\u03b9 \u03bf\u1f30\u03cc\u03bc\u03b5\u03bd\u03bf\u03c2 \u03b5\u1f36\u03bd\u03b1\u03b9 \u03bb\u03cc\u03b3\u03bf\u03c5\u03c2 \u03b3\u03b5\u03b3\u03c1\u03b1\u03bc\u03bc\u03ad\u03bd\u03bf\u03c5\u03c2 \u03c4\u03bf\u1fe6 \u03c4\u1f78\u03bd \u03b5\u1f30\u03b4\u03ad\u03c4\u03b1 \u1f51\u03c0\u03bf\u03bc\u03bd\u1fc6\u03c3\u03b1\u03b9 cet. ", "ref": "Phdr.275.c", "refLink": "Phdr/275/c"}, {"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd \u03c4\u03b9 \u03b7\u03bc\u1fd6\u03bd \u1f14\u03c3\u03c4\u03b1\u03b9. ", "ref": "Cra.387.a", "refLink": "Cra/387/a"}, {"note": ".  \u03bf\u1f50\u03b4\u1f72\u03bd \u03b3\u03ac\u03c1 \u03bc\u03bf\u03b9 \u03c0\u03bb\u03ad\u03bf\u03bd \u1f22\u03bd (germ. es half mir nichts). ", "ref": "Symp.217.c", "refLink": "Symp/217/c"}, {"note": " (vi). ", "ref": "Resp.505.b", "refLink": "Resp/505/b"}, {"note": " (iii).  \u1f04\u03bd \u03c4\u03b9 \u03ba\u03b1\u1f76 \u03c3\u03bc\u03b9\u03ba\u03c1\u1f78\u03bd \u03c0\u03bb\u03ad\u03bf\u03bd \u1f11\u03ba\u03ac\u03c3\u03c4\u03bf\u03c4\u03b5 \u1f21\u03b3\u1ff6\u03bd\u03c4\u03b1\u03b9 \u1f14\u03c3\u03b5\u03c3\u03b8\u03b1\u03af \u03c3\u03c6\u03b9\u03bf\u03b9\u03bd. ", "ref": "Leg.697.d", "refLink": "Leg/697/d"}, {"note": " (vi). ", "ref": "Leg.751.b", "refLink": "Leg/751/b"}, {"note": ".  al. ", "ref": "Alc1.106.a", "refLink": "Alc1/106/a"}], "start": "\u2014 Cum v. \u03b5\u1f36\u03bd\u03b1\u03b9 est plus valere vel efficere et simpl. prodesse. "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u03bf\u1f50\u03b4\u1f72\u03bd \u03c0\u03bb\u03ad\u03bf\u03bd \u03c0\u03bf\u03b9\u03ae\u03c3\u03b5\u03c4\u03b5. ", "ref": "Phd.115.c", "refLink": "Phd/115/c"}, {"note": ".  \u03bf\u1f50\u03b4\u1f72\u03bd \u03c0\u03bb\u03ad\u03bf\u03bd \u03c0\u03bf\u03b9 \u03bf\u1f54\u03bd\u03c4\u03b5\u03c2. ", "ref": "Tht.200.c", "refLink": "Tht/200/c"}, {"note": ". ", "ref": "Cra.387.c", "refLink": "Cra/387/c"}, {"note": ". ", "ref": "Cra.428.a", "refLink": "Cra/428/a"}, {"note": " (iii).  \u1f41\u03c0\u03cc\u03c4\u03b1\u03bd \u03ba\u03b1\u03bb\u03bf\u1f76 \u1f10\u03bd \u03c8\u03c5\u03c7\u1fc7 \u03bb\u03cc\u03b3\u03bf\u03b9 \u1f10\u03bd\u03cc\u03bd\u03c4\u03b5\u03c2 \u03bc\u03b7\u03b4\u1f72\u03bd \u03c0\u03bf\u03b9\u1ff6\u03c3\u03b9 \u03c0\u03bb\u03ad\u03bf\u03bd. ", "ref": "Leg.689.b", "refLink": "Leg/689/b"}, {"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd \u03c4\u03b9 \u03bc\u03b5 \u03c0\u03bf\u03b9\u1fc6\u03c3\u03b1\u03b9 \u1f00\u03c0\u03bf\u03bb\u03bf\u03b3\u03bf\u03cd\u03bc\u03b5\u03bd\u03bf\u03bd.\u00a0\u2014\u00a0 ", "ref": "Ap.19.a", "refLink": "Ap/19/a"}, {"note": " (vii).  \u1f22 \u03c0\u03bb\u03b5\u03af\u03bf\u03c5\u03c2 \u1f22 \u03b5\u1f36\u03c2. ", "ref": "Resp.540.d", "refLink": "Resp/540/d"}, {"note": " (ix).  \u03c0\u03b1\u1fd6\u03b4\u03b5\u03c2 ... \u03c0\u03bb\u03b5\u03af\u03bf\u03c5\u03c2. ", "ref": "Leg.878.a", "refLink": "Leg/878/a"}, {"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd\u03b1 ... \u03c4\u1f70 \u03c0\u03b1\u03bd\u03c4\u03b1 \u1f11\u03bd\u1f78\u03c2 \u1f14\u03c3\u03c4\u03b1\u03b9. ", "ref": "Soph.245.b", "refLink": "Soph/245/b"}, {"note": " (ii).  \u1f10\u03ba ... \u03c4\u03bf\u03cd\u03c4\u03c9\u03bd \u03c0\u03bb\u03b5\u03af\u03c9 ... \u1f15\u03ba\u03b1\u03c3\u03c4\u03b1 \u03b3\u03af\u03b3\u03bd\u03b5\u03c4\u03b1\u03b9 cet. ", "ref": "Resp.370.c", "refLink": "Resp/370/c"}, {"note": " (iv).  \u03c4\u1f70 \u03c0\u03bb\u03b5\u03af\u03c9 ... \u03c0\u03c1\u1f78\u03c2 \u03c4\u1f70 \u1f10\u03bb\u03ac\u03c4\u03c4\u03c9. ", "ref": "Resp.438.b", "refLink": "Resp/438/b"}, {"note": " (ii).  \u03c0\u03bb\u03b5\u03b9\u03cc\u03bd\u03c9\u03bd ... \u03b3\u03b5\u03c9\u03c1\u03b3\u1ff6\u03bd ... \u03b4\u03b5\u1fd6 \u1f21\u03bc\u1fd6\u03bd \u03c4\u1fc7 \u03c0\u03cc\u03bb\u03b5\u03b9. ", "ref": "Resp.371.a", "refLink": "Resp/371/a"}, {"note": ".  \u1f10\u03ba \u03c4\u1ff6\u03bd \u1f11\u03c4\u03ad\u03c1\u03c9\u03bd \u03c0\u03bb\u03b5\u03b9\u03bf\u03bd\u1ff6\u03bd. ", "ref": "Phlb.46.d", "refLink": "Phlb/46/d"}, {"note": " (vi).  \u03c4\u03bf\u03c3\u03bf\u03cd\u03c4\u1ff3 \u03c0\u03bb\u03b5\u03b9\u03cc\u03bd\u03c9\u03bd \u1f10\u03bd\u03b4\u03b5\u1fd6 \u03c4\u1ff6\u03bd \u03c0\u03c1\u03b5\u03c0\u03cc\u03bd\u03c4\u03c9\u03bd. ", "ref": "Resp.491.d", "refLink": "Resp/491/d"}, {"note": " (xi).  \u03b4\u03ad\u03ba\u03b1 \u03c0\u03bb\u03b5\u03af\u03bf\u03c3\u03b9\u03bd \u1f14\u03c4\u03b5\u03c3\u03b9\u03bd \u03ba\u03bf\u03bb\u03b1\u03b6\u03ad\u03c3\u03b8\u1ff6\u03bd cet. ", "ref": "Leg.932.c", "refLink": "Leg/932/c"}, {"note": ".  \u03b4\u03b9\u1f70 \u03c4\u1f78 \u03c0\u03bb\u03b5\u03af\u03bf\u03c5\u03c2 \u03c4\u1f78\u03bd \u1f00\u03c1\u03b9\u03b8\u03bc\u1f78\u03bd \u03b3\u03b5\u03b3\u03bf\u03bd\u03ad\u03bd\u03b1\u03b9. ", "ref": "Symp.190.d", "refLink": "Symp/190/d"}, {"note": " (iv). ", "ref": "Resp.422.c", "refLink": "Resp/422/c"}, {"note": ".  \u03bc\u03b5\u03af\u03b6\u03bf\u03c5\u03c2 \u1f21\u03b4\u03bf\u03bd\u03ac\u03c2, \u03bf\u1f50 \u03c0\u03bb\u03b5\u03af\u03bf\u03c5\u03c2 \u03bb\u03ad\u03b3\u03c9. ", "ref": "Phlb.45.d", "refLink": "Phlb/45/d"}, {"note": ".  \u03c0\u03cc\u03c4\u03b5\u03c1\u03bf\u03bd \u1f21\u03b4\u03bf\u03bd\u1f74 \u1f22 \u03c6\u03c1\u03cc\u03bd\u03b7\u03c3\u03b9\u03c2 \u03c0\u03bb\u03b5\u03af\u03c9 (int. \u03bc\u03b5\u03c4\u03c1\u03b9\u03cc\u03c4\u03b7\u03c4\u03bf\u03c2) \u03ba\u03ad\u03ba\u03c4\u03b7\u03c4\u03b1\u03b9. ", "ref": "Phlb.65.d", "refLink": "Phlb/65/d"}, {"note": ".  \u1f00\u03c0\u03cc\u03ba\u03c1\u03b9\u03bd\u03b1\u03b9 \u1f40\u03bb\u03af\u03b3\u1ff3 \u03c0\u03bb\u03b5\u03af\u03c9. ", "ref": "Symp.199.e", "refLink": "Symp/199/e"}, {"note": ".  \u1f14\u03c4\u03b7 \u03b3\u03b5\u03b3\u03bf\u03bd\u1f7c\u03c2 \u03c0\u03bb\u03b5\u03af\u03c9 \u1f11\u03b2\u03b4\u03bf\u03bc\u03ae\u03ba\u03bf\u03bd\u03c4\u03b1. ", "ref": "Ap.17.c", "refLink": "Ap/17/c"}, {"note": ".  \u03c0\u03bb\u03b5\u03af\u03c9 \u03c7\u03c1\u03ae\u03bc\u03b1\u03c4\u03b1 \u03b5\u1f30\u03c1\u03b3\u03ac\u03c3\u03b8\u03b1\u03b9 \u1f22 cet. ", "ref": "HpMai.282.e", "refLink": "HpMai/282/e"}], "start": "Cum v. \u03c0\u03bf\u03b9\u03b5\u1fd6\u03bd, proficere. "}}], "text": {"identifier": "III", "keyPassageList": [], "refList": [{"note": ". ", "ref": "Grg.483.c", "refLink": "Grg/483/c"}, {"note": ". ", "ref": "Grg.483.d", "refLink": "Grg/483/d"}, {"note": ".  \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd (recentt. \u03c0\u03bb\u03ad\u03bf\u03bd) \u1f14\u03c7\u03b5\u03b9\u03bd. ", "ref": "Grg.488.b", "refLink": "Grg/488/b"}, {"note": " (i). ", "ref": "Resp.343.d", "refLink": "Resp/343/d"}, {"note": " (i). ", "ref": "Resp.349.b", "refLink": "Resp/349/b"}, {"note": " (i). ", "ref": "Resp.349.c", "refLink": "Resp/349/c"}, {"note": " (ix).  ", "ref": "Resp.574.a", "refLink": "Resp/574/a"}], "start": "\u2014 Cum v. \u1f14\u03c7\u03b5\u03b9\u03bd, plus habere, i. q. \u03c0\u03bb\u03b5\u03bf\u03bd\u03b5\u03ba\u03c4\u03b5\u1fd6\u03bd. "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u03bf\u1f50\u03b4\u03b5\u03bc\u03af\u03b1 ... \u03c4\u1fc6\u03c2 \u03b8\u03b7\u03c1\u03b5\u03c5\u03c4\u03b9\u03ba\u1fc6\u03c2 \u03b1\u1f50\u03c4\u1fc6\u03c2 \u1f10\u03c0\u1f76 \u03c0\u03bb\u03ad\u03bf\u03bd \u1f10\u03c3\u03c4\u03af\u03bd, \u1f22 \u1f45\u03c3\u03bf\u03bd \u03b8\u03b7\u03c1\u03b5\u03cd\u03c3\u03b1\u03b9 (nulla \u2014 ipsa venandi arte latius patet, sc. quam ut venetur).", "ref": "Euthd.290.b", "refLink": "Euthd/290/b"}, {"note": ".  \u1f22 \u1f14\u03c7\u03b5\u03b9\u03c2 \u03c4\u03b9 \u03bb\u03ad\u03b3\u03b5\u03b9\u03bd \u1f10\u03c0\u1f76 \u03c0\u03bb\u03ad\u03bf\u03bd \u03c4\u1f74\u03bd \u1fe5\u03b7\u03c4\u03bf\u03c1\u03b9\u03ba\u1f74\u03bd \u03b4\u03cd\u03bd\u03b1\u03c3\u03b8\u03b1\u03b9 \u1f22 \u2014; ", "ref": "Grg.453.a", "refLink": "Grg/453/a"}, {"note": ".  \u1f06\u03c1\u2019 \u03bf\u1f56\u03bd \u1f10\u03c0\u1f76 \u03c0\u03bb\u03ad\u03bf\u03bd \u03c4\u03b9 \u03b4\u03cd\u03bd\u03b1\u03c4\u03b1\u03b9 \u03c4\u03bf\u1fe6 \u03c0\u03b5\u03c1\u1f76 \u03c4\u1f70 \u03be\u03c5\u03bc\u03b2\u03cc\u03bb\u03b1\u03b9\u03b1 ... \u03ba\u03c1\u03af\u03bd\u03b5\u03b9\u03bd \u2014; ", "ref": "Plt.305.b", "refLink": "Plt/305/b"}, {"note": ".  \u1f10\u03c0\u1f76 \u03c0\u03bb\u03ad\u03bf\u03bd \u03b3\u1f70\u03c1 ... \u03b4\u1f72\u03bf\u03c2 \u03b1\u1f30\u03b4\u03bf\u1fe6\u03c2. ", "ref": "Euthphr.12.c", "refLink": "Euthphr/12/c"}], "start": "Cum v. \u03b5\u1f36\u03bd\u03b1\u03b9 et \u03b4\u03cd\u03bd\u03b1\u03c3\u03b8\u03b1\u03af est plus valere vel latius patere. "}}], "text": {"identifier": "IV", "keyPassageList": [], "refList": [{"note": ".  \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd \u1f22 \u03ba\u03b5\u1fd6\u03bd\u03bf\u03c2 \u1f00\u03c0\u03b5\u1fd6\u03c0\u03b5 \u03c3\u03ba\u03bf\u03c0\u03b5\u1fd6\u03bd \u1f21\u03bc\u03b5\u1fd6\u03c2 \u03b5\u1f30\u03c2 \u03c4\u1f78 \u03c0\u03c1\u03bf\u03c3\u03b8\u03b5\u03bd \u1f14\u03c4\u03b9 \u03b6\u03b7\u03c4\u03ae\u03c3\u03b1\u03bd\u03c4\u03b5\u03c2 \u1f00\u03c0\u03b5\u03b4\u03b5\u03af\u03be\u03b1\u03bc\u03b5\u03bd \u03b1\u1f50\u03c4\u1ff7. ", "ref": "Soph.258.c", "refLink": "Soph/258/c"}, {"note": ". ", "ref": "Plt.262.c", "refLink": "Plt/262/c"}, {"note": ".  \u1f10\u03bd \u03bc\u03ac\u03c1\u03c4\u03c5\u03c3\u03b9 \u03c4\u1ff6\u03bd \u1f19\u03bb\u03bb\u03ae\u03bd\u03c9\u03bd \u03c0\u03bb\u03b5\u1f78\u03bd \u1f22 \u03c4\u03c1\u03b9\u03c2\u03bc\u03c5\u03c1\u03af\u03bf\u03b9\u03c2. ", "ref": "Symp.175.e", "refLink": "Symp/175/e"}, {"note": " (vi).  \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd \u1f22 \u03c4\u03c1\u03b9\u03ac\u03ba\u03bf\u03bd\u03c4\u03b1 \u1f10\u03c0\u03b9\u03b4\u03b5\u03bf\u03bc\u03ad\u03bd\u03b7\u03bd (\u03c4\u1f74\u03bd \u1f00\u03c1\u03c7\u1f74\u03bd) \u1f21\u03bc\u03b5\u03c1\u1ff6\u03bd. ", "ref": "Leg.766.c", "refLink": "Leg/766/c"}, {"note": ".  \u03c0\u03bb\u03ad\u03bf\u03bd \u1f22 \u03c4\u03b5\u03c4\u03c4\u03b1\u03c1\u03ac\u03ba\u03bf\u03bd\u03c4\u03b1 \u1f14\u03c4\u03b7. ", "ref": "Men.91.e", "refLink": "Men/91/e"}, {"note": ".  \u1f10\u03b3\u03ad\u03bd\u03b5\u03c4\u03bf \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd (v. \u03c0\u03bb\u03b5\u03af\u03c9\u03bd) \u1f22 \u03b4\u03b5\u03ba\u03b1\u03c0\u03bb\u03ac\u03c3\u03b9\u03bf\u03c2. De dict. \u1f10\u03c0\u1f76 \u03c0\u03bb\u03ad\u03bf\u03bd vid. I. p. 768. ", "ref": "Euthd.300.d", "refLink": "Euthd/300/d"}], "start": "\u2014 Adverbiorum instar frequentatur \u03c0\u03bb\u03b5\u03cc\u03bd vel \u03c0\u03bb\u03b5\u1fd6\u03bf\u03bd, plus, amplius, ulterius. "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u1f40\u03bb\u03b9\u03b3\u03ac\u03ba\u03b9\u03c2 \u03bc\u1f72\u03bd \u1f60\u03c6\u03b5\u03bb\u03b5\u1fd6\u03bd, \u03b2\u03bb\u03ac\u03c0\u03c4\u03b5\u03b9\u03bd \u03b4\u1f72 \u03c4\u1f70 \u03c0\u03bb\u03b5\u03af\u03c9 \u03c4\u1f78\u03bd \u1f14\u03c7\u03bf\u03bd\u03c4\u03b1 \u03b1\u1f50\u03c4\u03ac. ", "ref": "Alc2.144.d", "refLink": "Alc2/144/d"}, {"note": ".  \u03b2\u03bb\u03ac\u03c0\u03c4\u03b5\u03c3\u03b8\u03b1\u03b9 \u03c4\u1f70 \u03c0\u03bb\u03b5\u03af\u03c9 \u03bc\u1fb6\u03bb\u03bb\u03bf\u03bd \u1f22 \u1f60\u03c6\u03b5\u03bb\u03b5\u1fd6\u03c3\u03b8\u03b1\u03b9. ", "ref": "Alc2.146.b", "refLink": "Alc2/146/b"}, {"note": ". ", "ref": "Alc2.146.e", "refLink": "Alc2/146/e"}, {"note": ".  \u03c3\u03c5\u03bd\u03ad\u03b2\u03b1\u03b9\u03bd\u03b5 \u03b3\u1f70\u03c1 \u03b1\u1f50\u03c4\u1ff7 \u03c4\u1f70 \u03c0\u03bb\u03b5\u03af\u03c9 \u03c4\u1f70\u03c2 \u03c4\u03bf\u03b9\u03b1\u03cd\u03c4\u03b1\u03c2 \u03b2\u03bb\u03b1\u03b2\u03b5\u03c1\u03ac\u03c2, \u1f22 \u1f00\u03b3\u03b1\u03b8\u1f70\u03c2 \u03b5\u1f36\u03bd\u03b1\u03b9. ", "ref": "Clit.409.e", "refLink": "Clit/409/e"}], "start": "Cum art. est saepius. "}}], "text": {"identifier": "V", "keyPassageList": [], "refList": [{"note": ".  \u03b5\u1f30 \u03c0\u03bb\u03b5\u03af\u03c9 \u03c7\u03b1\u03af\u03c1\u03bf\u03c5\u03c3\u03b9\u03bd \u03bf\u1f31 \u03c3\u03c6\u03cc\u03b4\u03c1\u03b1 \u03bd\u03bf\u03c3\u03bf\u1fe6\u03bd\u03c4\u03b5\u03c2 \u03c4\u1ff6\u03bd \u1f51\u03b3\u03b9\u03b1\u03b9\u03bd\u03cc\u03bd\u03c4\u03c9\u03bd. ", "ref": "Phlb.45.c", "refLink": "Phlb/45/c"}, {"note": " (iii).  \u03c0\u03bf\u03bb\u1f7a \u03c0\u03bb\u03b5\u03af\u03c9 \u03ba\u03b1\u1f76 \u03bc\u1fb6\u03bb\u03bb\u03bf\u03bd \u03b4\u03b5\u03b4\u03b9\u03cc\u03c4\u03b5\u03c2 \u03c4\u03bf\u1f7a\u03c2 \u1f14\u03bd\u03b4\u03bf\u03bd \u1f22 \u03c4\u03bf\u1f7a\u03c2 \u1f14\u03be\u03c9\u03b8\u03b5\u03bd \u03c0\u03bf\u03bb\u03b5\u03bc\u03af\u03bf\u03c5\u03c2. ", "ref": "Resp.417.b", "refLink": "Resp/417/b"}, {"note": ".  \u03c0\u03b1\u03c1\u03b1\u03bc\u03ad\u03bd\u03b5\u03b9 \u1f21\u03bc\u03ad\u03c1\u03b1\u03c2 \u03c0\u03bb\u03b5\u03af\u1ff3 \u1f22 \u03c4\u03c1\u03b5\u1fd6\u03c2. ", "ref": "Menex.235.b", "refLink": "Menex/235/b"}, {"note": " (vi).  \u03c0\u03b5\u03c0\u03b5\u03b9\u03c1\u03b1\u03bc\u03ad\u03bd\u03bf\u03c2 \u1f18\u03c1\u03ac\u03c3\u03c4\u03bf\u03c5 ... \u03c0\u03bb\u03ad\u03bf\u03bd\u03b1 \u1f22 \u03c3\u03cd ", "ref": "Ep.323.a", "refLink": "Ep/323/a"}], "start": "\u2014 Similiter \u03c0\u03bb\u03b5\u03af\u03c9 usurpatur. "}}], "text": {"identifier": "\u03c0\u03bb\u03b5\u03af\u03c9\u03bd", "keyPassageList": [], "refList": [], "start": "\u03c0\u03bb\u03b5\u03af\u03c9\u03bd "}}, {"subList": [{"subList": [], "text": {"identifier": "I", "keyPassageList": [], "refList": [], "start": "vid. \u03c0\u03bb\u03b5\u03af\u03c9\u03bd. "}}], "text": {"identifier": "\u03c0\u03bb\u03ad\u03bf\u03bd", "keyPassageList": [], "refList": [], "start": "\u03c0\u03bb\u03ad\u03bf\u03bd "}}, {"subList": [{"subList": [], "text": {"identifier": "I", "keyPassageList": [], "refList": [{"note": ".  \u1f03 \u03bb\u03cc\u03b3\u03bf\u03c2 \u03c0\u03bf\u03bb\u1f7a\u03c2 \u1f02\u03bd \u03b5\u1f34\u03b7 \u03b4\u03b9\u03b5\u03bb\u03b8\u03b5\u1fd6\u03bd. ", "ref": "Phdr.274.e", "refLink": "Phdr/274/e"}, {"note": ".  \u03b2\u03b1\u03b8\u03cd\u03c2 \u03c4\u03b5 \u03ba\u03b1\u1f76 \u03c0\u03bf\u03bb\u03cd\u03c2 (\u1f41 \u03ba\u03b7\u03c1\u1f78\u03c2 \u1f10\u03bd \u03c4\u1fc7 \u03c8\u03c5\u03c7\u1fc6). ", "ref": "Tht.194.c", "refLink": "Tht/194/c"}, {"note": " (vi).  \u03bf\u1f50\u03ba \u1f02\u03bd \u03c0\u03bf\u03bb\u1f7a\u03c2 \u1f10\u03c0\u03b9\u03b4\u03b5\u03af\u03be\u03b5\u03b9\u03b5 \u03bc\u1fe6\u03b8\u03bf\u03c2. ", "ref": "Leg.761.c", "refLink": "Leg/761/c"}, {"note": ".  \u03c0\u03bf\u03bb\u03cd\u03c2, \u03b5\u1f30\u03ba\u1fc7 \u03c3\u03c5\u03bc\u03c0\u03b5\u03c6\u03bf\u03c1\u03b7\u03bc\u03ad\u03bd\u03bf\u03c2 (ubi est crassus; oppos. \u1f40\u03c1\u03b8\u1f78\u03c2 \u03ba\u03b1\u1f76 \u03b4\u03b9\u03b7\u03c1\u03b8\u03c1\u03c9\u03bc\u03ad\u03bd\u03bf\u03c2). ", "ref": "Phdr.253.e", "refLink": "Phdr/253/e"}, {"note": ".  \u03bf\u1f50 \u03b4 \u1f15\u03bd\u03b5\u03c7\u2019 \u1f21 \u03c0\u03bf\u03bb\u03bb\u1fc7 \u03c3\u03c0\u03bf\u03c5\u03b4\u03ae cet. ", "ref": "Phdr.248.b", "refLink": "Phdr/248/b"}, {"note": ".  \u1f26 ... \u03c0\u03b7\u03b3\u1f74 \u03c0\u03bf\u03bb\u03bb\u1f74 \u03c6\u03b5\u03c1\u03bf\u03bc\u03ad\u03bd\u1fc3 \u03c0\u03c1\u1f78\u03c2 \u03c4\u1f78\u03bd \u1f10\u03c1\u03b1\u03c3\u03c4\u03ae\u03bd. ", "ref": "Phdr.255.c", "refLink": "Phdr/255/c"}, {"note": ".  \u03bf\u1f50 \u03c0\u03bf\u03bb\u03bb\u1f74 \u1f02\u03bd \u1f00\u03bb\u03bf\u03b3\u03af\u03b1 \u03b5\u1f34\u03b7 \u2014; ", "ref": "Phd.67.e", "refLink": "Phd/67/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f74 ... \u1f21 \u1f19\u03bb\u03bb\u03ac\u03c2. ", "ref": "Phd.78.a", "refLink": "Phd/78/a"}, {"note": ".  \u1f21 \u03c0\u03bf\u03bb\u03bb\u1f74 \u03c3\u03c0\u03bf\u03c5\u03b4\u1f74 ... \u1f00\u03bc\u03c6\u03b9\u03c2\u03b2\u03ae\u03c4\u03b7\u03c3\u03b9\u03c2 \u03b3\u03af\u03b3\u03bd\u03b5\u03c4\u03b1\u03b9. ", "ref": "Phlb.15.a", "refLink": "Phlb/15/a"}, {"note": ".  \u1f45\u03b8\u03b5\u03bd ... \u03c4\u1ff7 \u03ba\u03c5\u03bf\u1fe6\u03bd\u03c4\u03b9 ... \u03c0\u03bf\u03bb\u03bb\u1f74, \u1f21 \u03c0\u03c4\u03cc\u03b7\u03c3\u03b9\u03c2 \u03b3\u03ad\u03b3\u03bf\u03bd\u03b5 \u03c0\u03b5\u03c1\u1f76 \u03c4\u1f78 \u03ba\u03b1\u03bb\u03cc\u03bd. ", "ref": "Symp.206.d", "refLink": "Symp/206/d"}, {"note": ".  \u03c4\u1f78 \u03c4\u03bf\u1fe6 \u1f08\u03bd\u03b1\u03be\u03b1\u03b3\u03bf\u03c1\u03bf\u03c5 \u1f02\u03bd \u03c0\u03bf\u03bb\u1f7a \u1f26\u03bd. ", "ref": "Grg.465.d", "refLink": "Grg/465/d"}, {"note": ".  \u03c0\u03b5\u03c1\u1f76 \u1f15\u03ba\u03b1\u03c3\u03c4\u03bf\u03bd \u03c4\u1ff6\u03bd \u03b5\u1f30\u03b4\u1ff6\u03bd \u03c0\u03bf\u03bb\u1f7a \u03bc\u03ad\u03bd \u1f10\u03c3\u03c4\u03b9 \u03c4\u1f78 \u1f45\u03bd cet. ", "ref": "Soph.256.e", "refLink": "Soph/256/e"}, {"note": ".  \u03c4\u1ff6\u03bd \u1f45\u03c0\u03bb\u03c9\u03bd \u03c4\u1f78 \u03c0\u03bf\u03bb\u03cd. ", "ref": "Plt.288.b", "refLink": "Plt/288/b"}, {"note": ".  \u03c4\u1f78 \u03c0\u03bf\u03bb\u1f7a \u03bb\u03af\u03b1\u03bd \u03ba\u03b1\u1f76 \u1f04\u03c0\u03b5\u03b9\u03c1\u03bf\u03bd \u1f00\u03c6\u03b5\u03af\u03bb\u03b5\u03c4\u03bf. ", "ref": "Phlb.26.a", "refLink": "Phlb/26/a"}, {"note": ".  \u1f04\u03c0\u03b5\u03b9\u03c1\u03bf\u03bd ... \u03c0\u03bf\u03bb\u03cd. ", "ref": "Phlb.30.c", "refLink": "Phlb/30/c"}, {"note": ".  \u03bf\u1f36\u03c3\u03b8\u02bc \u1f45\u03c4\u03b9 \u03c0\u03bf\u03af\u03b7\u03c3\u03af\u03c2 \u1f10\u03c3\u03c4\u03af \u03c4\u03b9 \u03c0\u03bf\u03bb\u03cd. ", "ref": "Symp.205.b", "refLink": "Symp/205/b"}, {"note": " (vii).  \u1f10\u03c0\u03b5\u03b9\u03b4\u1f74 \u03c0\u03bf\u03bb\u1f7a \u03c4\u1f78 \u1f14\u03c1\u03b3\u03bf\u03bd. ", "ref": "Resp.530.d", "refLink": "Resp/530/d"}, {"note": " (ii).  \u03c4\u1fc6\u03c2 \u1f51\u03c0\u1f78 \u03c4\u03bf\u1fe6 \u03c0\u03bf\u03bb\u03bb\u03bf\u1fe6 \u03c7\u03c1\u03cc\u03bd\u03bf\u03c5 \u03b5\u1f51\u03c1\u03b7\u03bc\u03ad\u03bd\u03b7\u03c2. ", "ref": "Resp.376.e", "refLink": "Resp/376/e"}, {"note": " (viii).  \u03bb\u03ac\u03bc\u03c0\u03c1\u03b1\u03c2 \u03bc\u03b5\u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u03bf\u1fe6 \u03c7\u03bf\u03c1\u03bf\u1fe6 \u03ba\u03b1\u03c4\u03ac\u03b3\u03bf\u03c5\u03c3\u03b9\u03bd \u1f10\u03c3\u03c4\u03b5\u03c6\u03b1\u03bd\u03c9\u03bc\u03ad\u03bd\u03b1\u03c2. ", "ref": "Resp.560.e", "refLink": "Resp/560/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1fc6\u03c2 \u1f02\u03bd \u03b5\u1f50\u1f20\u03b8\u03b5\u03af\u03b1\u03c2 \u03b3\u03ad\u03bc\u03bf\u03b9. ", "ref": "Phdr.275.c", "refLink": "Phdr/275/c"}, {"note": ".  \u03c4\u1f78 ... \u03c4\u1fc6\u03c2 \u03c0\u03bf\u03bb\u03bb\u1fc6\u03c2 \u03ba\u03b1\u1f76 \u03c0\u03b1\u03bd\u03c4\u03bf\u03b4\u03b1\u03c0\u1fc6\u03c2 \u1f00\u03b3\u03bd\u03bf\u03af\u03b1\u03c2 \u03c0\u03ac\u03b8\u03bf\u03c2. ", "ref": "Soph.228.e", "refLink": "Soph/228/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1fc6\u03c2 \u03bf\u1f55\u03c3\u03b7\u03c2 \u03c4\u1fc6\u03c2 \u03ba\u03bf\u03c3\u03bc\u03b7\u03c4\u03b9\u03ba\u1fc6\u03c2. ", "ref": "Plt.282.a", "refLink": "Plt/282/a"}, {"note": ".  \u03c4\u03b1\u03cd\u03c4\u03b7\u03c2 ... \u03c0\u03bf\u03bb\u03bb\u1fc6\u03c2 \u03bf\u1f54\u03c3\u03b7\u03c2 \u03ba\u03b1\u1f76 \u03c0\u03b1\u03bd. \u03c4\u03bf\u03af\u03b1\u03c2 ... \u03bc\u03ac\u03c7\u03b7\u03c2. ", "ref": "Phlb.15.d", "refLink": "Phlb/15/d"}, {"note": ".  \u03c4\u03bf\u1fe6 \u03c0\u03bf\u03bb\u03bb\u03bf\u1fe6 \u03ba\u03b1\u1f76 \u03bc\u1f74 \u03c4\u03bf\u03b9\u03bf\u03cd\u03c4\u03bf\u03c5 (\u03ba\u03b1\u03b8\u03b1\u03c1\u03bf\u1fe6) \u03b4\u03b9\u03b1\u03c6\u03ad\u03c1\u03b5\u03b9\u03bd. ", "ref": "Phlb.58.c", "refLink": "Phlb/58/c"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u03bf\u1fe6 \u03b4\u03b5\u1fd6. ", "ref": "Symp.203.c", "refLink": "Symp/203/c"}, {"note": ".  \u03bc\u03b5\u03b3\u03ac\u03bb\u1fc3 ... \u1f04\u03bb\u03bb\u1fc3 \u03ba\u03b1\u1f76 \u03c0\u03bf\u03bb\u03bb\u1fc7 \u03be\u03c5\u03bc\u03c0\u03b9\u03c0\u03c4\u03bf\u03cd\u03c3\u03b1\u03c2 \u03c0\u03bf\u03bd\u03b7\u03c1\u03af\u1fb3. ", "ref": "Phlb.41.a", "refLink": "Phlb/41/a"}, {"note": " (vii).  (\u1f10\u03bd) \u03c0\u03bf\u03bb\u03bb\u1ff7 ... \u03ba\u03b1\u1f76 \u03bc\u03b5\u03b3\u03ac\u03bb\u1ff3 \u03b3\u03ad\u03bd\u03b5\u03b9 (\u03b5\u1f34 \u03c4\u03b9\u03c2 \u03c4\u03c1\u03b1\u03c6\u03b5\u03af\u1fc3). ", "ref": "Resp.537.e", "refLink": "Resp/537/e"}, {"note": ".  \u1f67\u03bd ... \u03c0\u03ad\u03c1\u03b9 \u03c4\u1f78\u03bd \u03c0\u03bf\u03bb\u1f7a\u03bd \u03bb\u03cc\u03b3\u03bf\u03bd \u1f10\u03c0\u03bf\u03b9\u03b5\u1fd6\u03c4\u03bf \u1f08\u03bd\u03b1\u03be\u03b1\u03b3\u03cc\u03c1\u03b1\u03c2 (germ. wor\u00fcber An. so viel gesprochen hat). ", "ref": "Phdr.270.a", "refLink": "Phdr/270/a"}, {"note": ".  \u1f45\u03c4\u03b9 ... \u1f10\u03b3\u1f7c \u03c0\u03ac\u03bb\u03b1\u03b9 \u03c0\u03bf\u03bb\u1f7a\u03bd \u03bb\u03cc\u03b3\u03bf\u03bd \u03c0\u03b5\u03c0\u03bf\u03af\u03b7\u03bc\u03b1\u03b9. ", "ref": "Phd.115.d", "refLink": "Phd/115/d"}, {"note": ".  \u03b4\u03b5\u1fd6 ... \u03c0\u03b5\u03c1\u1f76 \u03c4\u1fc6\u03c2 \u1f00\u03c1\u03c7\u1fc6\u03c2 \u03c0\u03b1\u03bd\u03c4\u1f78\u03c2 \u03c0\u03c1\u03ac\u03b3\u03bc\u03b1\u03c4\u03bf\u03c2 \u03c0\u03b1\u03bd\u03c4\u1f76 \u1f04\u03bd\u03b4\u03c1\u03b9 \u03c4\u1f78\u03bd \u03c0\u03bf\u03bb\u1f7a\u03bd \u03bb\u03cc\u03b3\u03bf\u03bd \u03b5\u1f36\u03bd\u03b1\u03b9 \u03ba\u03b1\u1f76 \u03c4\u1f74\u03bd \u03c0\u03bf\u03bb\u03bb\u1f74\u03bd \u03c3\u03ba\u03ad\u03c8\u03b9\u03bd. ", "ref": "Cra.436.d", "refLink": "Cra/436/d"}, {"note": ".  \u03c0\u03bf\u03bb\u1f7a\u03bd \u03c8\u03cc\u03c6\u03bf\u03bd \u03c0\u03b1\u03c1\u03b1\u03c3\u03c7\u03b5\u1fd6\u03bd. ", "ref": "Symp.212.c", "refLink": "Symp/212/c"}, {"note": " (v).  \u03c0\u03c1\u1f78\u03c2 \u03c4\u1f78 \u03c0\u03b5\u03af\u03b8\u03b5\u03b9\u03bd \u03c4\u03b5 \u03ba\u03b1\u1f76 \u1f15\u03bb\u03ba\u03b5\u03b9\u03bd \u03c4\u1f78\u03bd \u03c0\u03bf\u03bb\u1f7a\u03bd \u03bb\u03b5\u03ce\u03bd. ", "ref": "Resp.458.d", "refLink": "Resp/458/d"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f74\u03bd (\u1f41\u03b4\u03cc\u03bd). ", "ref": "Phdr.272.c", "refLink": "Phdr/272/c"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f74\u03bd ... \u03ba\u03b1\u1f76 \u03c4\u1f74\u03bd \u03bc\u03b5\u03b3\u03af \u03c3\u03c4\u03b7\u03bd \u1f34\u03c3\u03c7\u03b5\u03b9\u03bd \u1f14\u03c7\u03b8\u03c1\u03b1\u03bd \u03ba\u03b1\u1f76 \u03c3\u03c4\u03ac\u03c3\u03b9\u03bd. ", "ref": "Plt.308.b", "refLink": "Plt/308/b"}, {"note": ".  \u03bc\u03b5\u03b3\u03ac\u03bb\u1f74\u03bd ... \u03c4\u1f74\u03bd \u03c0\u03bf\u03bb\u03bb\u03ae\u03bd (\u03c6\u03c9\u03bd\u03ae\u03bd). ", "ref": "Ti.67.c", "refLink": "Ti/67/c"}, {"note": ".  \u03c0\u03bf\u03bb\u1f7a \u03c7\u03c1\u03c5\u03c3\u03af\u03bf\u03bd. ", "ref": "Phdr.228.a", "refLink": "Phdr/228/a"}, {"note": ".  \u03c0\u03bf\u03bb\u03c5 \u1f14\u03c1\u03b3\u03bf\u03bd ... \u03c0\u03c1\u03bf\u03c3\u03c4\u03ac\u03c4\u03c4\u03b5\u03b9\u03c2 \u1f61\u03c2 \u03c4\u03b7\u03bb\u03b9\u03ba\u1ff3\u03b4\u03b5. ", "ref": "Prm.136.d", "refLink": "Prm/136/d"}, {"note": ".  \u03c4\u1f78 ... \u03c0\u03bf\u03bb\u1f7a \u03b1\u1f50\u03c4\u03bf\u1fe6 \u1f51\u03c0\u03bf\u03bb\u03b1\u03bc\u03b2\u03b1\u03bd\u03bf\u1fe6\u03c3\u03b9 \u03c4\u03bf\u03b9\u03bf\u1fe6\u03c4\u03cc\u03bd \u03c4\u03b9 \u03b5\u1f36\u03bd\u03b1\u03b9. ", "ref": "Cra.412.d", "refLink": "Cra/412/d"}, {"note": ".  \u03c0\u03c1\u1f78\u03c2 \u03c0\u03bf\u03bb\u1f7a ... \u03c4\u1f78 \u03ba\u03b1\u03bb\u03cc\u03bd. ", "ref": "Symp.210.c", "refLink": "Symp/210/c"}, {"note": ".  \u1f10\u03c0\u1f76 \u03c4\u1f78 \u03c0\u03bf\u03bb\u1f7a \u03c0\u03ad\u03bb\u03b1\u03b3\u03bf\u03c2 \u03c4\u03b5\u03c4\u03c1\u03ac\u03bc\u03bc\u03b5\u03bd\u03bf\u03c2 \u03c4\u03bf\u1fe6 \u03ba\u03b1\u03bb\u03bf\u1fe6. ", "ref": "Symp.210.d", "refLink": "Symp/210/d"}, {"note": " (vii).  \u03c4\u1f78 \u03b4\u1f72 \u03c0\u03bf\u03bb\u03c5 \u03b1\u1f50\u03c4\u1fc6\u03c2 \u03ba\u03b1\u1f76 \u03c0\u03bf\u1fe4\u1fe5\u03c9\u03c4\u03ad\u03c1\u03c9 \u03c0\u03c1\u03bf\u1fd6\u1f78\u03bd \u03c3\u03ba\u03bf\u03c0\u03b5\u1fd6\u03c3\u03b8\u03b1\u03b9 \u03b4\u03b5\u1fd6. ", "ref": "Resp.526.d", "refLink": "Resp/526/d"}, {"note": ".  \u03c4\u03b1\u03cd\u03c4\u1fc3 \u03c4\u1f78 \u03c0\u03bf\u03bb\u03c5 \u03c4\u1fc6\u03c2 \u03c0\u03bf\u03c1\u03b5\u03af\u03b1\u03c2 \u03b7\u03bc\u1fd6\u03bd \u1f14\u03b4\u03bf\u03c3\u03b1\u03bd. ", "ref": "Ti.45.a", "refLink": "Ti/45/a"}, {"note": " (i).  \u03c4\u1f78 ... \u03c0\u03bf\u03bb\u1f7a \u03c0\u03b1\u03c1\u2019 \u1f21\u03bc\u1fd6\u03bd \u03b3\u03b9\u03b3\u03bd\u03cc\u03bc\u03b5\u03bd\u03bf\u03bd. ", "ref": "Leg.633.b", "refLink": "Leg/633/b"}, {"note": ".  \u1f61\u03c2 \u03c4\u1f78 \u03c0\u03bf\u03bb\u1f7a \u03c4\u03bf\u1fe6\u03c4\u03bf \u03b4\u03b5\u03b9\u03bd\u03bf\u1fd6\u03bd \u1f44\u03bd\u03c4\u03bf\u03b9\u03bd (vos huius rei tam late patentis peritos esse). ", "ref": "Euthd.273.e", "refLink": "Euthd/273/e"}, {"note": ".  \u03ba\u03b1\u1f76 \u1f04\u03bb\u03bb\u03bf\u03b9 \u03c0\u03bf\u03bb\u03bb\u03bf\u1f76 \u03ba\u03b1\u1f76 \u03c3\u03bf\u03c6\u03bf\u03af. ", "ref": "Prt.314.c", "refLink": "Prt/314/c"}, {"note": ".  \u03bf\u1f37\u03c2 \u03c0\u03b9\u03c3\u03c4\u03b5\u03cd\u03bf\u03bd\u03c4\u03b5\u03c2 ... \u03bf\u1f31 \u03c0\u03bf\u03bb\u03bb\u03bf\u1f76 \u03ba\u03c1\u03af\u03bd\u03bf\u03c5\u03c3\u03b9 cet. ", "ref": "Phlb.67.b", "refLink": "Phlb/67/b"}, {"note": " (ii).  \u1f61\u03c2 \u03bf\u1f31 \u03c0\u03bf\u03bb\u03bb\u03bf\u1f76 \u03bb\u03ad\u03b3\u03bf\u03c5\u03c3\u03b9\u03bd. ", "ref": "Resp.379.c", "refLink": "Resp/379/c"}, {"note": ".  \u1f10\u03c0\u03b5\u1f76 \u03bf\u1f35 \u03b3\u03b5 \u03c0\u03bf\u03bb\u03bb\u03bf\u1f76 ... \u03bf\u1f50\u03b4\u1f72\u03bd \u03b1\u1f30\u03c3\u03b8\u03ac\u03bd\u03bf\u03bd\u03c4\u03b1\u03b9. ", "ref": "Prt.317.a", "refLink": "Prt/317/a"}, {"note": ".  \u03bf\u1f31 \u1f00\u03c3\u03b8\u03b5\u03bd\u03b5\u1fd6\u03c2 \u1f04\u03bd\u03b8\u03c1\u03c9\u03c0\u03bf\u03b9 ... \u03ba\u03b1\u1f76 \u03bf\u1f31 \u03c0\u03bf\u03bb\u03bb\u03bf\u03af. ", "ref": "Grg.483.b", "refLink": "Grg/483/b"}, {"note": " (iii).  \u03b4\u03af\u03bf\u03c4\u03b9 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03ba\u03b1\u1f76 \u1f00\u03bd\u03cc\u03c3\u03b9\u03b1 ... \u03b3\u03ad\u03b3\u03bf\u03bd\u03b5. ", "ref": "Resp.416.e", "refLink": "Resp/416/e"}, {"note": ".  \u03bd\u03bf\u03c5\u03b8\u03b5\u03c4\u03b5\u1fd6\u03c4\u03b1\u03b9 ... \u1f51\u03c0\u1f78 \u03c4\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd cet. ", "ref": "Phdr.249.d", "refLink": "Phdr/249/d"}, {"note": ".  \u1f51\u03c0\u1f78 \u03c4\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd \u1f00\u03bd\u03b8\u03c1\u03ce\u03c0\u03c9\u03bd. ", "ref": "Grg.475.d", "refLink": "Grg/475/d"}, {"note": ".  \u03c0\u03c1\u1f78\u03c2 \u03c4\u1f70\u03c2 \u03c4\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd \u03b4\u03cc\u03be\u03b1\u03c2. ", "ref": "Plt.306.a", "refLink": "Plt/306/a"}, {"note": ".  \u1f21\u03c4\u03c4\u03b7\u03bc\u03ad\u03bd\u1ff3 \u03c4\u1fc6\u03c2 \u03c4\u03b9\u03bc\u1fc6\u03c2 \u03c4\u1fc6\u03c2 \u1f51\u03c0\u1f78 \u03c4\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd. ", "ref": "Symp.216.b", "refLink": "Symp/216/b"}, {"note": ".  \u03bf\u1f54\u03c4\u03b5 \u03c4\u1ff7 \u1f02\u03bd \u03b4\u03b9\u03ac\u03c6\u03b5\u03c1\u03bf\u03b9 ... \u03c4\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd \u1f00\u03bd\u03b8\u03c1\u03ce\u03c0\u03c9\u03bd. ", "ref": "Euthphr.5.a", "refLink": "Euthphr/5/a"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd \u03ba\u03b1\u1f76 \u03bc\u03b1\u03ba\u03b1\u03c1\u03af\u03c9\u03bd ... \u03bc\u03b5\u03c4\u03b5\u03af\u03bb\u03b7\u03c6\u03b5\u03bd. ", "ref": "Plt.269.e", "refLink": "Plt/269/e"}, {"note": ".  \u03c4\u03b9\u03b8\u03ad\u03bc\u03b5\u03b8\u03b1 ... \u03c4\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd \u03c4\u03c1\u03b9\u03b3\u03ce\u03bd\u03c9\u03bd \u03ba\u03ac\u03bb\u03bb\u03b9\u03c3\u03c4\u03bf\u03bd \u1f15\u03bd. ", "ref": "Ti.54.a", "refLink": "Ti/54/a"}, {"note": ".  \u03b4\u03bf\u03ba\u03b5\u1f76 ... \u03c4\u03bf\u1fd6\u03c2 \u03c0\u03bf\u03bb\u03bb\u03bf\u1fd6\u03c2 \u1f00\u03bd\u03b8\u03c1\u03ce\u03c0\u03bf\u03b9\u03c2 cet. ", "ref": "Phd.65.a", "refLink": "Phd/65/a"}, {"note": " (i).  \u03be\u03c5\u03bd\u03b5\u03c3\u03cc\u03bc\u03b5\u03b8\u03b1 ... \u03c0\u03bf\u03bb\u03bb\u03bf\u1fd6\u03c2 \u03c4\u1ff6\u03bd \u03bd\u03ad\u03c9\u03bd \u03b1\u1f50\u03c4\u03bf\u03b8\u03b9. ", "ref": "Resp.328.a", "refLink": "Resp/328/a"}, {"note": " (v).  \u03c0\u03b1\u03c1\u1fb6 \u03b4\u03cc\u03be\u03b1\u03bd \u03c0\u03bf\u03bb\u03bb\u1f70 \u03c0\u03bf\u03bb\u03bb\u03bf\u1fd6\u03c2 \u03b4\u1f74 \u1f10\u03b3\u03ad\u03bd\u03b5\u03c4\u03bf. ", "ref": "Resp.467.d", "refLink": "Resp/467/d"}, {"note": " (x).  \u03c0\u03bf\u03bb\u03bb\u03bf\u1fd6\u03c3\u03b9 ", "ref": "Leg.888.c", "refLink": "Leg/888/c"}, {"note": ".  \u03bc\u1f74 \u03bf\u1f50 \u03c0\u03bf\u03bd\u03b5\u1fd6\u03bd \u03b1\u1f50\u03c4\u1f74\u03bd \u1f10\u03bd \u03c4\u03b1\u1fd6\u03c2 \u03c0\u03bf\u03bb\u03bb\u03b1\u1fd6\u03c2 \u03b3\u03b5\u03bd\u03ad\u03c3\u03b5\u03c3\u03b9. ", "ref": "Phd.88.a", "refLink": "Phd/88/a"}, {"note": " (vi).  \u03bf\u1f50\u03ba \u1f10\u03c0\u03b9\u03bc\u03ad\u03bd\u03bf\u03b9 \u1f10\u03c0\u1f76 \u03c4\u03bf\u1fd6\u03c2 \u03b4\u03bf\u03be\u03b1\u03b6\u03bf\u03bc\u03ad\u03bd\u03bf\u03b9\u03c2 \u03b5\u1f36\u03bd\u03b1t \u03c0\u03bf\u03bb\u03bb\u03bf\u1fd6\u03c2 \u1f11\u03ba\u03ac\u03c3\u03c4\u03bf\u03b9\u03c2. ", "ref": "Resp.490.b", "refLink": "Resp/490/b"}, {"note": ".  \u1f51\u03bc\u1ff6\u03bd \u03c4\u03bf\u1f7a\u03c2 \u03c0\u03bf\u03bb\u03bb\u03bf\u03cd\u03c2. ", "ref": "Symp.176.a", "refLink": "Symp/176/a"}, {"note": " (i).  \u03bf\u1f37\u03bc\u03b1\u03af \u03c3\u03bf\u03c5 \u03c4\u03bf\u1f7a\u03c2 \u03c0\u03bf\u03bb\u03bb\u03bf\u1f7a\u03c2 ... \u03bf\u1f50\u03ba \u1f00\u03c0\u03bf\u03b4\u03ad\u03c7\u03b5\u03c3\u03b8\u03b1\u03b9. ", "ref": "Resp.369.e", "refLink": "Resp/369/e"}, {"note": " (iii).  \u1f00\u03b3\u03ad\u03bb\u03b1\u03c2 \u1f00\u03bd\u03b4\u03c1\u1ff6\u03bd \u03c4\u03b5 \u03ba\u03b1\u1f76 \u1f04\u03bb\u03bb\u03c9\u03bd \u03c0\u03bf\u03bb\u03bb\u1ff6\u03bd \u03c0\u03bf\u03bb\u03bb\u1f70\u03c2 \u1f10\u03ba\u03c4\u1fb6\u03c4\u03bf. ", "ref": "Leg.694.e", "refLink": "Leg/694/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 \u1f51\u03c0\u1f72\u03c1 \u1f10\u03bc\u03bf\u1fe6 \u03b5\u1f36\u03c0\u03b5. ", "ref": "Prt.309.b", "refLink": "Prt/309/b"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 ... \u03ba\u03b1\u1f76 \u03ba\u03b1\u03bb\u1f70 ... \u03b5\u1f30\u03c1\u03b3\u03ac\u03c3\u03b1\u03bd\u03c4\u03bf. ", "ref": "Phdr.244.b", "refLink": "Phdr/244/b"}, {"note": ".  \u03ba\u03b1\u1f76 \u1f15\u03bd \u03ba\u03b1\u1f76 \u03c0\u03bf\u03bb\u03bb\u03ac. ", "ref": "Phdr.261.d", "refLink": "Phdr/261/d"}, {"note": " (x).  \u03c0\u03bf\u03bb\u03bb\u1f70 \u03ba\u03b1\u1f76 \u1f14\u03c1\u03b3\u03b1. ", "ref": "Resp.599.b", "refLink": "Resp/599/b"}, {"note": " (x).  \u1f04\u03bb\u03bb\u03b1 ... \u03c0\u03bf\u03bb\u03bb\u03ac \u03c4\u03b5 \u03ba\u03b1\u1f76 \u1f00\u03bd\u03cc\u03c3\u03b9\u03b1 \u03b5\u1f30\u03c1\u03b3\u03b1\u03c3\u03bc\u03ad\u03bd\u03bf\u03c2. ", "ref": "Resp.615.d", "refLink": "Resp/615/d"}, {"note": ".  \u1f43\u03c2 (\u03c7\u03c1\u03cc\u03bd\u03bf\u03c2) \u03b4\u1f74 \u03b4\u03bf\u03ba\u03b5\u1fd6 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03ba\u03b1\u03bb\u1ff6\u03c2 \u03b2\u03b1\u03c3\u03b1\u03bd\u03af\u03b6\u03b5\u03b9\u03bd. ", "ref": "Symp.184.a", "refLink": "Symp/184/a"}], "start": "\u03c0\u03bf\u03bb\u03bb\u03ae, \u03c0\u03bf\u03bb\u03cd, multus, multiplex, varius; largus; amplus, late patens; magnus et vehemens; longus; c. art. plurimus, maximus (etiam noti vel iam significati quid indicat, ita ut part. germ. so exprimi possit); pl. \u03bf\u1f31 \u03c0\u03bf\u03bb\u03bb\u03bf\u03af plurimi, plerique; etiam multitudo, vulgus. "}}, {"subList": [], "text": {"identifier": "II", "keyPassageList": [], "refList": [], "start": "\u2014 Compar. \u03c0\u03bb\u03b5\u03af\u03c9\u03bd et superl. \u03c0\u03bb\u03b5\u1fd6\u03c3\u03c4\u03bf\u03c2 v. p. 111 sq. "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u03ba\u03b1\u03c4\u03b1\u03b4\u03b1\u03c1\u03b8\u03b5\u1fd6\u03bd \u03c0\u03ac\u03bd\u03c5 \u03c0\u03bf\u03bb\u03cd); crebro (", "ref": "Symp.223.c", "refLink": "Symp/223/c"}, {"note": " (viii).  \u03bb\u03ad\u03b3\u03b5\u03c4\u03b1\u03b9 \u03b3\u1f70\u03c1 \u03b4\u1f74 ... \u03ba\u03b1\u1f76 \u03c0\u03bf\u03bb\u1f7a \u03c4\u03bf\u1fe6\u03c4\u03bf \u03c4\u1f78 \u1fe5\u1fc6\u03bc\u03b1); longe (", "ref": "Resp.562.c", "refLink": "Resp/562/c"}, {"note": " (x).  \u03c0\u03c1\u03bf\u1fd6\u03cc\u03bd\u03c4\u03b5\u03c2 ... \u03c4\u1fc6\u03c2 \u1f00\u03c1\u03c7\u1fc6\u03c2 \u03bf\u1f50 \u03c0\u03bf\u03bb\u03cd). ", "ref": "Leg.886.c", "refLink": "Leg/886/c"}], "start": " Etiam diu ( "}}, {"subList": [], "text": {"identifier": "B", "keyPassageList": [], "refList": [{"note": ".  \u03c0\u03bf\u03bb\u03c5 \u03bc\u03b5\u03af\u03b6\u03c9\u03bd \u03ba\u03af\u03bd\u03b4\u03c5\u03bd\u03bf\u03c2. ", "ref": "Prt.314.a", "refLink": "Prt/314/a"}, {"note": ".  \u03bd\u03b5\u03ce\u03c4\u03b5\u03c1\u03bf\u03c2 \u03c0\u03bf\u03bb\u03cd. ", "ref": "Symp.180.a", "refLink": "Symp/180/a"}, {"note": ".  \u03c0\u03bf\u03bb\u1f7a \u1f10\u03bd \u03c0\u03bb\u03b5\u03af\u03bf\u03bd\u03b9 \u1f00\u03c0\u03bf\u03c1\u03af\u1fb3 \u03b5\u1f30\u03bc\u03af. ", "ref": "Cra.413.c", "refLink": "Cra/413/c"}, {"note": " (ix).  \u03c0\u03bf\u03bb\u1f7a \u1f10\u03c0\u1f76 \u03b4\u03b5\u03b9\u03bd\u03bf\u03c4\u03ad\u03c1\u1ff3 \u1f40\u03bb\u03ad\u03b8\u03c1\u03c2 ", "ref": "Resp.590.a", "refLink": "Resp/590/a"}, {"note": " (v).  \u1f10\u03bb\u03ac\u03c4\u03c4\u03bf\u03c5\u03c2 \u03b4\u1f72 \u03c0\u03bf\u03bb\u03cd cet. ", "ref": "Leg.741.a", "refLink": "Leg/741/a"}, {"note": ".  \u1f26 \u03c0\u03bf\u03bb\u03cd \u03bc\u03bf\u03b9 \u03b4\u03b9\u1f70 \u03b2\u03c1\u03b1\u03c7\u03c5\u03c4\u03ad\u03c1\u03c9\u03bd ... \u03b5\u1f36\u03c0\u03b5\u03c2 \u1f02\u03bd \u03c4\u1f78 \u03ba\u03b5\u03c6\u03ac\u03bb\u03b1\u03b9\u03bf\u03bd cet. ", "ref": "Euthphr.14.b", "refLink": "Euthphr/14/b"}, {"note": ".  \u03c0\u03bf\u03bb\u03c5 \u1f14\u03c4\u03b9 \u03bc\u1fb6\u03bb\u03bb\u03bf\u03bd \u1f20\u03c1\u03c5\u03b8\u03c1\u03b9\u03ac\u03c3\u03b5\u03bd. Sic etiam \u03c0\u03bf\u03bb\u03bb\u1ff7. ", "ref": "Ly.204.c", "refLink": "Ly/204/c"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1ff7 \u03bc\u1fb6\u03bb\u03bb\u03bf\u03bd \u1f67\u03b4\u03b5 \u1f14\u03c7\u03b5\u03b9. ", "ref": "Phd.80.e", "refLink": "Phd/80/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1ff7 \u03bc\u03b5\u1fd6\u03b6\u03bf\u03bd. ", "ref": "Plt.274.e", "refLink": "Plt/274/e"}, {"note": ". ", "ref": "Plt.288.b", "refLink": "Plt/288/b"}, {"note": " (vii).  al. ", "ref": "Leg.807.c", "refLink": "Leg/807/c"}], "start": " Cum compar. multo, ut "}}, {"subList": [], "text": {"identifier": "C", "keyPassageList": [], "refList": [{"note": ".  \u1f65\u03c3\u03c4\u03b5 \u03c0\u03bf\u03bb\u03cd \u03bc\u03bf\u03b9 \u1f21\u03b4\u03b9\u03c3\u03c4\u03cc\u03bd \u1f10\u03c3\u03c4\u03b9\u03bd. ", "ref": "Prt.317.c", "refLink": "Prt/317/c"}, {"note": ".  \u03c0\u03bf\u03bb\u1f7a \u03ba\u03c1\u03ac\u03c4\u03b9\u03c3\u03c4\u03cc\u03bd \u1f10\u03c3\u03c4\u03b9\u03bd. ", "ref": "Phdr.228.c", "refLink": "Phdr/228/c"}, {"note": ".  \u03c0\u03bf\u03bb\u1f7a ... \u03bc\u03b5\u03b3\u03af\u03c3\u03c4\u03b7 cet. ", "ref": "Symp.209.c", "refLink": "Symp/209/c"}], "start": "Cum superl. longe. "}}, {"subList": [], "text": {"identifier": "D", "keyPassageList": [], "refList": [{"note": ". ", "ref": "Grg.477.c", "refLink": "Grg/477/c"}, {"note": " (iii). ", "ref": "Resp.387.e", "refLink": "Resp/387/e"}, {"note": " (iv). ", "ref": "Resp.421.d", "refLink": "Resp/421/d"}, {"note": " (v).  \u03ba\u03b1\u1f76 \u03c4\u03bf\u1fe6\u03c4\u03bf ... \u03c0\u03bf\u03bb\u03cd (i. q. \u03ba\u03b1\u1f76 \u03c4\u03bf\u1fe6\u03c4\u03bf \u1f14\u03c3\u03c4\u03b1\u03b9, \u03c0\u03bf\u03bb\u1f7a \u03b2\u03ad\u03bb\u03c4\u03b9\u03c3\u03c4\u03b1\u03b9 \u1f14\u03c3\u03bf\u03bd\u03c4\u03b1\u03b9). ", "ref": "Resp.456.e", "refLink": "Resp/456/e"}, {"note": " (viii).  \u03ba\u03b1\u1f76 \u03c0\u03bf\u03bb\u03cd \u03b3\u03b5, \u1f14\u03c6\u03b7. ", "ref": "Resp.556.b", "refLink": "Resp/556/b"}], "start": " Hinc respons. \u03c0\u03bf\u03bb\u03cd \u03b3\u03b5 anteced. compar. vel superlat. "}}], "text": {"identifier": "III", "keyPassageList": [], "refList": [{"note": ".  \u03c0\u03ac\u03bd\u03c5 \u03c0\u03bf\u03bb\u1f7a \u03c0\u03b5\u03c1\u03b9\u03ad\u03c3\u03bf\u03bc\u03b1\u03b9), valde (", "ref": "Alc1.119.c", "refLink": "Alc1/119/c"}, {"note": ".  \u03bf\u1f55\u03c4\u03c9 \u03c0\u03bf\u03bb\u1f7a \u03b4\u03b9\u03b1\u03c6\u03ad\u03c1\u03bf\u03bd\u03c4\u03b1. ", "ref": "Phlb.34.e", "refLink": "Phlb/34/e"}, {"note": ". ", "ref": "Phlb.58.a", "refLink": "Phlb/58/a"}, {"note": " (vi).  \u03bf\u1f50 \u03c0\u03cc\u03bb\u03c5 \u03c4\u03b9 \u03b4\u03b9\u03b1\u03c6\u03ad\u03c1\u03b5\u03b9\u03bd). ", "ref": "Resp.484.d", "refLink": "Resp/484/d"}], "start": "\u2014 Adverbiorum instar ponuntur \u03c0\u03bf\u03bb\u03cd, multum (ut "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u03c4\u1f78 \u03bc\u1f72\u03bd \u03c0\u03bf\u03bb\u1f7a \u03be\u03ad\u03bd\u03bf\u03b9 \u1f10\u03c6\u03b1\u03af\u03bd\u03bf\u03bd\u03c4\u03bf. ", "ref": "Prt.315.a", "refLink": "Prt/315/a"}, {"note": ".  \u1f21 \u03c4\u1f78 \u03c0\u03bf\u03bb\u1f7a \u03b5\u1f30\u03b8\u03af\u03c3\u03bc\u03b5\u03b8\u03b1 \u03c6\u03ac\u03bd\u03b1\u03b9 cet. ", "ref": "Tht.165.a", "refLink": "Tht/165/a"}, {"note": " (iii).  \u03c4\u1f78 \u03bc\u1f72\u03bd \u03c0\u03bf\u03bb\u1f7a \u1f41\u03bc\u03bf\u03af\u03bf\u03c5\u03c2 \u1f02\u03bd \u1f51\u03bc\u1fd6\u03bd \u03b1\u1f50\u03c4\u03bf\u1fd6\u03c2 \u03b3\u03b5\u03bd\u03bd\u1ff7\u03c4\u03b5. ", "ref": "Resp.415.a", "refLink": "Resp/415/a"}, {"note": " (vii).  al. ", "ref": "Resp.540.b", "refLink": "Resp/540/b"}], "start": "Cum art. (\u03c4\u1f78 \u03c0\u03bf\u03bb\u03cd), magnam vel maximam partem. "}}, {"subList": [], "text": {"identifier": "B", "keyPassageList": [], "refList": [{"note": ".  \u1f61\u03c2 \u03c4\u1f78 \u03c0\u03bf\u03bb\u1f7a \u03c7\u03b1\u03c5\u03bd\u03cc\u03c4\u03b5\u03c1\u03bf\u03bd. ", "ref": "Soph.227.b", "refLink": "Soph/227/b"}, {"note": ". ", "ref": "Plt.307.c", "refLink": "Plt/307/c"}, {"note": ". ", "ref": "Cra.427.a", "refLink": "Cra/427/a"}, {"note": " (i). ", "ref": "Resp.330.c", "refLink": "Resp/330/c"}, {"note": " (viii). ). ", "ref": "Resp.554.e", "refLink": "Resp/554/e"}], "start": "Similiter \u1f61\u03c2 \u03c4\u1f78 \u03c0\u03bf\u03bb\u03cd ( "}}, {"subList": [], "text": {"identifier": "C", "keyPassageList": [], "refList": [{"note": ". ", "ref": "Plt.294.e", "refLink": "Plt/294/e"}, {"note": ". ", "ref": "Plt.295.a", "refLink": "Plt/295/a"}, {"note": " (ii). ", "ref": "Resp.377.b", "refLink": "Resp/377/b"}, {"note": " (vii). ", "ref": "Leg.792.b", "refLink": "Leg/792/b"}, {"note": " (ix). ", "ref": "Leg.875.d", "refLink": "Leg/875/d"}], "start": "Cum praepos. \u1f61\u03c2 \u1f10\u03c0\u1f76 \u03c4\u1f78 \u03c0\u03bf\u03bb\u03cd. "}}], "text": {"identifier": "IV", "keyPassageList": [], "refList": [{"note": ".  \u03c4\u03bf\u1fe6 \u1f04\u03bd\u03b5\u03b9\u03bd \u1f10\u03c0\u1f76 \u03c0\u03bf\u03bb\u03cd (germ. veit kommen). ", "ref": "Cra.415.a", "refLink": "Cra/415/a"}], "start": "\u2014 Cum praep. \u1f10\u03c0\u1f76 \u03c0\u03bf\u03bb\u03cd, longe. "}}, {"subList": [{"subList": [], "text": {"identifier": "A", "keyPassageList": [], "refList": [{"note": ".  \u03c4\u1f70 \u03bc\u1f72\u03bd \u1f10\u03c0\u2019 \u1f40\u03bb\u03af\u03b3\u03bf\u03bd, \u03c4\u1f70 \u03b4\u2019 \u1f10\u03c0\u1f76 \u03c0\u03bf\u03bb\u03bb\u03ac. \u2014 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u03ac, maximam partem, plerumque. ", "ref": "Soph.254.b", "refLink": "Soph/254/b"}, {"note": ".  \u03ba\u03b1\u1f76 \u03b3\u1f70\u03c1 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03a0\u03c1\u03c9\u03c4\u03b1\u03c7\u03cc\u03c1\u03b1\u03c2 \u1f14\u03bd\u03b4\u03bf\u03bd \u03b4\u03b9\u03b1\u03c4\u03c1\u03af\u03b2\u03b5\u03b9. ", "ref": "Prt.311.a", "refLink": "Prt/311/a"}, {"note": ".  \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03b4\u03b9\u03b7\u03bc\u03b5\u03c1\u03b5\u03cd\u03bf\u03bc\u03b5\u03bd \u03bc\u03b5\u03c4\u02bc \u03b1\u1f50\u03c4\u03bf\u1fe6. ", "ref": "Phd.59.d", "refLink": "Phd/59/d"}, {"note": ".  \u1f65\u03c3\u03c0\u03b5\u03c1 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03b5\u1f30\u03ce\u03b8\u03b5\u03b9. ", "ref": "Phd.86.d", "refLink": "Phd/86/d"}, {"note": ". ", "ref": "Tht.184.c", "refLink": "Tht/184/c"}, {"note": " (ii). ", "ref": "Resp.372.a", "refLink": "Resp/372/a"}, {"note": ". ", "ref": "Ti.25.e", "refLink": "Ti/25/e"}, {"note": " (i). ", "ref": "Leg.639.e", "refLink": "Leg/639/e"}, {"note": ".  \u03bf\u1f37\u03b1 \u03b4\u1f74 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u1f70 \u1f00\u03b5\u1f76 \u03bc\u03b5\u03c4\u02bc \u1f10\u03bc\u03bf\u1fe6 \u03be\u03ad\u03bd\u03bf\u03b9 \u03c4\u03b9\u03bd\u1f72\u03c2 \u1f15\u03c0\u03bf\u03bd\u03c4\u03b1\u03b9 (germ. ie gew\u1f43hnlich, folgen mir immer cet.). Idem fere est \u1f61\u03c2 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u03ac. ", "ref": "Menex.235.b", "refLink": "Menex/235/b"}, {"note": ".  \u1f61\u03c2 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u1f70 ... \u03c4\u03b1\u1fe6\u03c4\u03b1 \u1f10\u03bd\u03b1\u03bd\u03c4\u03af\u03b1 \u1f00\u03bb\u03bb\u03ae\u03bb\u03bf\u03b9\u03c2 \u1f10\u03c3\u03c4\u03af\u03bd. ", "ref": "Grg.482.e", "refLink": "Grg/482/e"}, {"note": ".  \u1f51\u03b3\u03b9\u03b1\u03af\u03bd\u03bf\u03bd\u03c4\u03b1 ... \u1f14\u1ff6\u03c3\u03b9\u03bd \u03bf\u1f31 \u1f30\u03b1\u03c4\u03c1\u03bf\u1f76 \u1f61\u03c2 \u03c4\u1f70 \u03c0\u03bf\u03bb\u03bb\u03ac. ", "ref": "Grg.505.a", "refLink": "Grg/505/a"}, {"note": ". ", "ref": "Tht.144.a", "refLink": "Tht/144/a"}, {"note": " (v). ", "ref": "Leg.743.b", "refLink": "Leg/743/b"}, {"note": " (xii).  ", "ref": "Leg.952.d", "refLink": "Leg/952/d"}], "start": "Cum praep. "}}], "text": {"identifier": "V", "keyPassageList": [], "refList": [{"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 \u1f02\u03bd \u03c0\u03b5\u03c1\u03b9\u03b5\u03c3\u03ba\u03ad\u03c8\u03c9. ", "ref": "Prt.313.a", "refLink": "Prt/313/a"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 \u03b5\u1f30\u03c0\u03cc\u03bd\u03c4\u03b1 \u03c7\u03b1\u03af\u03c1\u03b5\u03b9\u03bd \u03c4\u1ff7 \u1f00\u03bb\u03b7\u03b8\u03b5\u1fd6. ", "ref": "Phdr.272.e", "refLink": "Phdr/272/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 ... \u1f24\u03b4\u03b7 \u1f10\u03bd\u03c4\u03b5\u03c4\u03cd\u03c7\u03b7\u03ba\u03b1 \u03c4\u1ff7 \u1f00\u03bd\u03b4\u03c1\u03af. ", "ref": "Phd.61.c", "refLink": "Phd/61/c"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 \u1f02\u03bd \u1f25\u03bc\u03c5\u03bd\u03b5. ", "ref": "Tht.164.e", "refLink": "Tht/164/e"}, {"note": ".  \u03c0\u03bf\u03bb\u03bb\u1f70 \u03ba\u03ac\u03bc\u03c0\u03c4\u03bf\u03bd\u03c4\u03b1\u03b9. ", "ref": "Tht.173.b", "refLink": "Tht/173/b"}, {"note": ".  \u1f26 \u03bf\u1f56\u03bd \u03ba\u03b1\u1f76 \u03c4\u03c5\u03b3\u03c7\u03ac\u03bd\u03b5\u03b9 \u1f00\u03b5\u03af, \u1f22 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03ba\u03b1\u1f76 \u03b4\u03b9\u03b1\u03bc\u03b1\u03c1\u03c4\u03ac\u03bd\u03b5\u03b9 \u1f11\u03ba\u03ac\u03c3\u03c4\u03b7; ", "ref": "Tht.178.a", "refLink": "Tht/178/a"}, {"note": ".  \u1f15\u03c9\u03b8\u03b5\u03bd \u03b3\u1f70\u03c1 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03b1\u1f50\u03c4\u1ff7 \u03c3\u03c5\u03bd\u03b7\u03bd. ", "ref": "Cra.396.d", "refLink": "Cra/396/d"}, {"note": " (iii).  \u1f02\u03bd ... \u03b3\u03c5\u03bc\u03bd\u03b1\u03c3\u03c4\u03b9\u03ba\u1fc7 \u03c0\u03bf\u03bb\u03bb\u1f70 \u03c0\u03bf\u03bd\u1fc7. ", "ref": "Resp.411.c", "refLink": "Resp/411/c"}, {"note": " (iv). ", "ref": "Leg.709.a", "refLink": "Leg/709/a"}, {"note": " (xi).  \u03c0\u03bf\u03bb\u03bb\u1f70 \u03b6\u03b7\u03bc\u03b9\u03bf\u1fe6\u03bd\u03c4\u03b1\u03b9.. ", "ref": "Leg.916.d", "refLink": "Leg/916/d"}], "start": "\u2014 \u03c0\u03bf\u03bb\u03bb\u03ac, multum; valde; saepe et diu. "}}], "text": {"identifier": "\u03c0\u03bf\u03bb\u03cd\u03c2", "keyPassageList": [], "refList": [], "start": "\u03c0\u03bf\u03bb\u03cd\u03c2 "}}]
    # json_string_to_test = json.dumps(json_obj)
    # quotes = extract_quotes_from_json(json_string_to_test)
    # print(json.dumps(quotes, indent=2, ensure_ascii=False))