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
    text = text.replace("ϑ", "θ")
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
                                if "GREEK" in unicodedata.name(char, ""):
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
    # quote = "ἵνα ... πρὸς τῷ καϑ' ἡμέραν ἀναγκάζωνται εἶναι"
    # ctx = "ἵνα χρήματα εἰσφέροντες πένητες γιγνόμενοι πρὸς τῷ καθʼ ἡμέραν ἀναγκάζωνται εἶναι"
    # print(quote_match(quote, ctx))
    
    # with open('database_analysis/json_test.json', 'r') as f:
    #     json_obj = json.load(f)
        
    # json_string_to_test = json.dumps(json_obj)
    # quotes = extract_quotes_from_json(json_string_to_test)
    # print(json.dumps(quotes, indent=2, ensure_ascii=False)) 