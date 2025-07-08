import requests
from bs4 import BeautifulSoup
import re
from xml.etree import ElementTree

URL = "http://www.poesialatina.it/_ns/greek/testi/Plato/Spuria.html"
r = requests.get(URL)
soup = BeautifulSoup(r.content.decode('utf-8','ignore'), "html.parser")
relevant = soup.find_all(["h2", "p"])
splitted_sections = []
store = True
i = 0

while i < len(relevant):
    ele = relevant[i]

    if ele.name == "h2":
        last_h2_ind = i
        j = i + 1

        while j < len(relevant) and relevant[j].name == "h2":
            last_h2_ind = j
            j += 1
        
        out_list = []

        k = last_h2_ind + 1
        
        while k < len(relevant) and relevant[k].name != "h2":
            if relevant[k].name == "p":
                para = relevant[k]
                para = re.sub("<[^>]*>", "", str(para))
                # processing punctuation
                para = para.replace("]‑", "] ‑")
                para = para.replace(".", " . ")
                para = para.replace(";", " ; ")
                para = para.replace(",", " , ")
                para = para.replace("·", " · ")
                para = para.replace("\"", "")
                for ele in para.split():
                    out_list.append(ele)
            k += 1
        
        if out_list != []:
            splitted_sections.append(out_list)
        i = k 
    else:
        i += 1

headers = [
    "ΠΕΡΙ ΔΙΚΑΙΟΥ ΣΩΚΡΑΤΗΣ ΑΝΩΝΨΜΟΣ ΤΙΣ", 
    "ΠΕΡΙ ΑΡΕΤΗΣ ΣΩΚΡΑΤΗΣ ΙΠΠΟΤΡΟΦΟΣ",
    "ΔΗΜΟΔΟΚΟΣ",
    "ΣΙΣΥΦΟΣ ΣΩΚΡΑΤΗΣ ΣΙΣΥΦΟΣ",
    "ΕΡΥΞΙΑΣ",
    "ΑΞΙΟΧΟΣ ΣΩΚΡΑΤΗΣ ΚΛΕΙΝΙΑΣ ΑΞΙΟΧΟΣ"
]

eng_headers = [
    "Axiochus",
    "DeIusto",
    "DeVirtute",
    "Demodocus",
    "Sisyphus",
    "Eryxias"
]

# process headers

# i = 0
# for sec in splitted_sections:
#     sec.insert(0, headers[i])
#     i += 1

splitted_sections.insert(0, splitted_sections.pop())
headers.insert(0, headers.pop())

def create_xml_string(text_name):
    xml_string = f"""
    <TEI.2>
    
    <teiHeader status="new" type="text">
    
    <fileDesc>
    
    <titleStmt>
    <title>{text_name}</title>
    <author>Plato</author>
    <sponsor>Lexeis Project, Cornell University</sponsor>
            <principal>Jeffrey Rusten</principal>
            <respStmt>
            <resp>Prepared under the supervision of</resp>
            <name>Ethan Della Rocca</name>
            <name>Ayan Kohli</name>
            </respStmt>
    <funder n="org:Cornell">Cornell University</funder>
    </titleStmt>
    
    <extent>About 78Kb</extent>
    
    <publicationStmt>
            <publisher>Cornell University</publisher>
            <pubPlace>Ithaca, NY</pubPlace>
            <authority>Lexeis Project</authority>
            <availability status="free">
    
    <p>This text may be freely distributed, subject to the following
    restrictions:</p>
    <list>
        <item>You credit Lexeis, as follows, whenever you use the document:
        <quote>Text provided by Lexeis Project, with funding from Cornell University. Original version available for viewing and download at http://www.poesialatina.it/_ns/greek/testi/Plato/Spuria.html.</quote>
        </item>
        <item>You leave this availability statement intact.</item>
        <item>You use it for non-commercial purposes only.</item>
        <item>You offer Lexeis any modifications you make.</item>
    </list>
    </availability>
    </publicationStmt>
    
    <notesStmt><headnote>Text was scanned ???.</headnote></notesStmt>
    <sourceDesc default="NO">
    <biblStruct default="NO">
        <monogr>
            <author>Plato</author><title>{text_name}</title>
        
            <imprint><publisher>????</publisher><date>????</date></imprint>
        </monogr>
    </biblStruct>
    </sourceDesc>
    
    
    </fileDesc>
    
    <encodingDesc>
    
    <refsDecl doctype="TEI.2">
    <state unit="section" />
    </refsDecl>

    </encodingDesc>
    
    <profileDesc>
    <langUsage default="NO">
    <language id="greek">Greek</language>
    </langUsage>
    <textClass>
        <keywords scheme="genre">
            <term>prose</term>
        </keywords>
    </textClass>
    </profileDesc>
    <revisionDesc>
    <change>
    <date>July, 1992</date><respStmt><name>WPM</name><resp>(n/a)</resp></respStmt><item>Tagged in conformance with Prose.e dtd.</item>
    </change>
    
    </revisionDesc>
    
    </teiHeader>
    
    <text n="{text_name}">   
    <body>   
    <div1>

    """
    
    return xml_string

ID = 4339718
PAGE = 0
PUNCS = ";.·,"

# SO.
def speaker_case1(curr_token, next_token):
    if re.match(r'^[Α-Ω]{2}$', curr_token) and next_token == ".":
        return curr_token + next_token, 2 
    return None, 0

# -
def speaker_case2(token):
    if token.strip() == "‑":
        return "unknown", 1
    return None, 0

# SO. and -
def speaker_case3(curr_token, next_token, subsequent_token):
    if re.match(r'^[Α-Ω]{2}$', curr_token) and next_token == ".":
        return curr_token + next_token, 2
    elif curr_token.strip() == "‑" and next_token and re.match(r'^[Α-Ω]{2}$', next_token) and subsequent_token == ".":
        return next_token + subsequent_token, 3
    return None, 0 

def string_to_xml_file(xml_string, f):
        try:
            root = ElementTree.fromstring(xml_string)
            tree = ElementTree.ElementTree(root)
            tree.write(f, encoding="utf-8", xml_declaration=True)
        except ElementTree.ParseError as e:
            print(f"Error parsing XML string: {e}")

SPEAKER_CASES = {
    "ΠΕΡΙ ΔΙΚΑΙΟΥ ΣΩΚΡΑΤΗΣ ΑΝΩΝΨΜΟΣ ΤΙΣ": 2,
    "ΠΕΡΙ ΑΡΕΤΗΣ ΣΩΚΡΑΤΗΣ ΙΠΠΟΤΡΟΦΟΣ": 2,
    "ΔΗΜΟΔΟΚΟΣ": 2,
    "ΣΙΣΥΦΟΣ ΣΩΚΡΑΤΗΣ ΣΙΣΥΦΟΣ": 3,
    "ΕΡΥΞΙΑΣ": 2,
    "ΑΞΙΟΧΟΣ ΣΩΚΡΑΤΗΣ ΚΛΕΙΝΙΑΣ ΑΞΙΟΧΟΣ": 1
}

speaker_case = 0
current_speaker = None

for j in range(len(eng_headers)):
    current_speaker = None
    xml_string = create_xml_string(eng_headers[j])
    sec = splitted_sections[j]
    sec_header = headers[j]
    sec.insert(0, sec_header)
    speaker_case = SPEAKER_CASES[sec_header]
    i = 0
    speaker_started_new_sec = False

    while i < len(sec):
        ele = sec[i]
        next_token = sec[i + 1] if i + 1 < len(sec) else None
        subsequent_token = sec[i + 2] if i + 2 < len(sec) else None

        if ele in headers:
            speaker_case = SPEAKER_CASES[ele]
            # close a current speaker
            if current_speaker:
                xml_string += "\n</p></sp>\n"
                current_speaker = None
            
            speaker_started_new_sec = False

            header_toks = ele.split()
            header_string = "<head>"
            for tok in header_toks:
                tok_string = f'<w id="{ID}">{tok}</w>'
                if tok != header_toks[-1]:
                    tok_string += " "
                ID += 1
                header_string += tok_string
            header_string += "</head>   <castList></castList>"
            xml_string += "\n" + header_string + "\n"
        elif ele[0] == "[" and ele[-1] == "]":
            sec_string = ""
            content = ele[1:-1]
            if content.isnumeric():
                PAGE = int(content)
                sec_string += f'\n<milestone unit="page" n="{PAGE}" />'
            else:
                sec_string += f'\n<milestone n="{PAGE}{content}" unit="section"\n />'
            xml_string += sec_string + "\n"
        elif ele in PUNCS:
            punc_string = f'<w>{ele}</w> '
            xml_string += punc_string
            ID += 1
        else:
            new_speaker, tokens_to_skip = None, 0

            if speaker_case == 1:
                new_speaker, tokens_to_skip = speaker_case1(ele, next_token)
            elif speaker_case == 2:
                new_speaker, tokens_to_skip = speaker_case2(ele)
                if new_speaker is None and current_speaker is None and not speaker_started_new_sec:
                    new_speaker = "unknown"
                    tokens_to_skip = 0
            elif speaker_case == 3:
                new_speaker, tokens_to_skip = speaker_case3(ele, next_token, subsequent_token)
            
            if new_speaker:
                if new_speaker == "unknown":
                    if current_speaker:
                        xml_string += "\n</p></sp>\n"
                    current_speaker = "unknown" 
                    xml_string += f'\n<sp who="{current_speaker}"><p>\n'
                    speaker_started_new_sec = True
                elif current_speaker != new_speaker:
                    if current_speaker:
                        xml_string += "\n</p></sp>\n"
                    current_speaker = new_speaker
                    xml_string += f'\n<sp who="{current_speaker}"><p>\n'
                    speaker_started_new_sec = True
                
                i += tokens_to_skip
                continue 
            else:
                if speaker_case == 2 and current_speaker is None and not speaker_started_new_sec:
                    current_speaker = "unknown"
                    xml_string += f'\n<sp who="{current_speaker}"><p>\n'
                    speaker_started_new_sec = True
                
                xml_string += f'<w id="{ID}">{ele}</w> '
                ID += 1
        i += 1   

    if current_speaker:
        xml_string += "\n</p></sp>\n"

    xml_string += """
    </div1>
    </body>
    </text>
    </TEI.2>   
    """

    file_name = "/Users/ayan/Desktop/Lexeis-Aristophanes/utilities/plato/input/texts/Plato" + eng_headers[j] + "Gr.xml"
    string_to_xml_file(xml_string, file_name)
    
