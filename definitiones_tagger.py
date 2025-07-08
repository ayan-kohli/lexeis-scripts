import requests
from bs4 import BeautifulSoup
import re
from xml.etree import ElementTree
from spuria_tagger import string_to_xml_file

URL = "http://www.poesialatina.it/_ns/greek/testi/Plato/Definitiones_[Sp.].html"
r = requests.get(URL)
soup = BeautifulSoup(r.content.decode('utf-8','ignore'), "html.parser")
splitted_sections = []
store = True
i = 0

paragraph = []

for p in soup.find_all("p"):
    processed = []

    parts = re.split(r'(\[[^\]]+\])', p.text)

    for part in parts:
        if not part:
            continue

        if part.startswith('[') and part.endswith(']'):
            content = part[1:-1].strip()

            if re.match(r'^(\d+|[a-zA-Z])$', content):
                processed.append(part)
            else:
                processed.append('[')
                processed.extend(content.split())
                processed.append(']')
        else:
            part = part.replace(".", " .")
            part = part.replace(";", " ;")
            part = part.replace(",", " ,")
            part = part.replace("·", " ·")
            processed.extend(part.split())

    paragraph.extend(processed)

xml_string = """
<TEI.2>
  
  <teiHeader status="new" type="text">
  
  <fileDesc>
  
  <titleStmt>
  <title>Definitiones</title>
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
  		<author>Plato</author><title>Definitiones</title>
  	
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
  
 <text n="Definitiones">   
<body>   
<div1>

<head><w id="4356168">Definitiones</w></head>   <castList />
"""

# starting ID at end of Spuria
ID = 4356169
PAGE = 0
PUNCS = ";.·,[]"

for ele in paragraph:
    if ele[0] == "[" and ele[-1] == "]":
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
        xml_string += f'<w id="{ID}">{ele}</w> '
        ID += 1

xml_string += """
</div1>
</body>
</text>
</TEI.2>   
"""

string_to_xml_file(xml_string, "/Users/ayan/Desktop/Lexeis-Aristophanes/utilities/plato/input/texts/PlatoDefinitionesGr.xml")