import requests
from bs4 import BeautifulSoup
import re

URL = "http://www.poesialatina.it/_ns/greek/testi/Plato/Spuria.html"
r = requests.get(URL)
soup = BeautifulSoup(r.content.decode('utf-8','ignore'), "html.parser")
relevant = soup.find_all(["h2", "p"])
sections = []
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
        
        out_string = ""

        k = last_h2_ind + 1
        
        while k < len(relevant) and relevant[k].name != "h2":
            if relevant[k].name == "p":
                para = relevant[k]
                para = re.sub("<[^>]*>", "", str(para))
                para = re.sub(r"\[[^\]]*\]", "", para)
                out_string += para
            k += 1
        
        if out_string != "":
            sections.append(out_string)
        i = k 
    else:
        i += 1