"""
given path to article parsed by scipdf download the articles from its bibliography
"""

import json
from habanero import Crossref
import os
from getarticle import GetArticle
import subprocess
from subprocess import Popen
from paperscraper.pdf import save_pdf

cr = Crossref()
from habanero import Crossref
# ga = GetArticle()

path_old="/workspaces/thyroid_knowledge_graph/preprocessed/old_guidelines.json"

with open(path_old, 'r') as file:
    data = json.load(file)


def downl(ref):
    title=ref['title']

    # try:
    result = cr.works(query = title)
    # fold=f"/workspaces/thyroid_knowledge_graph/data/auto_downloaded"
    # fold=f"/workspaces/thyroid_knowledge_graph/data/auto_downloaded/{title}"
    # os.makedirs(fold,exist_ok=True)
    # path_out=f'"{fold}"'
    doi=result['message']['items'][0]['DOI']
    print(f"doii {doi}")
    paper_data = {'doi': f"{doi}"}
    title= title.replace(" ", "_")
    save_pdf(paper_data, filepath=f'/workspaces/thyroid_knowledge_graph/data/auto_downloaded/{title}.pdf')
    # p = subprocess.Popen(f'python3 -m PyPaperBot --doi="{doi}" --dwn-dir={path_out}', shell=True)
    # p = subprocess.Popen('python3 -m PyPaperBot --doi="10.1016/s0084-3873(08)70297-9" --dwn-dir="/workspaces/thyroid_knowledge_graph/data/auto_downloaded/a"', shell=True)
    # p.wait()

    # os.system(f'python3 -m PyPaperBot --doi="{doi}" --dwn-dir={path_out}')
    # print os.popen("echo Hello World").read()
    # ga.input_article(f'"{doi}"')
    # ga.download(fold)
    # except:
        # print(f"Could not download {title}")    



list(map(downl, data['references']))
# downl(data['references'][3])