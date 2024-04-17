

import scipdf
import json
import os
import glob


def save_preprocessed(pathh):
    try:
        article_dict = scipdf.parse_pdf_to_dict(pathh)  # return dictionary
        # list(map( lambda el: el['heading'],article_dict['sections']))
        # article_dict['sections'][15]['text']
        file_name = os.path.basename(pathh).replace(".pdf", ".json")

        pp = f"/media/jm/hddData/projects/thyroid_knowledge_graph/preprocessed/{file_name}"
        with open(pp, "w") as file:
            json.dump(article_dict, file)
    except:
        print(f"error in {pathh}")
        pass


fold = "/media/jm/hddData/projects/thyroid_knowledge_graph/data/auto_downloaded"
pdf_files = glob.glob(os.path.join(fold, "*.pdf"))
print(pdf_files)

list(map(save_preprocessed, pdf_files))
# full_paths = [os.path.abspath(file) for file in pdf_files]
