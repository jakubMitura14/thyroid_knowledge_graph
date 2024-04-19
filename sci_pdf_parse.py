"""
!!! it requires grobid server to be running


#!/bin/bash
# Recommended way to run Grobid is to have docker installed. See details here: https://grobid.readthedocs.io/en/latest/Grobid-docker/

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker before running Grobid."
    exit 1
fi


machine_arch=$(uname -m)

if [ "$machine_arch" == "armv7l" ] || [ "$machine_arch" == "aarch64" ]; then
    docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.7.3-arm
else
    docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.7.3
fi

"""
#docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0

import scipdf
import json
import os
import glob
import shutil
import docker
import random

def is_any_container_running():
    client = docker.from_env()
    running_containers = client.containers.list(filters={"status": "running"})
    return len(running_containers)

def save_preprocessed(pathh):
    try:
        # if(not is_any_container_running()):
        #     print("No grobid server running")
        #     client = docker.from_env()
        #     container = client.containers.run(
        #         "grobid/grobid:0.8.0", 
        #         detach=True, 
        #         auto_remove=True, 
        #         init=True, 
        #         ulimits=[docker.types.Ulimit(name='core', soft=0, hard=0)], 
        #         ports={8070: 8070}
        #     )
                        
        article_dict = scipdf.parse_pdf_to_dict(pathh)  # return dictionary
        # list(map( lambda el: el['heading'],article_dict['sections']))
        # article_dict['sections'][15]['text']
        file_name = os.path.basename(pathh).replace(".pdf", ".json")
        print(f"pathh: {pathh}")
        pp = f"/home/jakubmitura/projects/thyroid_knowledge_graph/preprocessed/{file_name}"
        with open(pp, "w") as file:
            json.dump(article_dict, file)
        # os.remove(pathh)
        print("succcesss")
    except Exception as er:
        print(f"Error: {er}")
        # backup=f"/home/jakubmitura/projects/thyroid_knowledge_graph/not_parsed_pdfs/{os.path.basename(pathh)}"
        # shutil.copy(pathh, backup)
        # print(f"error in {pathh}")
  


fold = "/home/jakubmitura/projects/thyroid_knowledge_graph/autodownloaded"
# fold = "/home/jakubmitura/projects/thyroid_knowledge_graph/not_parsed_pdfs"
pdf_files = glob.glob(os.path.join(fold, "*.pdf"))
# print(pdf_files)
random.shuffle(pdf_files)

list(map(save_preprocessed, pdf_files))
# full_paths = [os.path.abspath(file) for file in pdf_files]

# !!!!!!!!!1 good alternative here https://github.com/inukshuk/anystyle
# and pdfx "/home/jakubmitura/projects/thyroid_knowledge_graph/autodownloaded/Bone Metastases in Medullary Thyroid Carcinoma_ High Morbidity and Poor Prognosis Associated With Osteolytic Morphology.pdf" -t >output.txt