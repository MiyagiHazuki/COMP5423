import pandas as pd
import time
from .base import FaissSaver, FaissQuery
import numpy as np
import json
import os

class faiss_save:
    '''
    input:
    json_path: Path to the json file
    output:
    id List[str], 
    vector List[List[float]]
    '''
    def __init__(self, json_path):
        self.df = pd.read_json(json_path)
        self.doc_ids = self.df.doc_ids
        self.vectors = self.df.vectors

    def init_vector(self):
        self.doc_ids = self.df.doc_ids.tolist()
        print(type(self.doc_ids))
        self.vectors = self.df.vectors.tolist()
        print(type(self.vectors))

class query_init:
    '''
    input:
    json_path: Path to the json file
    output:
    query: str
    '''
    def __init__(self, json_path):
        self.df = pd.read_json(json_path, lines=True)

    def init_query(self):
        return self.df.question.tolist()

class rank_init:
    '''
    input:
    json_path: Path to the json file
    output:
    json_path: Path to the json file
    '''
    def __init__(self, json_path):
        self.df = pd.read_json(json_path, lines=True)
        self.doc_ids = self.df.doc_ids
        self.vectors = self.df.vectors
        

class VectorSaver(FaissSaver):
    '''
    Save the text and vector information of the query to a JSON file
    input:
    ids: Each unique ID of document List[str]
    vectors: Embedding vectors of documents List[List[float]] [MUST BE NORMALIZED]
    request: Retrieval method request str
    save_path: Path to save the index str
    '''
    def __init__(self, ids, vectors, request, save_path):
        self.id = ids
        self.vector = vectors
        self.request = request
        self.save_path = save_path

    def save(self):
        super().save()
        # Create save directory
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Save original vectors for future use
        data = {"id": self.id, "vector": self.vector}
        with open(self.save_path + self.request + ".json", "w") as f:
            json.dump(data, f)

class VectorQuery(FaissQuery):
    '''
    Query the index file with a vector
    input:
    vector: Embedding vector of query List[List[float]] [MUST BE NORMALIZED]
    top_k: Number of nearest neighbors to retrieve int
    faiss_path: Path to the saved index file str
    output:
    doc_id: IDs of documents List[str]
    '''
    
    def __init__(self, vector, top_k, faiss_path):
        self.vector = vector
        self.top_k = top_k
        self.faiss_path = faiss_path

if __name__ == "__main__":
    # faiss_save = faiss_save("intermediate_results/vectors_1744199857.json")
    # faiss_save.init_vector()
    # from faiss_composer.base import FaissSaver
    # faiss_saver = FaissSaver(faiss_save.doc_ids, faiss_save.vectors, "bgem3", "faiss/")
    # faiss_saver.save()

    # ======================================================================== #

    # query_init = query_init("./bgem3/val.jsonl")
    # queries = query_init.init_query() 
    # from bgem3.vectorizer import query_vectorizer as qv
    # import json
    # for i in range(len(queries)):
    #     query_vectorizer = qv(queries[i], "sk-nfizfypjawwnixaimezwbkxbhomzpuozlungqykzkwyporuk", "BAAI/bge-m3")
    #     query_vector = query_vectorizer.vectorize()
    #     record = {
    #         "question": queries[i],
    #         "vector": query_vector
    #     }
    #     with open("query_vectors.jsonl", "a") as f:
    #         json.dump(record, f)
    #         f.write("\n")
    #     time.sleep(1)

    # ======================================================================== #

    # from faiss_composer.base import FaissQuery
    # import pandas as pd
    # import json
    # combine = pd.read_json("query_vectors.jsonl", lines=True)
    # question = combine.question.tolist()
    # vector = combine.vector.tolist()
    # for i in range(len(question)):
    #     faiss_query = FaissQuery([vector[i]], top_k=20, faiss_path="faiss/bgem3.faiss")
    #     doc_ids, scores = faiss_query.query()
    #     dict = {
    #         "question": question[i],
    #         "answer": None,
    #         "document_ids": doc_ids,
    #     }
    #     score = {
    #         "question": question[i],
    #         "score": scores
    #     }
    #     with open("faiss_results.jsonl", "a") as f:
    #         json.dump(dict, f)
    #         f.write("\n")
    #     with open("score.jsonl", "a") as f:
    #         json.dump(score, f)
    #         f.write("\n")
    # ======================================================================== #
    from faiss_composer.base import FaissQuery
    import pandas as pd
    import json
    combine = pd.read_json("query_vectors.jsonl", lines=True)
    question = combine.question.tolist()
    vector = combine.vector.tolist()
    for i in range(len(question)):
        faiss_query = FaissQuery([vector[i]], top_k=5, faiss_path="faiss/bgem3.faiss")
        doc_ids, scores = faiss_query.query()
        for doc_id in doc_ids:
            doc_ids[doc_ids.index(doc_id)] = int(doc_id)
        dict = {
            "question": question[i],
            "answer": "Placeholder",
            "document_id": doc_ids,
        }
        with open("data/val_predict.jsonl", "a") as f:
            json.dump(dict, f)
            f.write("\n")
    pass
