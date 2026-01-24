# Copyright 2021, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from gensim.models import Word2Vec
import hashlib

def corpus_to_embedding(
    corpus,
    nodes,
    vector_size=64,
    window=5,
    sg=1,
    negative=10,
    min_count=0,
    workers=8,
    epochs=100,
):
    """
    Train Word2Vec embeddings from a compiled corpus.

    corpus: List[List[str]] or List[List[int]]
    nodes:  List[str] or List[int]
    """
    
    assert all(isinstance(w[0], str) for w in corpus)
    # ensure all nodes appear at least once
    sentences = [list(w) for w in corpus]
    seen = set(x for w in sentences for x in w)
    sentences += [[v] for v in nodes if v not in seen]
    #print("deep hash:", hash(tuple(tuple(w) for w in sentences[:50])))
    assert all(isinstance(w, list) and all(isinstance(t, str) for t in w) for w in corpus)
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        sg=sg,
        negative=negative,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    #print("model id:", id(model))
    
    assert set(nodes) <= set(model.wv.key_to_index), \
    "Node list and Word2Vec vocabulary are inconsistent"
    assert all(v in model.wv for v in nodes)
    
    Z = np.zeros((len(nodes), vector_size))
    for i, v in enumerate(nodes):
        Z[i] = model.wv[v]

    
    W = model.wv.vectors
    h = hashlib.md5(W[:10,:10].tobytes()).hexdigest()
    # print("weights md5:", h)
    
    # print("vocab size:", len(model.wv))
    # print("first 3 vocab:", list(model.wv.key_to_index.keys())[:3])
    # print("vector[0] head:", model.wv[nodes[0]][:5])
    
    # print(
    # "sanity",
    # np.std(Z),
    # np.mean(Z),
    # np.sum(Z[:5, :5])   
    # )
    
    return Z
