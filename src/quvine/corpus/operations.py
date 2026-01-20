
def compile_corpus(node_corpus): 
    all_corpus = [] 
    for _, walks in node_corpus.items():
        for walk in walks:
            all_corpus.append(walk)
    return all_corpus