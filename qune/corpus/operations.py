
def define_corpus(view_walks): 
    corpus = [] 
    
    for view in view_walks:
        corpus.extend(view)
    
    return corpus 

def compile_corpus(node_corpus): 
    all_corpus = [] 
    for _, walks in node_corpus.items():
        for walk in walks:
            all_corpus.append(walk)
    return all_corpus