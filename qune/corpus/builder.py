
from qune.corpus.operations import define_corpus, compile_corpus 

class CorpusBuilder: 
    def __init__(self):
        self._node_corpus = {} 
    
    def add(self, root, view_walks): 
        self._node_corpus[root] = define_corpus(view_walks)
        
    def build(self):
        return compile_corpus(self._node_corpus)