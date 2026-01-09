
class EmbeddingStore: 
    def __init__(self): 
        self._Z = {} 
        
    def add(self, name, Z): 
        self._Z[name] = Z 
    
    def names(self): 
        return list(self._Z.keys())
    
    def items(self): 
        return self._Z.items() 
    
    def get(self, name): 
        return self._Z[name]
    
    def as_list(self): 
        return list(self._Z.values())
    
    