__all__ = []

try:
    from quvine.embedding.word2vec import corpus_to_embedding
    __all__.append("corpus_to_embedding")
except ImportError:
    pass