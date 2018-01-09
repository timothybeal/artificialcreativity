# Reads a textual corpus and, based on an algorithm adapted from Google's PageRank,
# provides the n most representative sentences.

import re

import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def get_highest_pagerank_scores(fileid, n=5):

    with open(fileid, encoding="utf-8") as f:
        text = f.read()
        sentences = re.findall(r'.*?\n', text[0:10000], flags=re.DOTALL)

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    
    transformer = TfidfTransformer()
    normalized = transformer.fit_transform(matrix)
    
    similarity_graph = normalized * normalized.T
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    
    scores = nx.pagerank(nx_graph)
    
    index_scores = scores.items()
    sorted_scores = sorted(index_scores, key=lambda x: x[1], reverse=True)
    
    for index, score in sorted_scores[:n]:
        print(score, sentences[index])
        
get_highest_pagerank_scores("RIM.txt", n=5)
