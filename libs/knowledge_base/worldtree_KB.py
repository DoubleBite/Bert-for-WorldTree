from typing import List, Tuple, Dict
from tqdm import tqdm
import copy
from collections import defaultdict

import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from libs.knowledge_base.bm25 import BM25Vectorizer
from libs.knowledge_base.utils import preprocess


class WorldTreeKB:
    """The knowledge base used to store scientific facts and return relevant facts given a query.

    The KB first takes a list of scientific facts as input. Then, it calculates the bm25 parameters on top
    of these facts. Finally, given a query, it returns the relevant facts assicuated with this query.

        For example: 
            facts = [
                {"id": 1, "original_fact": "an apples is a kind of fruit", "lemmatized_fact": "apple be kind of fruit"},
                {"id": 2, "original_fact": "a girl is a kind of human", "lemmatized_fact": "girl be kind of human"},]
            query = "what is an apple?"
        it should return the relevant facts:
            relevant_facts = [
                {"id": 1, "original_fact": "an apples is a kind of fruit", "lemmatized_fact": "apple be kind of fruit"}
            ]

    """

    def __init__(self,
                 ranking_func: str = "BM25",
                 lemmatizer=None):
        """
        Args:
            facts: `List`, required 
                A list of facts. A fact is a user-defined dict object that contains the information
                for a scientific fact. It should contain the required fields "id" and "lemmatized_fact".
            rank_func: `str`, optional (default=`BM25`)  
                A string to assign the ranking function for this knowledge base.
            corpus_to_fit: `List[str]`, optional (default=None)  
                A list of strings used to feed to the ranking function and tune its parameters.
                If `corpus_to_fit` is None, the ranking function will fit on `self.facts` by default.
        """
        self.facts = None
        self._transformed_facts = None

        self.ranking_function = None
        self.lemmatizer = lemmatizer

        # Initialize the ranking function
        # TODO: add another ranking function?
        if ranking_func == "BM25":
            self.ranking_function = BM25Vectorizer()

    def fit_to_corpus(self, corpus: List[str]):
        """
        """

        # Preprocess each sentence in the corpus
        corpus = [preprocess(x, self.lemmatizer) for x in corpus]

        self.ranking_function.fit(corpus)

    def set_documents(self, facts):
        """
        """
        self.facts = facts

        text_to_transform = []
        for fact in self.facts:
            processed = preprocess(fact["fact"], self.lemmatizer)
            fact["processed_fact"] = processed
            text_to_transform.append(processed)

        self._transformed_facts = self.ranking_function.transform(
            text_to_transform)

    def query_relevant_facts(self, query: str, topk: int = 10):
        """
        Args
            query: `str`, required
                The query string.
            topk: `int`, optional
                The number of top candidates to be considered.
        Returns
            relevant_facts: `List[Dict]`
                A list of relevant facts.
            id_to_score: `Dict`
                The dict that maps the ids of the relevant facts to their score.
        """
        query = preprocess(query, self.lemmatizer)

        # Calculate cosine similarity
        transformed_query = self.ranking_function.transform([query])
        # Shape: 1*num_facts -> facts
        similarities = cosine_distances(
            transformed_query, self._transformed_facts
        )[0]

        # Get topk relevant facts
        rank = np.argsort(similarities)  # Descending order
        if topk:
            rank = rank[:topk]
        relevant_facts = []
        for index in rank:
            fact = copy.deepcopy(self.facts[index])
            score = 1 - similarities[index]
            fact["relevance_score"] = score
            relevant_facts.append(fact)

        return relevant_facts
