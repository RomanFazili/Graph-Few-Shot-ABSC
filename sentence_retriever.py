import os
import re
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from data_set import DataSet
from enums import Polarity
from rdflib import Graph

Tokenizer = Callable[[str], list[str]]

_SIMCSE_MODEL_NAME = "princeton-nlp/unsup-simcse-bert-base-uncased"


def _sparql_escape_double_quoted_literal(value: str) -> str:
    """Escape for inclusion inside SPARQL double-quoted string literals."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _normalize_sentence_for_lex_match(sentence: str) -> str:
    """Strip punctuation/symbols; keep letters, digits, and whitespace (Unicode). Collapse runs of spaces."""
    no_marks = re.sub(r"[^\w\s]", "", sentence, flags=re.UNICODE)
    return re.sub(r"\s+", " ", no_marks).strip()


class SentenceRetriever:
    """Retrieves demonstration sentences; BM25 / SimCSE indices are built lazily once each."""

    def __init__(self, data_set: DataSet, ontology: Graph):
        self.data_set = data_set
        self.ontology = ontology

        self._corpus_rows: list[tuple[str, list[tuple[str, Polarity]]]] | None = None
        self._bm25_tokenized: list[list[str]] | None = None
        self._bm25_index: BM25Okapi | None = None
        self._simcse_model: SentenceTransformer | None = None
        self._simcse_embeddings: np.ndarray | None = None

        self._graph_lex_nodes_by_sentence: dict[str, list[str]] | None = None

    def _get_corpus_rows(self) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        if self._corpus_rows is None:
            self._corpus_rows = list(self.data_set.all_sentences_with_aspects_and_polarities)
        return self._corpus_rows

    def _ensure_bm25_index(self) -> None:
        if self._bm25_index is not None:
            return
        corpus = self._get_corpus_rows()
        self._bm25_tokenized = [self.tokenize_bm25(sentence) for sentence, _ in corpus]
        self._bm25_index = BM25Okapi(corpus=self._bm25_tokenized)

    def _ensure_simcse_embeddings(self) -> None:
        if self._simcse_embeddings is not None:
            return
        corpus = self._get_corpus_rows()
        if self._simcse_model is None:
            self._simcse_model = SentenceTransformer(_SIMCSE_MODEL_NAME)
        sentences = [sentence for sentence, _ in corpus]
        self._simcse_embeddings = self._simcse_model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def tokenize_bm25(self, text: str) -> list[str]:
        """Tokenize for BM25: lowercase alnum tokens including simple apostrophe forms."""
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    def BM25_demonstration_selection(
        self, query_sentence: str, top_k: int
    ) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """BM25 over the training corpus; index built on first call only."""

        if top_k == 0:
            return []

        self._ensure_bm25_index()
        corpus = self._get_corpus_rows()
        assert self._bm25_index is not None

        tokenized_query = self.tokenize_bm25(query_sentence)
        scores: np.ndarray = self._bm25_index.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [corpus[int(i)] for i in top_indices]

    def SimCSE_demonstration_selection(
        self, query_sentence: str, top_k: int
    ) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """SentenceTransformer similarity; model + corpus embeddings built on first call only."""

        if top_k == 0:
            return []

        self._ensure_simcse_embeddings()
        corpus = self._get_corpus_rows()
        assert self._simcse_model is not None
        assert self._simcse_embeddings is not None

        query_embedding: np.ndarray = self._simcse_model.encode(
            [query_sentence],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = util.pytorch_cos_sim(query_embedding, self._simcse_embeddings)
        top_indices = scores[0].argsort(descending=True)[:top_k].cpu().tolist()
        return [corpus[int(i)] for i in top_indices]

    def _get_nodes_from_sentence_via_lex_restaurant(self, sentence: str) -> list[str]:
        """
        Return local names of every class whose ``restaurant:lex`` occurs in ``sentence`` (case-insensitive),
        deduplicated.

        The sentence is normalized by removing punctuation and other non-word, non-space symbols, then
        matching requires ASCII spaces on both sides of the lex substring (via leading/trailing padding
        and ``CONCAT(" ", …, " ")`` in SPARQL).
        """
        normalized = _normalize_sentence_for_lex_match(sentence)
        safe = _sparql_escape_double_quoted_literal(f" {normalized} ")

        sparql_query = f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX restaurant: <http://www.kimschouten.com/sentiment/restaurant#>
        SELECT DISTINCT ?s WHERE {{
            ?s a owl:Class .
            ?s restaurant:lex ?lexValue .
            FILTER(CONTAINS(LCASE("{safe}"), CONCAT(" ", LCASE(STR(?lexValue)), " ")))
        }}
        """
        qres = self.ontology.query(sparql_query)

        def _local_name(node: object) -> str:
            s = str(node).rstrip("/#")
            return s.split("/")[-1].split("#")[-1]

        accum: list[str] = []
        for row in qres:
            accum.append(_local_name(row[0]))
        return list(dict.fromkeys(accum))


    def graph_based_demonstration_selection_naive(self, query_sentence: str, top_k: int) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """
        For the input sentence, fetch a list (or set?) of nodes that can be found in the ontology
        Do the same for all sentences in the training data

        Then, compare the sentences in the training data to the input sentence using a similarity:
        - jaccard
        - cosine

        And return the top k sentences.
        """

        if self._graph_lex_nodes_by_sentence is None:
            nodes_by_sentence: dict[str, list[str]] = {}
            for sentence in self.data_set.all_sentences_as_text:
                nodes = self._get_nodes_from_sentence_via_lex_restaurant(sentence)
                nodes_by_sentence[sentence] = nodes

            self._graph_lex_nodes_by_sentence = nodes_by_sentence
        else:
            nodes_by_sentence = self._graph_lex_nodes_by_sentence

        query_nodes = self._get_nodes_from_sentence_via_lex_restaurant(query_sentence)

        corpus = self._get_corpus_rows()
        row_by_sentence: dict[str, tuple[str, list[tuple[str, Polarity]]]] = {
            row[0]: row for row in corpus
        }

        # Find top k sentences that are most similar to the input sentence using Jaccard similarity
        similarities: list[tuple[str, float]] = []
        q_set = set(query_nodes)
        for sentence, nodes in nodes_by_sentence.items():
            n_set = set(nodes)
            union = q_set | n_set
            similarity = 1.0 if not union else len(q_set & n_set) / len(union)
            similarities.append((sentence, similarity))
        similarities.sort(key=lambda pair: pair[1], reverse=True)
        return [row_by_sentence[sentence] for sentence, _ in similarities[:top_k]]


if __name__ == "__main__":
    load_dotenv()
    from data_set_ontology import DataSetOntology
    file_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA")
    assert file_path

    ontology_path = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")
    data_set_ontology = DataSetOntology(ontology_path)
    g = data_set_ontology.get_rdflib_graph()
    sentence_retriever = SentenceRetriever(DataSet(file_path), g)
    print(sentence_retriever._get_nodes_from_sentence_via_lex("I enjoyed the green tea."))
    print(sentence_retriever._get_nodes_from_sentence_via_lex("I enjoyed the green ravelling tea."))
    print(sentence_retriever.graph_based_demonstration_selection_naive("We very much enjoyed the restaurant and the food.", 3))