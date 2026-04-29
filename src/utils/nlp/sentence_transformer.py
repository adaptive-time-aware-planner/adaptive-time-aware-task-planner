### utils/nlp/sentence_transformer.py
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.common import create_module_logger
from src.utils.config.constants import SIMILARITY_THRESHOLD

log = create_module_logger(__name__, module_log=True)


class SentenceSimilarityModel:
    _instance = None

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if SentenceSimilarityModel._instance is not None:
            raise RuntimeError("싱글톤 클래스이므로, get_instance()를 사용하세요.")

        log.warning("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(model_name)
        SentenceSimilarityModel._instance = self

    @classmethod
    def get_instance(cls, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls(model_name)
        return cls._instance

    def encode_sentence(self, sentence: str) -> np.ndarray:
        return self.model.encode(sentence, convert_to_numpy=True)

    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        return self.model.encode(sentences, convert_to_numpy=True)

    def get_similar_ref(self, query: str, references: list[str]) -> str:
        """references 중에서 query와 유사한 문장을 반환한다.
        유사도 점수가 SIMILARITY_THRESHOLD보다 큰 경우 가장 유사한 문장을 반환하고,
        그렇지 않으면 원본 query를 반환한다.

        Args:
            query (str): 찾고자 하는 문장
            references (list[str]): 비교 대상 문장들

        Returns:
            str: 가장 유사한 문장 또는 원본 query
        """
        if not references:  # references가 비어있는 경우 query 반환
            return query

        query_vec = self.encode_sentence(query)
        ref_vecs = self.encode_sentences(references)

        query_norm = query_vec / np.linalg.norm(query_vec)
        ref_norms = ref_vecs / np.linalg.norm(ref_vecs, axis=1, keepdims=True)
        cosine_similarities = np.dot(ref_norms, query_norm)

        max_idx = np.argmax(cosine_similarities)
        max_score = cosine_similarities[max_idx]

        return references[max_idx] if max_score > SIMILARITY_THRESHOLD else query
