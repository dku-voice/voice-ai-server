"""
app/services/menu_embedding.py - 메뉴 메타데이터 hash TF-IDF 임베딩 인덱스

제5주차 RAG 검색 최적화에 쓰는 가벼운 인메모리 임베딩.
무거운 외부 모델, 네트워크 임베딩 API, 벡터 DB 의존성은 일부러 안 쓴다.
학생 프로젝트라 SHA1 해시 + 문자 bigram + IDF 가중으로 단순하게 둔다.

키워드 매치가 못 잡는 부분 일치/동의어 입력도 후보로 끌어올리는 게 목적.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from math import log, sqrt
from typing import Iterable, Sequence

# 해시 버킷 수. 충돌이 잦으면 무관한 쿼리도 점수가 새서, 메뉴 5개 기준 넉넉히 4096으로 잡는다.
# sparse dict 저장이라 빈 버킷은 메모리를 차지하지 않는다.
EMBEDDING_DIM = 4096

# 임베딩 유사도를 keyword 점수에 더할 때 곱하는 배수. exact 매치(10+)보다 작게 둔다.
EMBEDDING_SCORE_SCALE = 5

# 유사도가 이 값 미만이면 점수에 더하지 않는다. 해시 충돌로 인한 false positive 차단용.
EMBEDDING_MIN_BOOST_SIMILARITY = 0.15


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def _hash_to_bucket(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % EMBEDDING_DIM


def _iter_terms(text: str) -> Iterable[str]:
    """단어 단위와 길이 2 문자 bigram을 같이 뽑아 hash bag 입력으로 쓴다."""
    for word in _TOKEN_PATTERN.findall(text.lower()):
        if not word:
            continue
        yield word
        if len(word) >= 2:
            for idx in range(len(word) - 1):
                yield word[idx:idx + 2]


def _term_buckets(text: str) -> Counter:
    counter: Counter = Counter()
    for term in _iter_terms(text):
        counter[_hash_to_bucket(term)] += 1
    return counter


def _build_idf(corpus: Sequence[Counter]) -> dict[int, float]:
    """corpus 전체 기준으로 bucket별 inverse document frequency를 계산한다."""
    num_docs = len(corpus)
    if num_docs == 0:
        return {}
    doc_freq: dict[int, int] = {}
    for counter in corpus:
        for bucket in counter:
            doc_freq[bucket] = doc_freq.get(bucket, 0) + 1
    return {
        bucket: log((num_docs + 1) / (freq + 1)) + 1.0
        for bucket, freq in doc_freq.items()
    }


def _to_unit_vector(counter: Counter, idf: dict[int, float]) -> dict[int, float]:
    weighted = {bucket: tf * idf.get(bucket, 1.0) for bucket, tf in counter.items()}
    norm = sqrt(sum(value * value for value in weighted.values()))
    if norm == 0.0:
        return {}
    return {bucket: value / norm for bucket, value in weighted.items()}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    return sum(value * larger.get(bucket, 0.0) for bucket, value in smaller.items())


def embedding_boost(similarity: float) -> int:
    """코사인 유사도를 정수 boost 점수로 바꾼다. 약한 신호는 0으로 누른다."""
    if similarity < EMBEDDING_MIN_BOOST_SIMILARITY:
        return 0
    return int(round(similarity * EMBEDDING_SCORE_SCALE))


class MenuEmbeddingIndex:
    """메뉴 메타데이터 텍스트로 만든 해시 TF-IDF 인덱스."""

    def __init__(self, sources: Sequence[tuple[str, str]]):
        """
        sources: [(menu_id, embedding_source_text), ...] 형태.
        같은 menu_id가 들어오면 마지막 값으로 덮어쓰지만, 호출 측에서 중복은 막아 둔다.
        """
        counters = [_term_buckets(text) for _, text in sources]
        self._idf = _build_idf(counters)
        self._vectors: dict[str, dict[int, float]] = {
            menu_id: _to_unit_vector(counter, self._idf)
            for (menu_id, _), counter in zip(sources, counters)
        }

    def query_vector(self, query: str) -> dict[int, float]:
        """쿼리 문자열을 동일한 해시 TF-IDF 공간에 투영한다."""
        return _to_unit_vector(_term_buckets(query), self._idf)

    def similarity(self, query_vector: dict[int, float], menu_id: str) -> float:
        """이미 만들어둔 query_vector와 menu_id 문서 벡터 간 코사인 유사도."""
        doc_vector = self._vectors.get(menu_id)
        if not doc_vector:
            return 0.0
        return _cosine(query_vector, doc_vector)
