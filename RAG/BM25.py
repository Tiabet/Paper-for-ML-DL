import math
from collections import Counter
import pandas as pd

class TFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.doc_count = len(documents)
        self.df = self._compute_df()  # 문서 빈도수 계산
        self.idf = self._compute_idf()  # IDF 계산

    def _compute_df(self):
        """ 문서 빈도수(DF) 계산 """
        df = {}
        for doc in self.documents:
            for word in set(doc):  # 각 문서에서 중복 제거된 단어 집합 사용
                df[word] = df.get(word, 0) + 1
        return df

    def _compute_idf(self):
        """ IDF(Inverse Document Frequency) 계산 """
        idf = {}
        for word, freq in self.df.items():
            idf[word] = math.log((self.doc_count + 1) / (freq + 1)) + 1  # Smoothing 적용
        return idf

    def compute_tf(self, document):
        """ TF(Term Frequency) 계산 """
        tf = Counter(document)
        doc_len = len(document)
        return {word: count / doc_len for word, count in tf.items()}

    def compute_tfidf(self, query, document):
        """ 특정 문서에 대한 TF-IDF 점수 계산 """
        tf = self.compute_tf(document)
        score = 0
        for word in query:
            if word in self.idf:
                score += tf.get(word, 0) * self.idf[word]
        return score

    def rank_documents(self, query):
        """ 질의(query)에 대해 문서별 TF-IDF 점수 계산 """
        query = query.split()
        scores = []
        for i, doc in enumerate(self.documents):
            score = self.compute_tfidf(query, doc)
            scores.append((i, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.avgdl = sum(len(doc) for doc in documents) / self.doc_count  # 평균 문서 길이
        self.df = self._compute_df()  # 문서 빈도수 계산
        self.idf = self._compute_idf()  # IDF 계산

    def _compute_df(self):
        """ 문서 빈도수(DF) 계산 """
        df = {}
        for doc in self.documents:
            for word in set(doc):  # 각 문서에서 중복 제거된 단어 집합 사용
                df[word] = df.get(word, 0) + 1
        return df

    def _compute_idf(self):
        """ IDF(Inverse Document Frequency) 계산 """
        idf = {}
        for word, freq in self.df.items():
            idf[word] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def compute_bm25(self, query, document):
        """ 특정 문서에 대한 BM25 점수 계산 """
        score = 0
        doc_len = len(document)
        for word in query:
            if word not in self.idf:
                continue
            f = document.count(word)  # 단어 빈도(TF)
            numerator = self.idf[word] * f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += numerator / denominator
        return score

    def rank_documents(self, query):
        """ 질의(query)에 대해 문서별 BM25 점수 계산 """
        query = query.split()
        scores = []
        for i, doc in enumerate(self.documents):
            score = self.compute_bm25(query, doc)
            scores.append((i, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)

# 테스트 데이터
documents = [
    "deepfake detection technology is improving".split(),
    "deepfake videos are becoming more realistic".split(),
    "the best way to detect deepfakes is AI".split()
]

# 모델 생성
tfidf = TFIDF(documents)
bm25 = BM25(documents)

# 질의 입력
query = "deepfake detection"

# 문서 순위 계산
tfidf_scores = tfidf.rank_documents(query)
bm25_scores = bm25.rank_documents(query)

# 결과 비교
df_results = pd.DataFrame({
    "Document": [" ".join(doc) for doc in documents],
    "TF-IDF Score": [score for _, score in tfidf_scores],
    "BM25 Score": [score for _, score in bm25_scores],
})

print(df_results)