from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt

# 예제 문서들
documents = [
    "나는 머신러닝을 공부하고 있다.",
    "딥러닝과 머신러닝은 인공지능의 한 분야이다.",
    "자연어 처리는 텍스트 데이터를 다루는 기술이다."
]

# 토큰화 진행
okt = Okt()
tokenized_docs = [" ".join(okt.morphs(doc)) for doc in documents]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

print(vectorizer.vocabulary_)
print(tfidf_matrix)
# 문서 간 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 결과 출력
import pandas as pd

df = pd.DataFrame(cosine_sim, index=[f"문서 {i+1}" for i in range(len(tokenized_docs))],
                   columns=[f"문서 {i+1}" for i in range(len(tokenized_docs))])

print(df)
