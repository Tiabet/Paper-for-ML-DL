import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random

# 데이터셋 (예제 문장)
corpus = [
    "the king loves the queen",
    "the queen loves the king",
    "the king is strong",
    "the queen is wise",
    "the strong king loves the wise queen"
]

# 단어 토큰화
tokenized_corpus = [sentence.split() for sentence in corpus]

# 어휘 사전 (Vocabulary)
word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
vocab = list(word_counts.keys())
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 하이퍼파라미터 설정
CONTEXT_SIZE = 2  # 문맥 크기 (앞뒤 몇 개의 단어를 고려할지)
EMBEDDING_DIM = 10  # 단어 임베딩 차원
EPOCHS = 100  # 학습 횟수
LEARNING_RATE = 0.01  # 학습률

# Skip-gram 데이터셋 생성 함수
def create_skipgram_dataset(tokenized_corpus, context_size):
    dataset = []
    for sentence in tokenized_corpus:
        for i, target_word in enumerate(sentence):
            # 중심 단어를 기준으로 앞뒤 context_size 범위 내에서 문맥 단어 선택
            start = max(0, i - context_size)
            end = min(len(sentence), i + context_size + 1)
            for j in range(start, end):
                if i != j:
                    dataset.append((target_word, sentence[j]))  # (중심 단어, 문맥 단어) 쌍 저장
    return dataset

# Skip-gram 데이터셋 생성
dataset = create_skipgram_dataset(tokenized_corpus, CONTEXT_SIZE)

# 입력 데이터를 텐서로 변환
def word_to_tensor(word, word_to_idx):
    return torch.tensor([word_to_idx[word]], dtype=torch.long)

# Skip-gram 모델 정의
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word):
        embeds = self.embeddings(target_word)  # 중심 단어의 임베딩 추출
        out = self.linear(embeds)  # 선형 변환 후 소프트맥스 적용
        return out

# 모델 초기화
model = SkipGram(len(vocab), EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 모델 학습
print("Training Skip-gram Model...\n")
for epoch in range(EPOCHS):
    total_loss = 0
    for target, context in dataset:
        target_tensor = word_to_tensor(target, word_to_idx)
        context_tensor = word_to_tensor(context, word_to_idx)

        optimizer.zero_grad()
        output = model(target_tensor)
        loss = criterion(output.view(1, -1), context_tensor)  # 중심 단어 → 문맥 단어 예측 손실 계산
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# 학습된 단어 임베딩 확인
print("\nWord Embeddings:")
for word, idx in word_to_idx.items():
    print(f"{word}: {model.embeddings.weight[idx].detach().numpy()}")

# 모델 저장
torch.save(model.state_dict(), "skipgram_model.pth")
print("\nModel saved as skipgram_model.pth")
