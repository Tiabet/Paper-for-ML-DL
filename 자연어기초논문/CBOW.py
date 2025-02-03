import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

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
# 언급빈도 큰 순에서 작은 순으로 정렬해서 사전 생성
word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
vocab = list(word_counts.keys())
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 하이퍼파라미터 설정
CONTEXT_SIZE = 2  # 주변 단어 개수
EMBEDDING_DIM = 10  # 단어 임베딩 차원
EPOCHS = 100  # 학습 횟수
LEARNING_RATE = 0.01  # 학습률


# CBOW 데이터셋 생성 함수

def create_cbow_dataset(tokenized_corpus, context_size):
    dataset = []
    for sentence in tokenized_corpus:
        for i in range(context_size, len(sentence) - context_size):
            context = [sentence[j] for j in range(i - context_size, i + context_size + 1) if j != i]
            target = sentence[i]
            dataset.append((context, target))
    return dataset


# CBOW 데이터셋 생성
dataset = create_cbow_dataset(tokenized_corpus, CONTEXT_SIZE)


# 입력 데이터를 텐서로 변환
def context_to_tensor(context, word_to_idx):
    return torch.tensor([word_to_idx[word] for word in context], dtype=torch.long)


def target_to_tensor(target, word_to_idx):
    return torch.tensor([word_to_idx[target]], dtype=torch.long)


# CBOW 모델 정의
# Context size를 고정
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # 여기서  inputs를 넣으면 embeddings가 알아서 index에 맞는 행을 찾아서 추출해줌
        embeds = self.embeddings(inputs).mean(dim=0)  # 주변 단어 임베딩 평균
        out = self.linear(embeds)  # 출력층
        return out


# 모델 초기화
model = CBOW(len(vocab), EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 모델 학습
print("Training CBOW Model...\n")
for epoch in range(EPOCHS):
    total_loss = 0
    for context, target in dataset:
        context_tensor = context_to_tensor(context, word_to_idx)
        target_tensor = target_to_tensor(target, word_to_idx)

        optimizer.zero_grad()
        output = model(context_tensor)
        loss = criterion(output.view(1, -1), target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss:.4f}")

# 테스트: 단어 임베딩 확인
print("\nWord Embeddings:")
for word, idx in word_to_idx.items():
    print(f"{word}: {model.embeddings.weight[idx].detach().numpy()}")

# 모델 저장
torch.save(model.state_dict(), "cbow_model.pth")
print("\nModel saved as cbow_model.pth")
