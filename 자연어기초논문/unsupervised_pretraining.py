import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random


# -----------------------------
# 1. 간단한 토크나이저 및 데이터셋 정의
# -----------------------------
class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            # 예시를 위한 간단한 vocabulary. 실제로는 훨씬 큰 vocab이 필요합니다.
            self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
            self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
            self.next_index = 4
        else:
            self.vocab = vocab
            self.idx2word = {i: w for w, i in vocab.items()}
            self.next_index = max(vocab.values()) + 1

    def tokenize(self, text):
        # 아주 간단한 공백 기반 토크나이징
        tokens = text.strip().split()
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return ids

    def add_sentence(self, text):
        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_index
                self.idx2word[self.next_index] = token
                self.next_index += 1

    def decode(self, ids):
        return " ".join([self.idx2word.get(i, "<UNK>") for i in ids])


# 예시 데이터셋: 여러 문장으로 구성
texts = [
    "hello world this is a test",
    "how are you doing today",
    # "this is another example sentence",
    "unsupervised fine tuning of a language model",
    "we are building a simple gpt model"
]

# 토크나이저 생성 및 vocabulary 업데이트
tokenizer = SimpleTokenizer()
for text in texts:
    tokenizer.add_sentence(text)
vocab_size = tokenizer.next_index


# Dataset 클래스 정의 (다음 토큰 예측을 위한 데이터셋)
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=10):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            # 각 문장에 <BOS>와 <EOS> 토큰 추가
            encoded = [tokenizer.vocab["<BOS>"]] + tokenizer.encode(text) + [tokenizer.vocab["<EOS>"]]
            # 길이가 seq_len보다 작으면 padding, 크면 슬라이딩 윈도우 방식으로 분할
            for i in range(0, len(encoded), seq_len):
                chunk = encoded[i:i + seq_len]
                if len(chunk) < seq_len:
                    chunk = chunk + [tokenizer.vocab["<PAD>"]] * (seq_len - len(chunk))
                self.data.append(torch.tensor(chunk))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 입력과 타겟을 오른쪽 쉬프트하여 구성
        x = self.data[idx][:-1]  # 마지막 토큰 제외
        y = self.data[idx][1:]  # 첫 토큰 제외
        return x, y


# Dataset 및 DataLoader 생성
seq_len = 10
dataset = TextDataset(texts, tokenizer, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# -----------------------------
# 2. 간단한 GPT 스타일 모델 정의
# -----------------------------
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=seq_len):
        super(SimpleGPT, self).__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # 여러 Transformer 블록을 쌓기
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        token_embeddings = self.token_embed(x)  # [batch_size, seq_len, embed_dim]

        # 포지션 인덱스 생성
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.pos_embed(positions)

        x = token_embeddings + pos_embeddings  # [batch_size, seq_len, embed_dim]
        x = self.dropout(x)

        # Transformer Encoder는 기본적으로 인과적 마스크(causal mask)를 제공하지 않으므로, 직접 마스크를 생성합니다.
        # 여기서는 upper-triangular 마스크를 만들어 미래 정보를 보지 못하도록 함
        # nn.TransformerEncoderLayer는 [seq_len, batch_size, embed_dim]를 입력으로 받으므로 transpose 해줍니다.
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        # 인과적 마스크 생성
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        logits = self.fc_out(x)  # [batch_size, seq_len, vocab_size]
        return logits


# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGPT(vocab_size=vocab_size).to(device)

# -----------------------------
# 3. 학습 설정 및 unsupervised fine-tuning (다음 토큰 예측)
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["<PAD>"])

num_epochs = 50

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)  # [batch_size, seq_len-1]
        y = y.to(device)  # [batch_size, seq_len-1]

        optimizer.zero_grad()
        # 모델 출력: [batch_size, seq_len, vocab_size]
        logits = model(x)
        # 예측 대상은 각 시점의 다음 토큰이므로, 마지막 차원을 vocab_size로 맞춰줌.
        # 입력과 타겟의 길이를 맞추기 위해 x와 y 모두 seq_len-1 길이로 구성하는 방식도 가능함.
        # 여기서는 간단하게 전체 시퀀스를 대상으로 loss를 계산합니다.
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# -----------------------------
# 4. 간단한 생성 테스트 (inference)
# -----------------------------
def generate_text(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()
    with torch.no_grad():
        # prompt를 토큰화 및 인코딩 (BOS 토큰 포함)
        encoded = [tokenizer.vocab["<BOS>"]] + tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded, device=device).unsqueeze(0)  # [1, seq_len]

        for _ in range(max_new_tokens):
            logits = model(input_ids)
            # 마지막 토큰의 logits 사용
            next_token_logits = logits[0, -1, :]
            # 간단하게 argmax로 선택 (샘플링 등 다양한 기법 사용 가능)
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # 종료 토큰 만나면 종료
            if next_token.item() == tokenizer.vocab["<EOS>"]:
                break

        generated_ids = input_ids[0].tolist()
        # BOS 토큰은 제외하여 디코딩 (후처리)
        return tokenizer.decode(generated_ids[1:])


# 생성 테스트
prompt = ""
generated_text = generate_text(model, tokenizer, prompt)
print("Generated text:", generated_text)
