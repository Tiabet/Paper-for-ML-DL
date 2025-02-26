import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 이미지 크기 설정
fig, ax = plt.subplots(figsize=(6, 3))

# Encoder 박스
encoder_box = patches.FancyBboxPatch((0.2, 0.6), 2.0, 0.6, boxstyle="round,pad=0.1",
                                     edgecolor="black", facecolor="#FF9999", label="Pre-trained Encoder")
ax.add_patch(encoder_box)
ax.text(1.2, 0.9, "Pre-trained Encoder", fontsize=10, ha="center", va="center", color="black")

# Decoder 박스
decoder_box = patches.FancyBboxPatch((3.0, 0.6), 2.0, 0.6, boxstyle="round,pad=0.1",
                                     edgecolor="black", facecolor="#99CCFF", label="Pre-trained Decoder")
ax.add_patch(decoder_box)
ax.text(4.0, 0.9, "Pre-trained Decoder", fontsize=10, ha="center", va="center", color="black")

# 입력 토큰들
tokens = ["A", "B", "C", "D", "E"]
token_positions = [0.6, 1.0, 1.4, 1.8, 2.2]

# Encoder 입력 화살표
for pos, token in zip(token_positions, tokens):
    ax.annotate("", xy=(0.2, pos), xytext=(-0.5, pos), arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(-0.6, pos, token, fontsize=12, ha="center", va="center", color="black")

# Encoder → Decoder 연결 화살표
ax.annotate("", xy=(3.0, 1.4), xytext=(2.2, 1.4), arrowprops=dict(arrowstyle="->", color="black"))

# Decoder 입력 (시작 토큰 포함)
decoder_tokens = ["<s>"] + tokens
decoder_positions = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]

for pos, token in zip(decoder_positions, decoder_tokens):
    ax.text(2.9, pos, token, fontsize=12, ha="right", va="center", color="black")

# Decoder 출력 화살표 및 Token Label
for pos, token in zip(decoder_positions[1:], tokens):
    ax.annotate("", xy=(5.2, pos), xytext=(5.0, pos), arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(5.3, pos, "label", fontsize=10, ha="left", va="center", color="black")

# 축 숨기기
ax.set_xlim(-1, 6)
ax.set_ylim(0, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")

# 저장 및 표시
plt.show()
