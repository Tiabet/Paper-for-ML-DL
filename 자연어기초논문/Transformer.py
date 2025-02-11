from torch import nn

# encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#
# print(encoder_layer)
#
# transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, )
#
#
# print(transformer)

# mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)

# print(mha)

V = 37000  # Vocabulary Size
L = 512    # Sequence Length
d_model = 512  # Model dimension
h = 8  # Number of attention heads
d_ff = 2048  # Feedforward dimension
N = 6  # Number of layers in encoder and decoder

# 1. Embedding Layers
token_embedding = V * d_model
positional_embedding = L * d_model * 2  # Encoder + Decoder

# 2. Multi-Head Attention (Encoder + Decoder)
# Each attention block has Query, Key, Value projections + Output projection
attention_params_per_layer = 3 * d_model * d_model + d_model * d_model
encoder_attention = N * attention_params_per_layer
decoder_attention = N * attention_params_per_layer
cross_attention = N * attention_params_per_layer  # Cross-Attention in Decoder

# 3. Feedforward Layers (Encoder + Decoder)
feedforward_params_per_layer = 2 * d_model * d_ff
encoder_feedforward = N * feedforward_params_per_layer
decoder_feedforward = N * feedforward_params_per_layer

# 4. Final Linear Projection
# Final Linear Projection은 임베딩과 같은 파라미터를 공유해서 계산하지 않음
# final_projection = d_model * V

# 총 파라미터 수 계산
total_params = (
    token_embedding + positional_embedding +
    encoder_attention + decoder_attention + cross_attention +
    encoder_feedforward + decoder_feedforward
    # final_projection
)

print(total_params)