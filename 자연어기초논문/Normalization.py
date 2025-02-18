import numpy as np
import pandas as pd

# 입력 데이터 (Batch, Sequence Length, Feature Size) = (4, 12, 256)
np.random.seed(42)  # 재현성을 위해 고정된 랜덤 시드 사용
input_data = np.random.randn(4, 12, 256)  # 가우시안 정규분포에서 샘플링

# Batch Normalization (Across Batch and Sequence Length for Each Feature)
batch_mean = np.mean(input_data, axis=(0, 1), keepdims=True)  # 평균 계산 (Across Batch and Sequence)
batch_std = np.std(input_data, axis=(0, 1), keepdims=True) + 1e-5  # 분산 계산 (Across Batch and Sequence), 작은 값 추가하여 안정성 확보
batch_norm = (input_data - batch_mean) / batch_std  # 정규화 수행

# Layer Normalization (Across Feature for Each Token)
layer_mean = np.mean(input_data, axis=2, keepdims=True)  # 평균 계산 (Across Feature for Each Token)
layer_std = np.std(input_data, axis=2, keepdims=True) + 1e-5  # 분산 계산 (Across Feature for Each Token)
layer_norm = (input_data - layer_mean) / layer_std  # 정규화 수행
