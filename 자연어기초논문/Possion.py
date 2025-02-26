import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 평균 (λ) 설정
lambda_value = 3

# x 값 설정 (포아송 분포는 이산적이므로 정수 값 사용)
x_values = np.arange(0, 15)

# 확률 질량 함수 (PMF) 계산
y_values = poisson.pmf(x_values, lambda_value)

# 그래프 그리기
plt.bar(x_values, y_values, alpha=0.7)
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.title(f'Poisson Distribution (λ={lambda_value})')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 그래프 표시
plt.show()
