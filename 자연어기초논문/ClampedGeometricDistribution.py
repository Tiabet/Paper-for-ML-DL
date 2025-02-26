import matplotlib.pyplot as plt
import numpy as np

p = 0.5
max_val = 10
ks = np.arange(0, max_val + 1)
probs = np.zeros(max_val + 1)

# k=0 ~ max_val-1 까지 확률 계산
for k in range(max_val):
    probs[k] = (1 - p) * p**k
# k = max_val에서 나머지 확률 할당
probs[max_val] = p**max_val

plt.bar(ks, probs, color='skyblue', edgecolor='black')
plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Clamped Geometric Distribution (p=0.5, max=10)')
plt.xticks(ks)
plt.show()
