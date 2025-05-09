import numpy as np
from collections import Counter

# 예시: 각 모델의 predict 결과 (같은 순서의 테스트 데이터에 대해 예측한 값)
predict_knn = [...]          # e.g., [1, 0, 1, 1, 0]
predict_rf = [...]           # e.g., [1, 1, 1, 0, 0]
predict_nn = [...]           # e.g., [0, 1, 1, 1, 1]

# 리스트로 묶기
all_predictions = [predict_knn, predict_rf, predict_nn]

# 전치하여 각 테스트 인스턴스마다 3개의 예측 결과가 오도록 함
transposed_predictions = list(zip(*all_predictions))

# Majority Voting
def majority_vote(votes):
    count = Counter(votes)
    return count.most_common(1)[0][0]

# 최종 예측 결과 생성
final_predictions = [majority_vote(votes) for votes in transposed_predictions]

# 결과 출력
print("Final Ensemble Predictions:", final_predictions)
