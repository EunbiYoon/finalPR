import numpy as np
import pandas as pd
from collections import Counter
import sys, os

# 상위 디렉토리에서 knn, tree, nn 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 알고리즘 모듈 import
from knn_algorithm import knn
from random_forest import tree
from neural_network import nn

# 데이터셋 이름
DATASET_NAME = "digits"

# === 각 알고리즘 수행 ===
print("🔷 KNN Algorithm Will Start For Ensemble Algorithm")
predict_knn_df = knn.main(DATASET_NAME)  # DataFrame
print("✅ KNN Algorithm End.\n🔷 Random Forest Algorithm Will Start For Ensemble Algorithm")
predict_rf = tree.main(DATASET_NAME)     # list or np.array
print("✅ Random Forest End.\n🔷 Neural Network Will Start For Ensemble Algorithm")
predict_nn = nn.neural_network(DATASET_NAME)  # list
print("✅ Neural Network End.")

# === Majority Voting ===
# choose k
choose_k_knn = predict_knn_df.iloc[:, 0].tolist()  # 첫 번째 열: k=1
all_predictions = list(zip(choose_k_knn, predict_rf, predict_nn))

def majority_vote(votes):
    count = Counter(votes)
    return count.most_common(1)[0][0]

final_predictions = [majority_vote(v) for v in all_predictions]
df_final = pd.DataFrame({'FinalEnsemble': final_predictions})

# === ExcelWriter로 시트별 저장 (index 통일 없이) ===
output_path = f'ensemble_results_{DATASET_NAME}.xlsx'
with pd.ExcelWriter(output_path) as writer:
    predict_knn_df.to_excel(writer, sheet_name="KNN", index=True)
    pd.DataFrame(predict_rf, columns=["RandomForest"]).to_excel(writer, sheet_name="RandomForest", index=True)
    pd.DataFrame(predict_nn, columns=["NeuralNetwork"]).to_excel(writer, sheet_name="NeuralNetwork", index=True)
    df_final.to_excel(writer, sheet_name="FinalEnsemble", index=True)

print(f"📁 Excel file saved to: {output_path}")
