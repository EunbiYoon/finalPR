import pandas as pd

# CSV 불러오기
df = pd.read_csv("ecommerce.csv")

# Purchase_Category 열 이름을 'label'로 변경
df = df.rename(columns={'Purchase_Category': 'label'})

# 변경된 결과를 다시 CSV로 저장
df.to_csv("ecommerce_labeled.csv", index=False)