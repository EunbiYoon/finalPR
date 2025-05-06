import pandas as pd

# CSV 불러오기
df = pd.read_csv("parkinsons.csv")

# 마지막 열 이름을 'label'로 변경
df.rename(columns={df.columns[-1]: "label"}, inplace=True)

# 변경된 결과를 다시 CSV로 저장
df.to_csv("parkinsons_labeled.csv", index=False)