from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd

# check which datasets can be imported
list_available_datasets()

# import dataset
heart_disease = fetch_ucirepo(id=45) # dataset number 45 in uci
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = heart_disease.data.features
y = heart_disease.data.targets
# train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

# save as csv file
df = pd.concat([X, y], axis=1)
df.rename(columns={df.columns[-1]: "label"}, inplace=True)
df.to_csv("heart_disease.csv", index=False)
