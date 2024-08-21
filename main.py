import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras

url = "https://www.openml.org/data/download/22102255/dataset"
r = requests.get(url, allow_redirects=True)
with open("dataset.txt", "wb") as f:
    f.write(r.content)

data = []

with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@") or line.startswith("%") or line == "":
            continue
        data.append(line)
    
columns = []

with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@ATTRIBUTE"):
            columns.append(line.split(" ")[1])

with open("df.csv", "w") as f:
    f.write(",".join(columns))
    f.write("\n")
    f.write("\n".join(data))

df = pd.read_csv("df.csv")
df.columns = columns

# Convert categorical columns to numeric
df['t_win'] = df.round_winner.astype("category").cat.codes
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

# Calculate correlation only for numeric columns
correlations = df[numeric_columns].corr()

print(correlations['t_win'].apply(abs).sort_values(ascending=False).iloc[:25])

selected_columns = []

for col in columns+["t_win"]:
    try:
        if abs(correlations[col]['t_win']) > 0.15:
                selected_columns.append(col)
    except KeyError:
                pass

df_selected = df[selected_columns]

plt.figure(figsize=(18, 12))
sns.heatmap(df_selected.corr().sort_values(by="t_win"), annot=True, cmap="YlGnBu")

plt.show()

df_selected.hist(figsize=(18, 12))

plt.show()

X, y = df_selected.drop(['t_win'], axis=1), df_selected['t_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

print("ACCURACY FOR THE MODEL SO FAR: ", knn.score(X_test_scaled, y_test))

param_grid = {
     "n_neighbors": list(range(5, 17, 2)),
     "weights": ["uniform", "distance"]
}

knn = KNeighborsClassifier(n_jobs=4)

clf = RandomizedSearchCV(knn, param_grid, n_jobs=4, n_iter=3, verbose=2, cv=3)
clf.fit(X_train_scaled, y_train)

knn = clf.best_estimator_

print("ACCURACY FOR THE MODEL AFTER KNeighborsClassifier: ", knn.score(X_test_scaled, y_test))

forest = RandomForestClassifier(n_jobs=4)
forest.fit(X_train_scaled, y_train)

print("ACCURACY FOR THE MODEL AFTER RandomForestClassifier(BEST MODEL): ", forest.score(X_test_scaled, y_test))
