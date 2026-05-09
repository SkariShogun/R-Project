import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Spotify Dataset.csv')

print('Columns:', df.columns.tolist())

df = df[["Energy", "Danceability", "Bpm", "Popularity"]]

df.columns = ["energy", "danceability", "tempo", "popularity"]

df = df.dropna()

sns.scatterplot(x="energy", y="popularity", data=df)
plt.title("Energy vs Popularity")
plt.show()

sns.scatterplot(x="danceability", y="popularity", data=df)
plt.title("Danceability vs Popularity")
plt.show()

sns.scatterplot(x="tempo", y="popularity", data=df)
plt.title("Tempo vs Popularity")
plt.show()

corr = df.corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

X = df[["energy", "danceability", "tempo"]]
y = df["popularity"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted")
plt.show()
