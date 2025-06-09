import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_excel("ТВиМС 2025.xlsx")
df.head()

df["Тесты1"] = df[["Т1", "Т2", "Т3", "Т4", "Т5"]].mean(axis = 1)
df["Тесты2"] = df[["Т6", "Т7", "Т8"]].mean(axis = 1)

data = df[['КР', 'ДЗ1', 'ДЗ2', 'ДЗ3', 'Семинары', 'Тесты1', 'Тесты2']]
data.corr()

X = StandardScaler().fit_transform(data)
data_scaled = pd.DataFrame(X, columns = data.columns)

data_scaled.head()
data_scaled.describe().round(2)

print(data_scaled.shape)
p = data_scaled.shape[1]

pca_names = ["PC" + str(i) for i in range(1, p + 1)]
print(pca_names)

pca = PCA(n_components = p)
pca_res = pd.DataFrame(pca.fit_transform(data_scaled), 
                       columns = pca_names)
print(pca_res)

pca_var = pca.explained_variance_
print(pca_var.round(2))

print(pca.components_)

rotation = pd.DataFrame(pca.components_).T
rotation.columns = pca_names
rotation.index = data.columns
rotation

print((rotation ** 2).sum(axis = 0))
print((rotation ** 2).sum(axis = 1))
print(np.dot(rotation["PC1"], rotation["PC2"]))

A = rotation * np.sqrt(pca_var)
A

print((A ** 2).sum(axis = 0))
print((A ** 2).sum(axis = 1))
print(np.dot(A["PC1"], A["PC2"]))
print(np.dot(A.loc["ДЗ1", :], A.loc["КР", :]))

print(pca_var.round(2))

plt.plot(range(1, p + 1), pca_var, "o-");
plt.title("Scree plot");
plt.xlabel("Number of components");
plt.ylabel("Eigenvalues (variances)");

pca_var_ratio = pca.explained_variance_ratio_
print((pca_var_ratio * 100).round(2))
print(np.cumsum((pca_var_ratio * 100).round(2)))
