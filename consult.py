import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df1 = pd.read_csv("https://raw.githubusercontent.com/jamovi/r-datasets/refs/heads/master/data/USJudgeRatings.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/jamovi/r-datasets/refs/heads/master/data/InsectSprays.csv")



X = StandardScaler().fit_transform(data)
data_scaled = pd.DataFrame(X, columns = data.columns)
p = data_scaled.shape[1]
pca_names = ["PC" + str(i) for i in range(1, p + 1)]

pca = PCA(n_components = p)
pca_res = pd.DataFrame(pca.fit_transform(data_scaled),
                       columns = pca_names)
pca_var = pca.explained_variance_

rotation = pd.DataFrame(pca.components_).T
rotation.columns = pca_names
rotation.index = data.columns
A = rotation * np.sqrt(pca_var)

