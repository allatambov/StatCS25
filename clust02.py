import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

### SURVEY DATA ###

# https://disk.yandex.ru/i/BdIJFXS_yxXOnw
npk = pd.read_excel("NPK25_share.xlsx")
heroes = npk.columns[4:14]

# results before
npk_before = npk[npk["опрос"] != 14]
npk_before.groupby("профиль")[heroes].sum()
npk_before.groupby("пол")[heroes].sum()

# your results
npk_css = npk[npk["опрос"] == 14]
npk_css[heroes].sum().sort_values(ascending = False)

### CLUSTERING ###

X = npk.iloc[:, 4:14]

# try different distances
hc = linkage(X, metric = "cityblock")
dendrogram(hc);

npk["clust"] = cut_tree(hc, 4)
npk["clust"].value_counts()
npk.groupby("clust")[heroes].sum()







