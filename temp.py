import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

### LOAD AND DESCRIBE ###
df = pd.read_csv("https://raw.githubusercontent.com/allatambov/StatCS25/refs/heads/main/cities.csv")
print(df.head())
print(df.groupby("city")["height"].describe())

df.hist("height", by = "city", edgecolor = "white");
df.boxplot("height", by = "city");

### ANOVA ###
mod = ols('height ~ city', data = df).fit()
tab = sm.stats.anova_lm(mod)
print(tab)
