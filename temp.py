import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

flats = pd.read_csv("https://raw.githubusercontent.com/allatambov/StatCS25/refs/heads/main/flats_cian.csv")
print(flats.head())
print(flats[["price", "square"]].describe())

flats["lprice"].hist(edgecolor = "w", bins = 20);
flats["square"].hist(edgecolor = "w", bins = 20);

flats["lprice"].plot.box();
flats["square"].plot.box();

p_Q1 = flats["lprice"].quantile(0.25)
p_Q3 = flats["lprice"].quantile(0.75)
P_iqr = p_Q3 - p_Q1
PL, PU = p_Q1 - 1.5 * P_iqr, p_Q3 + 1.5 * P_iqr

s_Q1 = flats["square"].quantile(0.25)
s_Q3 = flats["square"].quantile(0.75)
S_iqr = s_Q3 - s_Q1
SL, SU = s_Q1 - 1.5 * S_iqr, s_Q3 + 1.5 * S_iqr

flats_new = flats[(flats["lprice"] >= PL) & (flats["lprice"] <= PU) & 
                 (flats["square"] >= SL) & (flats["square"] <= SU)]

print(m01.params)
print(m01.tvalues)
print(m01.pvalues)
print(m01.conf_int())
print(m01.rsquared)

print(sm.stats.anova_lm(m01))
print(m01.ess)
print(m01.ssr)
print(m01.centered_tss)

residuals = m01.resid
print(residuals.describe())
residuals.hist(bins = 20, edgecolor = "w");

m01.predict()
m01.predict(exog = {"square" : 50})

ring = ['метро Павелецкая', 'метро Курская', 'метро Октябрьская', 
'метро Добрынинская', 'метро Проспект Мира', 'метро Белорусская', 
'метро Краснопресненская', 'метро Киевская', 'метро Парк культуры',
'метро Таганская']

small = flats_new[flats_new["station"].isin(ring)]

m02 = ols("lprice ~ square", data = small).fit()
print(m02.summary())

import seaborn as sns

sns.lmplot(data = small, x = "square", 
           y = "lprice", markers = '.');

sns.lmplot(data = small, x = "square", 
           y = "lprice", 
           markers = '.', 
           col = "ametro", 
           hue = "ametro");
