import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from math import sqrt

# low_low = pd.read_csv('C:/Users/Lazar/Downloads/low-low.csv')
# low_medium = pd.read_csv('C:/Users/Lazar/Downloads/low-medium.csv')
# low_high = pd.read_csv('C:/Users/Lazar/Downloads/low-high.csv')
# medium_low = pd.read_csv('C:/Users/Lazar/Downloads/medium-low.csv')
# medium_medium = pd.read_csv('C:/Users/Lazar/Downloads/medium-medium.csv')
# medium_high = pd.read_csv('C:/Users/Lazar/Downloads/medium-high.csv')
# high_low = pd.read_csv('C:/Users/Lazar/Downloads/high-low.csv')
# high_medium = pd.read_csv('C:/Users/Lazar/Downloads/high-medium.csv')
# high_high = pd.read_csv('C:/Users/Lazar/Downloads/high-high.csv')

dfs = [low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high]
df_n = ['low_low', 'low_medium', 'low_high', 'medium_low', 'medium_medium', 'medium_high', 'high_low', 'high_medium',
        'high_high']
targ = 'cr_dens'
num_features = ['ma_morf', 'minor_roads_dens', 'tree_dens', 'poi_dens']
ntarg_f = ['cr_dens', 'ma_morf', 'minor_roads_dens', 'tree_dens', 'poi_dens']
cat_features = ['empt_place', 'parks', 'rail', 'trunk', 'morfotype_unique']

del (low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high)


# hists
for i in num_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        dfs[0][i].hist(bins=50, grid=False, )
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        dfs[1][i].hist(bins=50, grid=False, )
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        dfs[2][i].hist(bins=50, grid=False, )
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        dfs[3][i].hist(bins=50, grid=False, )
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        dfs[4][i].hist(bins=50, grid=False, )
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        dfs[5][i].hist(bins=50, grid=False, )
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        dfs[6][i].hist(bins=50, grid=False, )
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        dfs[7][i].hist(bins=50, grid=False, )
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        dfs[8][i].hist(bins=50, grid=False, )
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()

# log1p hists
for i in num_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        np.log1p(dfs[0][i]).hist(bins=50, grid=False, )
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        np.log1p(dfs[1][i]).hist(bins=50, grid=False, )
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        np.log1p(dfs[2][i]).hist(bins=50, grid=False, )
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        np.log1p(dfs[3][i]).hist(bins=50, grid=False, )
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        np.log1p(dfs[4][i]).hist(bins=50, grid=False, )
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        np.log1p(dfs[5][i]).hist(bins=50, grid=False, )
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        np.log1p(dfs[6][i]).hist(bins=50, grid=False, )
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        np.log1p(dfs[7][i]).hist(bins=50, grid=False, )
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        np.log1p(dfs[8][i]).hist(bins=50, grid=False, )
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()

# log hists
for i in num_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        np.log(dfs[0][i]).hist(bins=50, grid=False, )
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        np.log(dfs[1][i]).hist(bins=50, grid=False, )
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        np.log(dfs[2][i]).hist(bins=50, grid=False, )
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        np.log1p(dfs[3][i]).hist(bins=50, grid=False, )
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        np.log(dfs[4][i]).hist(bins=50, grid=False, )
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        np.log(dfs[5][i]).hist(bins=50, grid=False, )
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        np.log(dfs[6][i]).hist(bins=50, grid=False, )
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        np.log(dfs[7][i]).hist(bins=50, grid=False, )
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        np.log1p(dfs[8][i]).hist(bins=50, grid=False, )
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()

# square root hists
for i in num_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        np.sqrt(dfs[0][i]).hist(bins=50, grid=False, )
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        np.sqrt(dfs[1][i]).hist(bins=50, grid=False, )
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        np.sqrt(dfs[2][i]).hist(bins=50, grid=False, )
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        np.sqrt(dfs[3][i]).hist(bins=50, grid=False, )
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        np.sqrt(dfs[4][i]).hist(bins=50, grid=False, )
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        np.sqrt(dfs[5][i]).hist(bins=50, grid=False, )
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        np.sqrt(dfs[6][i]).hist(bins=50, grid=False, )
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        np.sqrt(dfs[7][i]).hist(bins=50, grid=False, )
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        np.sqrt(dfs[8][i]).hist(bins=50, grid=False, )
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()

# shapiro-wilk tests
from scipy.stats import shapiro
for i in num_features:
        shap = shapiro(np.log1p(dfs[0][i]))
        print(f'{i} (log1p) \n{shap}')
        shap = shapiro(np.log(dfs[0][i]))
        print(f'{i} (log) \n{shap}')
        shap = shapiro(np.sqrt(dfs[0][i]))
        print(f'{i} (sqrt) \n{shap}')
        shap = shapiro(np.cbrt(dfs[0][i]))
        print(f'{i} (cbrt) \n{shap} \n')

# violinplots
for i in cat_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        plt.violinplot(dfs[0][i])
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        plt.violinplot(dfs[1][i])
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        plt.violinplot(dfs[2][i])
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        plt.violinplot(dfs[3][i])
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        plt.violinplot(dfs[4][i])
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        plt.violinplot(dfs[5][i])
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        plt.violinplot(dfs[6][i])
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        plt.violinplot(dfs[7][i])
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        plt.violinplot(dfs[8][i])
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()

# corr matrices
for i in dfs:
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(i[ntarg_f].corr(), annot=True, cbar_kws={'orientation': 'horizontal'})
        mpl.rc('font', size=8)
        plt.show()

# spearman for non-numbered characteristics
from scipy.stats import spearmanr
for i in range(len(dfs)):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(df_n[i], fontsize=14)
        sns.heatmap(spearmanr(dfs[i][targ], dfs[i][cat_features])[0], annot=True, cbar_kws={'orientation': 'horizontal'})
        mpl.rc('font', size=8)
        plt.show()


# second part (working with transformed features) ----------------------------------------------------------------------

# reading data
lnd = pd.read_csv('C:/Users/Lazar/Downloads/london_crime.csv')

lnd['mrd_log'] = np.log(lnd['minor_roads_dens'])
lnd['td_cbrt'] = np.cbrt(lnd['tree_dens'])
lnd['poid_cbrt'] = np.cbrt(lnd['poi_dens'])
lnd['cr_dens'] = np.log(lnd['cr_dens'])

df_n = ['low_low', 'low_medium', 'low_high', 'medium_low', 'medium_medium', 'medium_high', 'high_low', 'high_medium',
        'high_high']
df_no = ['low', 'medium', 'high']
targ = 'cr_dens'
num_features = ['mrd_log', 'td_cbrt', 'poid_cbrt']
ntarg_f = ['cr_dens', 'mrd_log', 'td_cbrt', 'poid_cbrt']
cat_features = ['ma_morf', 'empt_place', 'parks', 'rail', 'trunk', 'morfotype_unique']
ctarg_f = ['cr_dens', 'ma_morf', 'empt_place', 'parks', 'rail', 'trunk', 'morfotype_unique']

low_low = lnd[lnd['cluster'] == 11]
low_medium = lnd[lnd['cluster'] == 12]
low_high = lnd[lnd['cluster'] == 13]
medium_low = lnd[lnd['cluster'] == 21]
medium_medium = lnd[lnd['cluster'] == 22]
medium_high = lnd[lnd['cluster'] == 23]
high_low = lnd[lnd['cluster'] == 31]
high_medium = lnd[lnd['cluster'] == 32]
high_high = lnd[lnd['cluster'] == 33]
low = lnd[lnd['morfotype'] == 1]
medium = lnd[lnd['morfotype'] == 2]
high = lnd[lnd['morfotype'] == 3]

dfs = [low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high]
df = [low, high, medium]

del (low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high,
     low, high, medium, lnd)

# number features analyze ----------------------------------------------------------------------------------------------
# hists for 2-factor
for i in num_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        dfs[0][i].hist(bins=50, grid=False, )
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        dfs[1][i].hist(bins=50, grid=False, )
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        dfs[2][i].hist(bins=50, grid=False, )
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        dfs[3][i].hist(bins=50, grid=False, )
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        dfs[4][i].hist(bins=50, grid=False, )
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        dfs[5][i].hist(bins=50, grid=False, )
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        dfs[6][i].hist(bins=50, grid=False, )
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        dfs[7][i].hist(bins=50, grid=False, )
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        dfs[8][i].hist(bins=50, grid=False, )
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_hist_2_factor.png')

# hists for 1-factor
for i in num_features:
        fig = plt.figure(figsize=(5, 10))

        plt.subplot(3, 1, 1)
        df[0][i].hist(bins=50, grid=False, )
        plt.title("low", fontsize=10)

        plt.subplot(3, 1, 2)
        df[1][i].hist(bins=50, grid=False, )
        plt.title("medium", fontsize=10)

        plt.subplot(3, 1, 3)
        df[2][i].hist(bins=50, grid=False, )
        plt.title("high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_hist_1_factor.png')


# shapiro-wilk tests for 2-factor
from scipy.stats import shapiro
for i in num_features:
        for n in range(len(dfs)):
                shap = shapiro(dfs[n][i])
                print(f'{i} for ({df_n[n]}) cluster \n {shap} \n')
        print('\n')

# shapiro-wilk tests for 1-factor
for i in num_features:
        for n in range(len(df)):
                shap = shapiro(dfs[n][i])
                print(f'{i} for ({df_n[n]}) cluster \n {shap} \n')
        print('\n')


# corr matrices for 2-factor
for i in range(len(dfs)):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(df_n[i], fontsize=14)
        sns.heatmap(dfs[i][ntarg_f].corr(), annot=True, cbar_kws={'orientation': 'horizontal'})
        mpl.rc('font', size=8)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_corrmatrix_2_factor.png')

# corr matrices for 1-factor
for i in range(len(df)):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(df_no[i], fontsize=14)
        sns.heatmap(df[i][ntarg_f].corr(), annot=True, cbar_kws={'orientation': 'horizontal'})
        mpl.rc('font', size=8)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_corrmatrix_1_factor.png')


# categorical features analyze -----------------------------------------------------------------------------------------
# violinplots for 2-factor
for i in cat_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        plt.violinplot(dfs[0][i])
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        plt.violinplot(dfs[1][i])
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        plt.violinplot(dfs[2][i])
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        plt.violinplot(dfs[3][i])
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        plt.violinplot(dfs[4][i])
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        plt.violinplot(dfs[5][i])
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        plt.violinplot(dfs[6][i])
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        plt.violinplot(dfs[7][i])
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        plt.violinplot(dfs[8][i])
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_violinplot_2_factor.png')

# violinplots for 1-factor
for i in cat_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(1, 3, 1)
        plt.violinplot(df[0][i])
        plt.title("low-low", fontsize=10)

        plt.subplot(1, 3, 2)
        plt.violinplot(df[1][i])
        plt.title("low-medium", fontsize=10)

        plt.subplot(1, 3, 3)
        plt.violinplot(df[2][i])
        plt.title("low-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_violinplot_1_factor.png')


# spearman for non-numbered characteristics (2-factor)
from scipy.stats import spearmanr
for i in range(len(dfs)):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(df_n[i], fontsize=14)
        sns.heatmap(spearmanr(dfs[i][targ], dfs[i][cat_features])[0], annot=True,
                    cbar_kws={'orientation': 'horizontal'}, xticklabels=ctarg_f, yticklabels=ctarg_f)
        mpl.rc('font', size=8)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_spearman_2_factor.png')

# spearman for non-numbered characteristics (1-factor)
for i in range(len(df)):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(df_no[i], fontsize=14)
        sns.heatmap(spearmanr(df[i][targ], df[i][cat_features])[0], annot=True, cbar_kws={'orientation': 'horizontal'})
        mpl.rc('font', size=8)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_spearman_1_factor.png')


# second point first part (clearing the numerical values) --------------------------------------------------------------
# second part (working with transformed features) ----------------------------------------------------------------------

# reading data
lnd = pd.read_csv('C:/Users/Lazar/Downloads/london_crime.csv')

lnd['mrd_log'] = np.log(lnd['minor_roads_dens'])
lnd['td_cbrt'] = np.cbrt(lnd['tree_dens'])
lnd['poid_cbrt'] = np.cbrt(lnd['poi_dens'])
lnd['cr_dens'] = np.log(lnd['cr_dens'])

lnd = lnd[lnd['td_cbrt'] > 0.1]
lnd = lnd[lnd['poid_cbrt'] > 0.1]

df_n = ['low_low', 'low_medium', 'low_high', 'medium_low', 'medium_medium', 'medium_high', 'high_low', 'high_medium',
        'high_high']
df_no = ['low', 'medium', 'high']
targ = 'cr_dens'
num_features = ['mrd_log', 'td_cbrt', 'poid_cbrt']
ntarg_f = ['cr_dens', 'mrd_log', 'td_cbrt', 'poid_cbrt']
cat_features = ['ma_morf', 'empt_place', 'parks', 'rail', 'trunk', 'morfotype_unique']
ctarg_f = ['cr_dens', 'ma_morf', 'empt_place', 'parks', 'rail', 'trunk', 'morfotype_unique']
clusters = ['low_low', 'low_medium', 'low_high', 'medium_low', 'medium_medium', 'medium_high', 'high_low',
            'high_medium', 'high_high']

low_low = lnd[lnd['cluster'] == 11]
low_medium = lnd[lnd['cluster'] == 12]
low_high = lnd[lnd['cluster'] == 13]
medium_low = lnd[lnd['cluster'] == 21]
medium_medium = lnd[lnd['cluster'] == 22]
medium_high = lnd[lnd['cluster'] == 23]
high_low = lnd[lnd['cluster'] == 31]
high_medium = lnd[lnd['cluster'] == 32]
high_high = lnd[lnd['cluster'] == 33]
low = lnd[lnd['morfotype'] == 1]
medium = lnd[lnd['morfotype'] == 2]
high = lnd[lnd['morfotype'] == 3]

dfs = [low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high]
df = [low, high, medium]

del (low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high,
     low, high, medium)

# hists for 2-factor
for i in num_features:
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(3, 3, 1)
        dfs[0][i].hist(bins=50, grid=False, )
        plt.title("low-low", fontsize=10)

        plt.subplot(3, 3, 2)
        dfs[1][i].hist(bins=50, grid=False, )
        plt.title("low-medium", fontsize=10)

        plt.subplot(3, 3, 3)
        dfs[2][i].hist(bins=50, grid=False, )
        plt.title("low-high", fontsize=10)

        plt.subplot(3, 3, 4)
        dfs[3][i].hist(bins=50, grid=False, )
        plt.title("medium-low", fontsize=10)

        plt.subplot(3, 3, 5)
        dfs[4][i].hist(bins=50, grid=False, )
        plt.title("medium-medium", fontsize=10)

        plt.subplot(3, 3, 6)
        dfs[5][i].hist(bins=50, grid=False, )
        plt.title("medium-high", fontsize=10)

        plt.subplot(3, 3, 7)
        dfs[6][i].hist(bins=50, grid=False, )
        plt.title("high-low", fontsize=10)

        plt.subplot(3, 3, 8)
        dfs[7][i].hist(bins=50, grid=False, )
        plt.title("high-medium", fontsize=10)

        plt.subplot(3, 3, 9)
        dfs[8][i].hist(bins=50, grid=False, )
        plt.title("high-high", fontsize=10)

        mpl.rc('font', size=8)
        fig.suptitle(i, fontsize=14)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_hist_2_factor.png')

# corr matrices for 2-factor
for i in range(len(dfs)):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(df_n[i], fontsize=14)
        sns.heatmap(dfs[i][ntarg_f].corr(), annot=True, cbar_kws={'orientation': 'horizontal'})
        mpl.rc('font', size=8)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/{i}_corrmatrix_2_factor.png')

# VIF for 2-factor
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

for i in dfs:
        y, X = dmatrices('cr_dens ~ mrd_log+td_cbrt+poid_cbrt', data=i, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, n) for n in range(X.shape[1])]
        vif['variable'] = X.columns
        print(f'{vif} \n')

# CatBoost regression for 2-factor clusters ----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

deep = [5, 7, 9]
temp = [0.05, 0.1, 0.5]
cluster = []
deep_index = []
temp_index = []
mae_index = []
rmse_index = []

# predicting the best model for the each cluster
for b in range(len(dfs)):
        X_train, X_test, y_train, y_test = train_test_split(dfs[b][num_features+cat_features], dfs[b][targ],
                                                            test_size=0.2, shuffle=False, random_state=13)

        from catboost import CatBoostRegressor
        for n in deep:
            for i in temp:
                        clf = CatBoostRegressor(random_state=13, task_type="GPU", loss_function='RMSE',
                                                depth=n, learning_rate=i)
                        clf.fit(X_train, y_train, cat_features=cat_features, plot=True)
                        acc_rmse = mse(y_test, clf.predict(X_test), squared=False)
                        acc_mae = mae(y_train, clf.predict(X_train))
                        mae_index.append(acc_mae)
                        rmse_index.append(acc_rmse)
                        deep_index.append(n)
                        temp_index.append(i)
                        cluster.append(clusters[b])

test_scores = pd.DataFrame()
test_scores['cluster'] = cluster
test_scores['deep'] = deep_index
test_scores['temp'] = temp_index
test_scores['mae'] = mae_index
test_scores['rmse'] = rmse_index


# final results and probing the best hyperparameters for CatBoost models ----------------------------------------------
for i in range(4, 9):
        X_train, X_test, y_train, y_test = train_test_split(dfs[i][num_features+cat_features], dfs[i][targ],
                                                                    test_size=0.2, shuffle=False, random_state=13)

        # clf_1 = CatBoostRegressor(random_state=13, task_type="GPU", loss_function='RMSE', depth=7, learning_rate=0.5)
        # clf_1.fit(X_train, y_train, cat_features=cat_features, plot=True)
        clf_2 = CatBoostRegressor(random_state=13, task_type="GPU", loss_function='RMSE', depth=9, learning_rate=0.5)
        clf_2.fit(X_train, y_train, cat_features=cat_features, plot=True)

        # importance_rates = pd.DataFrame()
        # importance_rates['low_high'] = clf_1.feature_importances_
        importance_rates[clusters[i]] = clf_2.feature_importances_
        # importance_rates.index = clf_1.feature_names_

for i in range(len(importance_rates)):
        importance_rates['mean'][i] = np.mean(importance_rates[i:i+1])


# building scatterplots
lnd = pd.read_csv('C:/Users/Lazar/Downloads/london_crime.csv')

lnd['mrd_log'] = np.log(lnd['minor_roads_dens'])
lnd['td_cbrt'] = np.cbrt(lnd['tree_dens'])
lnd['poid_cbrt'] = np.cbrt(lnd['poi_dens'])
lnd['cr_dens'] = np.log(lnd['cr_dens'])

lnd = lnd[lnd['td_cbrt'] > 0.1]
lnd = lnd[lnd['poid_cbrt'] > 0.1]

df_n = ['low_low', 'low_medium', 'low_high', 'medium_low', 'medium_medium', 'medium_high', 'high_low', 'high_medium',
        'high_high']
targ = 'cr_dens'
rel_f = ['mrd_log', 'td_cbrt', 'poid_cbrt', 'morfotype_unique']


low_low = lnd[lnd['cluster'] == 11]
low_medium = lnd[lnd['cluster'] == 12]
low_high = lnd[lnd['cluster'] == 13]
medium_low = lnd[lnd['cluster'] == 21]
medium_medium = lnd[lnd['cluster'] == 22]
medium_high = lnd[lnd['cluster'] == 23]
high_low = lnd[lnd['cluster'] == 31]
high_medium = lnd[lnd['cluster'] == 32]
high_high = lnd[lnd['cluster'] == 33]
low = lnd[lnd['morfotype'] == 1]
medium = lnd[lnd['morfotype'] == 2]
high = lnd[lnd['morfotype'] == 3]

dfs = [low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high]
df = [low, high, medium]

del (low_low, low_medium, low_high, medium_low, medium_medium, medium_high, high_low, high_medium, high_high,
     low, high, medium)

for i in range(len(dfs)):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax = axs[0][0]
        ax.scatter(dfs[i]['mrd_log'], dfs[i][targ], alpha=0.5, s=2)
        ax.set_title('mrd_log', fontsize=10)
        ax.set_xlabel('Плотность явления')
        ax.set_ylabel('Концентрация преступлений')

        ax = axs[0][1]
        ax.scatter(dfs[i]['td_cbrt'], dfs[i][targ], alpha=0.5, s=2)
        ax.set_title('td_cbrt', fontsize=10)
        ax.set_xlabel('Плотность явления')
        ax.set_ylabel('Концентрация преступлений')

        ax = axs[1][0]
        ax.scatter(dfs[i]['poid_cbrt'], dfs[i][targ], alpha=0.5, s=2)
        ax.set_title('poid_cbrt', fontsize=10)
        ax.set_xlabel('Плотность явления')
        ax.set_ylabel('Концентрация преступлений')

        ax = axs[1][1]
        ax.scatter(dfs[i]['morfotype_unique'], dfs[i][targ], alpha=0.5, s=2)
        ax.set_title('morfotype_unique', fontsize=10)
        ax.set_xlabel('Плотность явления')
        ax.set_ylabel('Концентрация преступлений')

        mpl.rc('font', size=8)
        fig.suptitle(df_n[i], fontsize=14)
        plt.show()
        fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/scatter_{df_n[i]}.png')


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

ax = axs[0][0]
ax.scatter(lnd['mrd_log'], lnd[targ], alpha=0.5, s=1)
ax.set_title('mrd_log', fontsize=10)
ax.set_xlabel('Плотность явления')
ax.set_ylabel('Концентрация преступлений')

ax = axs[0][1]
ax.scatter(lnd['td_cbrt'], lnd[targ], alpha=0.5, s=1)
ax.set_title('td_cbrt', fontsize=10)
ax.set_xlabel('Плотность явления')
ax.set_ylabel('Концентрация преступлений')
ax = axs[1][0]

ax.scatter(lnd['poid_cbrt'], lnd[targ], alpha=0.5, s=1)
ax.set_title('poid_cbrt', fontsize=10)
ax.set_xlabel('Плотность явления')
ax.set_ylabel('Концентрация преступлений')

ax = axs[1][1]
# ax.heatmap(lnd['morfotype_unique'], lnd[targ], alpha=0.5, s=1)
# ax.imshow(lnd['morfotype_unique'], lnd[targ])
ax = sns.heatmap([lnd['morfotype_unique'], lnd[targ]])
ax.set_title('morfotype_unique', fontsize=10)
ax.set_xlabel('Плотность явления')
ax.set_ylabel('Концентрация преступлений')

mpl.rc('font', size=8)
fig.suptitle('lnd', fontsize=14)
plt.show()
fig.savefig(f'C:/Users/Lazar/Downloads/london_graphs/scatter_lnd.png')
