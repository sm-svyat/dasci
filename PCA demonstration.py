import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import seaborn as sns
from tools import printf #print with separation

'''
All files need to be converted from R.data format first. The can be done by opening up R console and issuing the following commands:
load("/home/sam/Documents/BDS/APM/AppliedPredictiveModeling/data/segmentationOriginal.RData") ('Route dir of .RData File)
write.csv(segmentationOriginal, file = "/home/sam/Documents/BDS/APM/segmentation_original.csv")
'''

seg_org = pd.read_csv('data/segmentation_original.csv')

# Add name numbering
seg_org = seg_org.set_index('ID')

# Get a skew value for all rows where Case == 'Train'
train_skews = seg_org[seg_org.Case == 'Train'].skew()
printf('Skew of first five columns\n\n', train_skews.head())

train = seg_org[seg_org.Case == 'Train']

# Reduce skew for AreaCh1 column by boxcox transformation
AreaCh1_boxcox = stats.boxcox(train.AreaCh1)

printf('Descriptive statistics for AreaCh1\n\n', train.AreaCh1.describe())

# Do principal component analysis for n components
n_components = 10
pca = PCA(n_components)

# Run singular value decomposition
pca.fit(seg_org.select_dtypes(include=['float64', 'int']))

#PCA Components
printf('PCA Components\n\n', pd.DataFrame(pca.components_).T.head())

for i in range(10):
    print(str(i) + ' numerical columns managed to capture: ' +\
    str(sum(pca.explained_variance_ratio_[:i]))[:5] + ' of explained variance')
print('\n'+'#'*70 + '\n')
'''
Near zero variance is not a problem in Scikit learn, and therefore these variables do not need to be removed.
Firstly, predictors with a unique value or with zero variance are constant and therefore will automatically be eliminated by any good regression software, including Python's scikitlearn.
Secondly "Near zero," is utterly meaningless, because internally the variables will be standardised and thus all variances become unity (and are all equal).
'''

#Generate correlation matrix
corr_matrix = seg_org.corr()

printf('First five columns of correlation matrix\n\n', corr_matrix.head())

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(13, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 0, as_cmap=True)

sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=0.01, cbar_kws={"shrink": .5})

plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()