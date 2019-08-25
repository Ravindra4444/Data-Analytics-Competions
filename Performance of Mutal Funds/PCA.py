import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv('partB_data.csv')
df = df1.copy()








# graph for correlation between two features

# x = df['turnover']
# y = df['NAV']
# colors=('r')
# plt.xlabel('turnover')
# plt.ylabel('NAV')
# plt.scatter(x,y,c=colors)
# plt.show()


# x = df['GDP']
# y = df['NAV']
# colors=('b')
# plt.xlabel('GDP')
# plt.ylabel('NAV')
# plt.scatter(x,y,c=colors)
# plt.show()

# x = df['F_days']
# y = df['NAV']
# colors=('r')
# plt.xlabel('F_days')
# plt.ylabel('NAV')
# plt.scatter(x,y,c=colors)
# plt.show()

# x = df['CPI']
# y = df['GDP']
# colors=('r', 'b')
# plt.xlabel('CPI')
# plt.ylabel('GDP')
# plt.scatter(x,y,c=colors)
# plt.show()

popped = df.pop('Fund_Name')
popped = df.pop('MF_name')
popped = df.pop('Asstes_in_Cr')
popped = df.pop('Min_investment')
#adding new features

df['stock*shares'] = pd.Series(np.multiply(df['stock_index'], df['shares']), index=df.index)
df['stock*GDP'] = pd.Series(np.multiply(df['stock_index'], df['GDP']), index=df.index)
df['turnover*F_days'] = pd.Series(np.multiply(df['turnover'], df['F_days']), index=df.index)
df['log_shares'] = pd.Series(np.log(df['shares']), index=df.index)
df['turnover_squared'] = pd.Series(np.multiply(df['turnover'], df['turnover']), index=df.index)

cols = list(df)
cols.insert(14, cols.pop(cols.index('NAV')))
print(cols)
df=df.ix[:,cols]

df = (df - df.mean())/(df.max() - df.min())

correlation = df.corr()
plt.figure(figsize=(13,13))
sns.heatmap(correlation, vmax=1, square=True, annot=True,cmap='viridis')
plt.title('correlation between different features')
#plt.show()


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x = df.iloc[:,0:14]
y = df.iloc[:, 14]
X = scaler.fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)
#print(pca.get_covariance())
explained_variance = pca.explained_variance_ratio_
print(explained_variance)


with plt.style.context('dark_background'):
	plt.figure(figsize=(12,12))

	plt.bar(range(14), explained_variance, alpha=0.5, align='center',
		label='individual explained variance')
	plt.ylabel('Explained Variance Ratio')
	plt.xlabel('Principal components')
	plt.legend(loc='best')
	plt.tight_layout()
plt.show()


pca =PCA(n_components=9)
X_new = pca.fit_transform(X)
pca.get_covariance()
# print(pca.get_covariance())

explained_variance=pca.explained_variance_ratio_


#plot of PCA data
with plt.style.context('dark_background'):
	plt.figure(figsize=(12,12))

	plt.bar(range(9), explained_variance, alpha=0.5, align='center',
		label='individual explained variance')
	plt.ylabel('Explained Variance Ratio')
	plt.xlabel('Principal components')
	plt.legend(loc='best')
	plt.tight_layout()
plt.show()

#learning data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

estimators = np.arange(10, 200, 10)

scores=[]
for n in estimators:
	model.set_params(n_estimators=n)
	model.fit(X_train, y_train)
	scores.append(model.score(X_test, y_test))
	print(scores)
