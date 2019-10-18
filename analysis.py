#%%
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv",sep='\t')
data=data.drop(["accuracy","country","source","gender","age","elapsed"],axis=1)

data_np=data.values
#%%
from sklearn.decomposition import PCA

pca=PCA(n_components=3)
result_pca=pca.fit_transform(data_np)
plt.scatter(result_pca[:,0],result_pca[:,1],edgecolors="k")
#plt.show()

#%%
import umap

umap_ins=umap.UMAP(n_neighbors=50,metric='canberra',verbose=True)
result_umap=umap_ins.fit_transform(data)

plt.scatter(result_umap[:,0],result_umap[:,1],edgecolors='k')
plt.show()

#%%
