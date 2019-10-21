#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv",sep='\t')
drop_fuctor = [[chr(ord('A')+i) + str(j) for j in range(1,11)]for i in range(0,14)]
drop_fuctor=np.array(drop_fuctor)
drop_fuctor=drop_fuctor.flatten()
data=data.drop(["accuracy","country","source","gender"],axis=1)
data=data.drop(drop_fuctor,axis=1)

data_np=data.values

#%%
from sklearn.decomposition import PCA

pca=PCA(n_components=3)
result_pca=pca.fit_transform(data_np)
plt.scatter(result_pca[:,0],result_pca[:,1],edgecolors="k")
#plt.show()

#%%
import umap

for i in range(5,100):
    umap_ins=umap.UMAP(n_neighbors=i,metric='euclidean',verbose=True,random_state=0,min_dist=0.1)
    result_umap=umap_ins.fit_transform(data_np)
    save_csv_name="./neighbor_eud/csv_files/neighbor_"+str(i)+".csv"
    np.savetxt(save_csv_name,result_umap,delimiter=",")

    plt.close()
    plt.scatter(result_umap[:,0],result_umap[:,1],edgecolors='k')
    save_name="./neighbor_eud/image/neighbor_"+str(i)+".png"
    plt.savefig(save_name)



#%%
