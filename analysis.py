#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv",sep='\t')
columns=data.columns.tolist()
drop_fuctor = [chr(ord('A')+i) for i in range(0,15)]
drop_index=data.columns.str.startswith(tuple(drop_fuctor))
drop_fuctor = [columns[i] for i in range(drop_index.shape[0]) if  drop_index[i]]
#drop_fuctor=np.array(drop_fuctor)
#drop_fuctor=drop_fuctor.flatten()
data=data.drop(["accuracy","country","source","gender","age","elapsed"],axis=1)
#data=data.drop(drop_fuctor,axis=1)

data_np=data.values

#%%
from sklearn.model_selection import train_test_split
data_np,data_noused=train_test_split(data_np, test_size=0.9, random_state=42)


#%%
from sklearn.decomposition import PCA

pca=PCA(n_components=3)
result_pca=pca.fit_transform(data_np)
plt.scatter(result_pca[:,0],result_pca[:,1],edgecolors="k")
plt.show()

#%%
import umap

for i in range(25,50,25):
    umap_ins=umap.UMAP(n_neighbors=i,metric='canberra',verbose=True,random_state=0,min_dist=0.0)
    result_umap=umap_ins.fit_transform(data_np)
    save_csv_name="./only_likert_scale/csv_files/neighbor_"+str(i)+".csv"
    np.savetxt(save_csv_name,result_umap,delimiter=",")

    plt.close()
    plt.scatter(result_umap[:,0],result_umap[:,1],edgecolors='k')
    save_name="./only_likert_scale/image/neighbor_"+str(i)+".png"
    plt.show()
    #plt.savefig(save_name)



#%%
