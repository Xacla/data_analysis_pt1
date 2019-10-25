#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data=pd.read_csv("data.csv",sep='\t')
data=pd.read_csv("data_recreate.csv")
columns=data.columns.tolist()
drop_fuctor = [chr(ord('A')+i) for i in range(0,15)]
drop_index=data.columns.str.startswith(tuple(drop_fuctor))
drop_fuctor = [columns[i] for i in range(drop_index.shape[0]) if  drop_index[i]]
#drop_fuctor=np.array(drop_fuctor)
#drop_fuctor=drop_fuctor.flatten()
#data=data.drop(["accuracy"],axis=1)
#data=data.drop(drop_fuctor,axis=1)

data_np=data.values

#%%
from sklearn.decomposition import PCA

pca=PCA(n_components=3)
result_pca=pca.fit_transform(data_np)
plt.scatter(result_pca[:,0],result_pca[:,1],edgecolors="k")
plt.show()

#%%
import umap

for i in range(25,50,25):
    umap_ins=umap.UMAP(n_neighbors=i,metric='euclidean',verbose=True,random_state=0,min_dist=0.0)
    result_umap=umap_ins.fit_transform(data_np)
    save_csv_name="./only_likert_scale/csv_files/neighbor_"+str(i)+".csv"
    np.savetxt(save_csv_name,result_umap,delimiter=",")

    plt.close()
    plt.scatter(result_umap[:,0],result_umap[:,1],edgecolors='k')
    save_name="./only_likert_scale/image/neighbor_"+str(i)+".png"
    plt.show()
    #plt.savefig(save_name)



#%%
from sklearn.cluster import KMeans

cluster=KMeans(n_clusters=5).fit(result_umap[:,0:2])

plt.scatter(result_umap[:,0],result_umap[:,1],edgecolors='k',c=cluster.labels_)
plt.show()

#%%
data_1_cluster=data.loc[np.where(cluster.labels_==1)]
describe_datas_cluster1=data_1_cluster.describe()
describe_datas_cluster1.sort_values('std',ascending=True,axis=1)
#%%
data_2_cluster=data.loc[np.where(cluster.labels_==2)]
describe_datas_cluster2=data_2_cluster.describe()
describe_datas_cluster2.sort_values('std',ascending=True,axis=1)
# %%
data_3_cluster=data.loc[np.where(cluster.labels_==3)]
describe_datas_cluster3=data_3_cluster.describe()
describe_datas_cluster3.sort_values('std',ascending=True,axis=1)
#%%
data_4_cluster=data.loc[np.where(cluster.labels_==4)]
describe_datas_cluster4=data_4_cluster.describe()
describe_datas_cluster4.sort_values('std',ascending=True,axis=1)

#%%
data_0_cluster=data.loc[np.where(cluster.labels_==0)]
describe_datas_cluster0=data_0_cluster.describe()
describe_datas_cluster0.sort_values('std',ascending=True,axis=1)

# %%
