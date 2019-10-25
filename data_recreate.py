#%%
import pandas as pd

data=pd.read_csv("data.csv",sep='\t')
data=data.drop(["country","source","age","elapsed","gender"],axis=1)
# %%
data=data.replace({1:"Y",2:"Y",3:"M",4:"N",5:"N"})
data=data.replace({"Y":1,"M":2,"N":3})

# %%
data.to_csv("data_recreate.csv",index=False)