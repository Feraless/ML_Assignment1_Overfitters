import os
import pandas as pd
import numpy as np

dataset_dir = 'Processed files'
files=os.listdir(dataset_dir)
classes = {"Walking":1,"Climbing_up":2,"Climbing_down":3,"Sitting":4,"Standing":5,"Laying":6}

X=[]
Y=[]
for file in files:
    df=pd.read_csv(os.path.join(dataset_dir,file))
    X.append(df.values)

    for i in classes:
        if file.startswith(i):
            Y.append(classes[i])
            break

X=np.array(X)
Y=np.array(Y)

print(X.shape)
print(Y.shape)

