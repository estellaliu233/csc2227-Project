import os
import random
import warnings
from datetime import timedelta
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM
import math



# Set seeds to make the experiment more reproducible.
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)



# reading dataset
data = pd.read_csv(r"index.csv",skiprows=5)
data.head(3)
data.info()
with open('info_output.csv','w') as file_out:
    data.info(buf=file_out)
info = pd.read_csv(r"info_output.csv",sep=" ", header=None)
area_dict=dict(zip(info.iloc[:,[4]],info.iloc[:,[31]]))

# trim dataset
temp = data.loc[(data.Unit=="Hrs")]
newdata = temp[["EffectiveDate","PricePerUnit","Location","Instance Type", "Operating System"]]

# data quality check
newdata.isnull().sum()
newdata.dropna(inplace=True)

# get the unique values
for col in newdata[["Location","Instance Type","Operating System"]]:
    print(pd.unique(newdata[col]))

# Check the trend in pricing values with time
grouped = newdata[["EffectiveDate","PricePerUnit"]].groupby("EffectiveDate").mean().reset_index()
fig,ax = plt.subplots(figsize=(18,6))
ax.plot(grouped["EffectiveDate"],grouped["PricePerUnit"],".-")
ax.set_title("Price Changes", fontsize=16)
ax.set_xlabel("Effective Date")
ax.set_ylabel("Price per Unit")
ax.grid(True)
plt.show()

# find the max and min value
print("max value is" + newdata["EffectiveDate"].max())
print("min value is" + newdata["EffectiveDate"].min())

# backfill missing observations
L=pd.unique(newdata["Location"])
I=pd.unique(newdata["Instance Type"])
O=pd.unique(newdata["Operating System"])
E=pd.unique(newdata["EffectiveDate"])
mux = pd.MultiIndex.from_product([L,I,O,E],names=["Location","Instance Type","Operating System","EffectiveDate"])
gdata = newdata.groupby(["Location","Instance Type","Operating System","EffectiveDate"]).mean().reset_index()
gdata = gdata.set_index(["Location","Instance Type","Operating System","EffectiveDate"]).reindex(mux).groupby(level=0).bfill().reset_index()
gdata = gdata.sort_values(by=["Location","Instance Type","Operating System","EffectiveDate"])
gdata["EffectiveDate"]= pd.to_datetime(gdata["EffectiveDate"])

# assign date_block_num
temp1 = pd.date_range(end = gdata["EffectiveDate"].max()+timedelta(days=31),start = gdata["EffectiveDate"].min(),freq="M")
temp2 = temp1.to_pydatetime()
temp3 = pd.Series(temp2)
# gdata["EffectiveDate"]= pd.to_datetime(gdata["EffectiveDate"])
cutoff_date = gdata["EffectiveDate"].max()-timedelta(days=365)
test_length = gdata[(gdata["EffectiveDate"] >= cutoff_date)].shape[0]
gdata["month_year"] = gdata["EffectiveDate"].dt.to_period("M")
temp3 = pd.DataFrame(pd.to_datetime(temp3))
list1 = list(range(60))
temp3["date_block_num"] = pd.Series(list1)
temp3.columns=["month_year","date_block_num"]
temp3["month_year"] = temp3["month_year"].dt.to_period("M")
temp3["month_year"] = temp3["month_year"].astype(str)
temp3["month_year"]=pd.to_datetime(temp3["month_year"])
grouped2 = gdata[["month_year","Location","Instance Type","Operating System","PricePerUnit"]].groupby(["month_year","Location","Instance Type","Operating System"]).mean().reset_index()
grouped2["month_year"] = grouped2["month_year"].astype(str)
grouped2["month_year"] = pd.to_datetime(grouped2["month_year"])
grouped3 = grouped2.merge(temp3,
               on="month_year",how="left")


grouped4 = grouped3.pivot_table(index=["Location","Instance Type","Operating System"],
                           columns="date_block_num",values="PricePerUnit",fill_value=0).reset_index()
m = 87087 - grouped4.shape[0]
temp4 = grouped4.sample(n=m)
grouped5 = pd.concat([grouped4,temp4])


# using one year as a series for unique pair

first_month = 46
last_month = 59
serie_size = 12
data_series=[]

for index, row in grouped5.iterrows():
    for month1 in range((last_month - (first_month + serie_size)) + 1):
        serie = [row['Location'], row['Instance Type'], row["Operating System"]]
        for month2 in range(serie_size + 1):
            serie.append(row[month1 + first_month + month2])
        data_series.append(serie)

columns = ['Location', 'Instance Type', 'Operating System']
[columns.append(i) for i in range(serie_size)]
columns.append('label')

data_series = pd.DataFrame(data_series, columns=columns)
data_series.head()

data_series=data_series.drop(["Location","Instance Type","Operating System"],axis=1)

labels = data_series["label"]
data_series.drop("label",axis=1,inplace=True)




train,valid,Y_train,Y_valid = train_test_split(data_series,labels.values,test_size=1740,random_state=0)

X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))

print("Train set reshaped", X_train.shape)
print("Validation set reshaped", X_valid.shape)

serie_size =  X_train.shape[1]
n_features =  X_train.shape[2]

epochs = 5
batch = 1
lr = 0.001

save_best = ModelCheckpoint("best_weights.h5",monitor="val_loss",save_best_only=True,
                            save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.25,
                              patience=5,min_lr=0.0001,verbose=1)

model = Sequential()
model.add(LSTM(1,batch_input_shape=(batch,serie_size,n_features),stateful=True))
model.add(Dense(1))

model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='mse', optimizer=optimizer)

history = model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),
                         epochs=epochs,batch_size=batch,
                         verbose=1)

model.save("my_model.h5")
Predictions = model.predict(X_valid, batch_size=1)


fig= plt.figure(figsize=(10,6))
plt.title("Price Changes")
plt.xlabel("Time")
plt.ylabel("Price")
plt.plot(Predictions)
plt.plot(Y_valid)
plt.show()

mse = sklearn. metrics. mean_squared_error(Y_valid, Predictions)
rmse = math. sqrt(mse)


smodel = load_model("my_model.h5")

