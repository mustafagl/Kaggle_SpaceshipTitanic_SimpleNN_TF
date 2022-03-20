# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import category_encoders as ce
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
import os


data = pd.read_csv("../input/spaceship-titanic/train.csv")
data_test = pd.read_csv("../input/spaceship-titanic/test.csv")   
display(data_test)
data_test = data_test.fillna(0)
data = data.fillna(0)

display(data_test)
def preprocces(train,state):
    train = pd.DataFrame(train)
    train = train.dropna()
    train = train.reset_index(drop=True)


    train_x = train.iloc[:,1:12]
    
    if state == "train":
        train_y = train.iloc[:,13:14]
        train_y["Transported"] = train_y["Transported"].astype(int)

    train_x["VIP"] = train_x["VIP"].astype(int)
    train_x["CryoSleep"] = train_x["CryoSleep"].astype(int)

    train_x.iloc[:,6:11] = preprocessing.normalize(train_x.iloc[:,6:11], norm='l2',axis=0)
    train_x.iloc[:,4:5] = preprocessing.normalize(train_x.iloc[:,4:5], norm='l2',axis=0)
    y = pd.get_dummies(train_x.Destination, prefix='Destination')
    train_x=pd.concat([train_x, y], axis=1)


    #encoder=ce.HashingEncoder(cols='Cabin',n_components=15)
    #y=encoder.fit_transform(train_x.iloc[:,2:3])
    #train_x=pd.concat([train_x, y], axis=1)
    train_x["Cabin"] = train_x["Cabin"].astype(str)
    mlb = MultiLabelBinarizer()
    y=mlb.fit_transform(train_x["Cabin"])
    y=pd.DataFrame(y)
    display(y)
    train_x=pd.concat([train_x, y], axis=1)


    y = pd.get_dummies(train_x.HomePlanet, prefix='HomePlanet')
    train_x=pd.concat([train_x, y], axis=1)
    del train_x["Cabin"]
    del train_x["Destination"]
    del train_x["HomePlanet"]
    print(len(train_x.columns))
    display(train_x)     
    
    if state=="train":
        return train_x,train_y
    else:
        return train_x


train_x,train_y=preprocces(data,"train")
val_x=train_x.iloc[int(len(train_x.index)*0.95):int(len(train_x.index)),:]

train_x=train_x.iloc[0:int(len(train_x.index)*0.95),:]


val_y=train_y.iloc[int(len(train_y.index)*0.95):int(len(train_y.index)),:]

train_y=train_y.iloc[0:int(len(train_y.index)*0.95),:]

train_x=tf.convert_to_tensor(train_x)
train_y=tf.convert_to_tensor(train_y)
val_x=tf.convert_to_tensor(val_x)
val_y=tf.convert_to_tensor(val_y)

test_x=preprocces(data_test,"test")

#print(train_x)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(37,)),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512, activation='relu'),    
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_x, train_y, epochs=100,validation_data=(val_x,val_y),)

predictions = model.predict(test_x)
y_classes = predictions.argmax(axis=-1)

test_results=pd.DataFrame({'PassengerId':data_test['PassengerId'],'Transported':list(map(bool,y_classes))})
display(test_results)
test_results.to_csv('results.csv' , index=False)







# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
