# importing pandas module
import pandas as pd
import numpy as np 
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier

# reading csv files
df = pd.read_csv("../input/train-dataset/train.csv")
test_df = pd.read_csv("../input/dataset/test.csv")

# dropping rows containing null values
df = df.dropna(inplace=False)

# df

# seperate date and time from pickup_time
df["pickup_time"] = pd.to_datetime(df["pickup_time"])
df["drop_time"] = pd.to_datetime(df["drop_time"])

#pickup = df["pickup_time"].str.split(" ", n = 1, expand = True)

df["pickup_date_only"]= df.pickup_time.dt.date
df["pickup_time_only"]= df.pickup_time.dt.hour

df["drop_date_only"]= df.drop_time.dt.date
df["drop_time_only"]= df.drop_time.dt.hour


# convert to correct format
df["pickup_date_only"] = pd.to_datetime(df["pickup_date_only"])

df["drop_date_only"] = pd.to_datetime(df["drop_date_only"])

# convert week days to number 0 to 6
df["pickup_day_in_number"] = df["pickup_date_only"].dt.dayofweek

df["drop_day_in_number"] = df["drop_date_only"].dt.dayofweek
#df

# add one hot encoding to numeric days format
pickup_one_hot = pd.get_dummies(df["pickup_day_in_number"], prefix = "pickup_day")

df = df.join(pickup_one_hot)

drop_one_hot = pd.get_dummies(df["drop_day_in_number"], prefix = "drop_day")

df = df.join(drop_one_hot)

# df

def sphere_dist(pick_lat, pick_lon, drop_lat, drop_lon):
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pick_lat, pick_lon, drop_lat, drop_lon = map(np.radians, [pick_lat, pick_lon, drop_lat, drop_lon])
    
    #Compute distances along lat, lon dimensions
    distance_lat = drop_lat - pick_lat
    distance_lon = drop_lon - pick_lon
    
    #Compute haversine distance
    a = np.sin(distance_lat/2.0)**2 + np.cos(pick_lat) * np.cos(drop_lat) * np.sin(distance_lon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))

df['distance'] = sphere_dist(df['pick_lat'], df['pick_lon'], 
                                   df['drop_lat'] , df['drop_lon'])

# df.head()

# take copy of the dataframe
df_copy = df.copy()

# df_copy

# divide into training and test set
train_train_set = df_copy.sample(frac=0.8, random_state=0)

# drop some invalid rows
train_train_set = train_train_set.drop(train_train_set[(train_train_set.meter_waiting_till_pickup == 0) & (train_train_set.additional_fare != 5)].index)

train_test_set = df_copy.drop(train_train_set.index)


train_train_set["label"] = train_train_set["label"].map({"correct": 1, "incorrect": 0})
train_train_output = train_train_set[["label"]].astype(int)

# train_train_output

train_train_input = train_train_set.drop(["pickup_time", "drop_time", "pickup_date_only", "drop_date_only", "pickup_day_in_number", "drop_day_in_number", "label"], axis=1)


train_test_set["label"] = train_test_set["label"].map({"correct": 1, "incorrect": 0})
train_test_output = train_test_set[["label"]].astype(int)

# train_test_output

train_test_input = train_test_set.drop(["pickup_time", "drop_time", "pickup_date_only", "drop_date_only", "pickup_day_in_number", "drop_day_in_number", "label"], axis=1)

# This is for test.csv file

# seperate date and time from pickup_time
test_df["pickup_time"] = pd.to_datetime(test_df["pickup_time"])
test_df["drop_time"] = pd.to_datetime(test_df["drop_time"])

#pickup = df["pickup_time"].str.split(" ", n = 1, expand = True)

test_df["pickup_date_only"]= test_df.pickup_time.dt.date
test_df["pickup_time_only"]= test_df.pickup_time.dt.hour

test_df["drop_date_only"]= test_df.drop_time.dt.date
test_df["drop_time_only"]= test_df.drop_time.dt.hour

# convert to correct format
test_df["pickup_date_only"] = pd.to_datetime(test_df["pickup_date_only"])

test_df["drop_date_only"] = pd.to_datetime(test_df["drop_date_only"])

# convert week days to number 0 to 6
test_df["pickup_day_in_number"] = test_df["pickup_date_only"].dt.dayofweek

test_df["drop_day_in_number"] = test_df["drop_date_only"].dt.dayofweek

# test_df

# add one hot encoding to numeric days format
test_pickup_one_hot = pd.get_dummies(test_df["pickup_day_in_number"], prefix = "pickup_day")

test_df = test_df.join(test_pickup_one_hot)

test_drop_one_hot = pd.get_dummies(test_df["drop_day_in_number"], prefix = "drop_day")

test_df = test_df.join(test_drop_one_hot)

# test_df

test_df['distance'] = sphere_dist(test_df['pick_lat'], test_df['pick_lon'], 
                                   test_df['drop_lat'] , test_df['drop_lon'])

# take copy of the dataframe
test_df_copy = test_df.copy()

# test_df_copy

test_csv_input = test_df_copy.drop(["pickup_time", "drop_time", "pickup_date_only", "drop_date_only", "pickup_day_in_number", "drop_day_in_number"], axis=1)

# test_csv_input

%%time
model_Fare = XGBClassifier(
learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
model_Fare.fit(train_train_input, train_train_output)

model_Fare.score(train_test_input, train_test_output)

test_pred_Fare=model_Fare.predict(test_csv_input)

submission_df = pd.read_csv("../input/dataset/sample_submission.csv", index_col="tripid")

submission_df["prediction"] = test_pred_Fare
submission_df

submission_df.to_csv('my_submission4.csv', index=True)