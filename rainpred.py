# Problem Statement
# This dataset contains about 10 years of daily weather observations from many locations across Australia.RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more. 

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import scipy.stats as stats


# Loading Dataset
df=pd.read_csv("weatherAUS.csv")
pd.set_option('display.max_columns', None)

#lets check how big the data is 
print(df.shape)

# lets check the random five rows of dataset 
print(df.sample(5))

#lets get an overview of feature datatype
print(df.dtypes)

#lets check the missing values if any
print(df.isnull().sum())
# We can clearly see that there are lots of missing values in a dataset which we'll have to fill or drop to start the further analysis.

# DATA CLEANING
# Analysis of Target Feature
# sns.countplot(df['RainTomorrow'])
# plt.show()

numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']
print("Numerical Features Count {}".format(len(numerical_feature)))

discrete_feature=[feature for feature in numerical_feature if len(df[feature].unique())<25]
print("Discrete feature Count {}".format(len(discrete_feature)))

continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
print("Continuous feature Count {}".format(len(continuous_feature)))

categorical_feature = [feature for feature in df.columns if feature not in numerical_feature]
print("Categorical feature Count {}".format(len(categorical_feature)))

binary_categorical_features = [feature for feature in categorical_feature if len(df[feature].unique()) <=3]
print("Binary Categorical features Count {}".format(len(binary_categorical_features)))

# corrmat=df.corr()
# sns.heatmap(corrmat,annot=True)
# plt.show()

# for feature in continuous_feature:
#     sns.distplot(df[feature])
#     plt.xlabel(feature)
#     plt.ylabel("Count")
#     plt.title(feature)
    # plt.show()

# for feature in continuous_feature:
#     sns.boxplot(df[feature])  
#     plt.title(feature)
#     plt.show()  

# handling missing values 
for feature in continuous_feature:
    if (df[feature].isnull().sum())>0:
        df[feature]=df[feature].fillna(df[feature].mean())
print(df.isnull().sum())    

#for discrete features
def mode_nan(df,variable):
    mode=df[variable].value_counts().index[0]
    df[variable].fillna(mode,inplace=True)
    
mode_nan(df,"Cloud9am")
mode_nan(df,"Cloud3pm")
print(df.isnull().sum()) 

# pd.to_numeric(df['RainToday'], downcast='integer')
# pd.to_numeric(df['RainTomorrow'], downcast='integer')
# print(df.head())
print(df.isnull().sum()) 

#handle null values for remaining categorical features
for feature in categorical_feature:
    print ( df[feature].unique())

df["WindGustDir"].replace({'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15},inplace=True)

df["WindDir9am"].replace({'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15},inplace=True)

df["WindDir3pm"].replace({'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15},inplace=True)

df["WindGustDir"] = df["WindGustDir"].fillna(df["WindGustDir"].value_counts().index[0])
df["WindDir9am"] = df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])
df["WindDir3pm"] = df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0])
print(df.isnull().sum())

print(df["RainToday"].unique())
print(df["RainTomorrow"].unique())
df["RainToday"]=pd.get_dummies(df["RainToday"],drop_first=True)
df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"] ,drop_first=True)
print(df.isnull().sum())

df1=df.groupby(["Location"])["RainTomorrow"].value_counts().unstack()
print(df1)
print(df1[1].sort_values(ascending=False))
print(df1[1].sort_values(ascending=False).index)
print(len(df1[1].sort_values(ascending=False).index))
df['Location'].replace({'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,'Woomera':48, 'Uluru':49}
,inplace=True)
print(df['Location'])

df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d", errors = "coerce")
df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day
print(df.head())

# corrmat = df.corr()
#plot heat map
# sns.heatmap(corrmat,annot=True)
# plt.show()

for feature in continuous_feature:
    
    # sns.boxplot(df[feature])
    # plt.title(feature)
    # plt.show()
    print(feature)

#removing outliers from the continuous features
Q1=df.MinTemp.quantile(0.25)
Q3=df.MinTemp.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.MinTemp>lowerlimit)&(df.MinTemp<upperlimit)]
df['MinTemp']=np.where(df['MinTemp']>upperlimit,upperlimit,np.where(df['MinTemp']<lowerlimit,lowerlimit,df['MinTemp'])
)

Q1=df.MaxTemp.quantile(0.25)
Q3=df.MaxTemp.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.MaxTemp>lowerlimit)&(df.MaxTemp<upperlimit)]
df['MaxTemp']=np.where(df['MaxTemp']>upperlimit,upperlimit,np.where(df['MaxTemp']<lowerlimit,lowerlimit,df['MaxTemp'])
)

Q1=df.Rainfall.quantile(0.25)
Q3=df.Rainfall.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Rainfall>lowerlimit)&(df.Rainfall<upperlimit)]
df['Rainfall']=np.where(df['Rainfall']>upperlimit,upperlimit,np.where(df['Rainfall']<lowerlimit,lowerlimit,df['Rainfall']))

Q1=df.Evaporation.quantile(0.25)
Q3=df.Evaporation.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier= df[(df.Evaporation>lowerlimit)&(df.Evaporation<upperlimit)]
df['Evaporation']=np.where(df['Evaporation']>upperlimit,upperlimit,np.where(df['Evaporation']<lowerlimit,lowerlimit,df['Evaporation']))

Q1=df.Sunshine.quantile(0.25)
Q3=df.Sunshine.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Sunshine>lowerlimit)&(df.Sunshine<upperlimit)]
df['Sunshine']=np.where(df['Sunshine']>upperlimit,upperlimit,np.where(df['Sunshine']<lowerlimit,lowerlimit,df['Sunshine']))

Q1=df.WindGustSpeed.quantile(0.25)
Q3=df.WindGustSpeed.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier= df[(df.WindGustSpeed>lowerlimit)&(df.WindGustSpeed<upperlimit)]
df['WindGustSpeed']=np.where(df['WindGustSpeed']>upperlimit,upperlimit,np.where(df['WindGustSpeed']<lowerlimit,lowerlimit,df['WindGustSpeed']))

Q1=df.WindSpeed9am.quantile(0.25)
Q3=df.WindSpeed9am.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.WindSpeed9am>lowerlimit)&(df.WindSpeed9am<upperlimit)]
df['WindSpeed9am']=np.where(df['WindSpeed9am']>upperlimit,upperlimit,np.where(df['WindSpeed9am']<lowerlimit,lowerlimit,df['WindSpeed9am']))

Q1=df.WindSpeed3pm.quantile(0.25)
Q3=df.WindSpeed3pm.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier= df[(df.WindSpeed3pm>lowerlimit)&(df.WindSpeed3pm<upperlimit)]
df['WindSpeed3pm']=np.where(df['WindSpeed3pm']>upperlimit,upperlimit,np.where(df['WindSpeed3pm']<lowerlimit,lowerlimit,df['WindSpeed3pm']))

Q1=df.Humidity9am.quantile(0.25)
Q3=df.Humidity9am.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Humidity9am>lowerlimit)&(df.Humidity9am<upperlimit)]
df['Humidity9am']=np.where(df['Humidity9am']>upperlimit,upperlimit,np.where(df['Humidity9am']<lowerlimit,lowerlimit,df['Humidity9am']))

Q1=df.Humidity3pm.quantile(0.25)
Q3=df.Humidity3pm.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Humidity3pm>lowerlimit)&(df.Humidity3pm<upperlimit)]
df['Humidity3pm']=np.where(df['Humidity3pm']>upperlimit,upperlimit,np.where(df['Humidity3pm']<lowerlimit,lowerlimit,df['Humidity3pm']))

Q1=df.Pressure9am.quantile(0.25)
Q3=df.Pressure9am.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Pressure9am>lowerlimit)&(df.Pressure9am<upperlimit)]
df['Pressure9am']=np.where(df['Pressure9am']>upperlimit,upperlimit,np.where(df['Pressure9am']<lowerlimit,lowerlimit,df['Pressure9am']))

Q1=df.Pressure3pm.quantile(0.25)
Q3=df.Pressure3pm.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Pressure3pm>lowerlimit)&(df.Pressure3pm<upperlimit)]
df['Pressure3pm']=np.where(df['Pressure3pm']>upperlimit,upperlimit,np.where(df['Pressure3pm']<lowerlimit,lowerlimit,df['Pressure3pm']))

Q1=df.Temp9am.quantile(0.25)
Q3=df.Temp9am.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Temp9am>lowerlimit)&(df.Temp9am<upperlimit)]
df['Temp9am']=np.where(df['Temp9am']>upperlimit,upperlimit,np.where(df['Temp9am']<lowerlimit,lowerlimit,df['Temp9am']))

Q1=df.Temp3pm.quantile(0.25)
Q3=df.Temp3pm.quantile(0.75)
print(Q1,Q3)
IQR=Q3-Q1
lowerlimit=Q1-1.5*IQR
upperlimit=Q3+1.5*IQR
print(lowerlimit,upperlimit)
# df_no_outlier=df[(df.Temp3pm>lowerlimit)&(df.Temp3pm<upperlimit)]
df['Temp3pm']=np.where(df['Temp3pm']>upperlimit,upperlimit,np.where(df['Temp3pm']<lowerlimit,lowerlimit,df['Temp3pm']))
# for feature in continuous_feature:
#     sns.boxplot(df[feature])
#     plt.show()
for feature in continuous_feature:
    print(feature)
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[feature].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.show()
# df.to_csv("preprocessed_1.csv", index=False)
print(df.dtypes)
x=df.drop(['RainTomorrow','Date'],axis=1)
y=df['RainTomorrow']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=42)
#random_state=always select same samples after we run multiple times
#stratify=select the samples in balanced way from all the categories
print(df.shape)
print(x_train.shape)
print(x_test.shape)

# sns.countplot(df['RainTomorrow'])
# plt.show()

from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=0)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
print("The number of classes before fit : {}".format((y_train).value_counts()))
print("The number of classes after fit :  {}".format((y_train_res).value_counts()))

# sns.countplot(y_train_res)
# plt.show()

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train_res, y_train_res)
y_pred = logreg.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
cf=pd.DataFrame(confusion_matrix(y_test,y_pred),columns=list(range(0,2)))
print(cf)
print("classification report - ", classification_report(y_test,y_pred))
print("accuracy- " , accuracy_score(y_test,y_pred))
metrics.plot_roc_curve(logreg, x_test, y_test)
metrics.roc_auc_score(y_test, y_pred, average=None) 
plt.show()


from imblearn.under_sampling import NearMiss
nr = NearMiss()  
x_train_miss, y_train_miss = nr.fit_resample(x_train, y_train.ravel())

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train_miss, y_train_miss)
y_pred1 = logreg.predict(x_test)
# sns.countplot(y_train_miss)
# plt.show()

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
cf1=pd.DataFrame(confusion_matrix(y_test,y_pred1),columns=list(range(0,2)))
print(cf1)
print("classification report - ", classification_report(y_test,y_pred1))
print("accuracy- " , accuracy_score(y_test,y_pred1))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_res,y_train_res)
y_pred1 = rf.predict(x_test)
print(confusion_matrix(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
metrics.plot_roc_curve(rf, x_test, y_test)
metrics.roc_auc_score(y_test, y_pred1, average=None) 
plt.show()

import pickle
pickle.dump(rf,open('model.pkl','wb'))



  
