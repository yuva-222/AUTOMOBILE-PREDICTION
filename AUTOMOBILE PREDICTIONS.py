#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF  AUTOMOBILE
# 
# IMPORTING THE LIBRARIES

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# # DATA PROCESSING

# In[3]:


c_df=pd.read_csv("C:/Users/yuvak/OneDrive/Desktop/ML PROJECTS 1&2 FROM BI/AutoData (1).csv")


# In[4]:


c_df.head()


# In[5]:


c_df.tail()


# In[7]:


c_df.shape


# In[8]:


c_df.dtypes


# In[9]:


c_df.info()


# In[10]:


c_df.describe()


# # DATA CLEANING

# In[11]:


c_df.duplicated().sum()


# In[15]:


c_df.isnull().sum()


# In[16]:


c_df['symboling'].value_counts()


# In[25]:


sns.pairplot(y_vars = 'symboling', x_vars = 'price' ,data=c_df)


# In[26]:


c_df['make'].value_counts()


# In[27]:


c_df['car_company'] = c_df['make'].apply(lambda x:x.split(' ')[0])


# In[28]:


c_df.head()


# In[29]:


c_df = c_df.drop(['make'], axis =1)


# In[30]:


c_df['car_company'].value_counts()


# In[32]:


c_df['car_company'].replace('toyouta', 'toyota',inplace=True)
c_df['car_company'].replace('Nissan', 'nissan',inplace=True)
c_df['car_company'].replace('maxda', 'mazda',inplace=True)
c_df['car_company'].replace('vokswagen', 'volkswagen',inplace=True)
c_df['car_company'].replace('vw', 'volkswagen',inplace=True)
c_df['car_company'].replace('porcshce', 'porsche',inplace=True)


# In[33]:


c_df['car_company'].value_counts()


# mistakes are re-summerized

# In[34]:


c_df['fueltype'].value_counts()


# In[35]:


c_df['aspiration'].value_counts()


# In[36]:


c_df['doornumber'].value_counts()


# In[37]:


c_df['carbody'].value_counts()


# In[38]:


c_df['drivewheel'].value_counts()


# In[39]:


c_df['enginelocation'].value_counts()


# In[40]:


c_df['wheelbase'].value_counts().head()


# In[41]:


sns.distplot(c_df['wheelbase'],color="lime")
plt.show()


# In[42]:


c_df['carlength'].value_counts().head()


# In[43]:


sns.distplot(c_df['carlength'],color="orange")
plt.show()


# In[44]:


c_df['enginetype'].value_counts()


# In[45]:


c_df['cylindernumber'].value_counts()


# In[46]:


def number(x):
    return x.map({'four':4,'six':6,'five':5,'eight':8,'two':2,'three':3,'twelve':12})
c_df['cylindernumber']=c_df[['cylindernumber']].apply(number)


# In[47]:


c_df['cylindernumber'].value_counts()


# In[48]:


c_df['fuelsystem'].value_counts()


# # Data Visulaization:

# In[52]:


c_num = c_df.select_dtypes(include =['int64','float64'])


# In[53]:


c_num.head()


# In[54]:


c_num.info()


# In[57]:


plt.figure(figsize = (25,25))
sns.pairplot(c_num)
plt.show()


# In[61]:


plt.figure(figsize = (25,25))
sns.heatmap(c_df.corr(), annot = True ,cmap = 'YlOrBr')
plt.show()


# -Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower.
# 
# -Price is negatively correlated to symboling, citympg and highwaympg.
# 
# -This suggest that cars having high mileage may fall in the 'economy' cars category, and are priced lower.
# 
# -There are many independent variables which are highly correlated: wheelbase, carlength, curbweight, enginesize etc.. all are positively correlated.

# In[62]:


categorical_cols = c_df.select_dtypes(include = ['object'])
categorical_cols.head()


# In[64]:


plt.figure(figsize = (25,15))
plt.subplot(3,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = c_df,color="orange")
plt.subplot(3,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = c_df,color="lime")
plt.subplot(3,3,3)
sns.boxplot(x = 'carbody', y = 'price', data = c_df,color="green")
plt.subplot(3,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = c_df,color="blue")
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = c_df,color="red")
plt.subplot(3,3,6)
sns.boxplot(x = 'enginetype', y = 'price', data = c_df,color="yellow")
plt.subplot(3,3,7)
sns.boxplot(x = 'fuelsystem', y = 'price', data = c_df,color="black")


# In[71]:


plt.figure(figsize = (25,15))
sns.barplot(x = 'car_company', y = 'price', data = c_df,color="green")


# 1. From the price boxplot it is clear that The brands with the most expensive vehicles in the dataset belong to Bmw,Buick,Jaguar    and porsche.
# 
# 2. Whereas the lower priced cars belong to chevrolet
# 
# 3. The median price of gas vehicles is lower than that of Diesel Vehicles.
# 
# 4. 75th percentile of standard aspirated vehicles have a price lower than the median price of turbo aspirated vehicles.
# 
# 5. Two and four Door vehicles are almost equally priced. There are however some outliers in the price of two-door vehicles.
# 
# 6. Hatchback vehicles have the lowest median price of vehicles in the data set whereas hardtop vehicles have the highest median price.
# 
# 7. The price of vehicles with rear placed engines is significantly higher than the price of vehicles with front placed engines.
# 
# 8. Almost all vehicles in the dataset have engines placed in the front of the vehicle. However, the price of vehicles with rear placed engines is significantly higher than the price of vehicles with front placed engines.
# 
# 9. The median cost of eight cylinder vehicles is higher than other cylinder categories.
# 
# 10. It is clear that vehicles Multi-port Fuel Injection [MPFI] fuelsystem have the highest median price. There are also some outliers on the higher price side having MPFI systems.
# 
# 11. Vehicles with OHCV engine type falls under higher price range.

# # Extract Features and Target

# In[461]:


c_df


# In[462]:


y=c_df["price"]


# In[463]:


X=c_df.drop("price",axis="columns")


# In[464]:


X


# # Features should be of numeric nature:

# In[465]:


c_df.dtypes


# In[466]:


NonNumericColumns=X.columns[X.dtypes=="object"]


# In[467]:


NonNumericColumns


# In[468]:


X=pd.get_dummies(X,columns=NonNumericColumns,drop_first=True)


# # Features should be of type array/ dataframe:

# In[469]:


type(X)


# # Features should have some rows and some columns:

# In[470]:


X.shape


# # Split the dataset- training and testing:

# In[471]:


from sklearn.model_selection import train_test_split


# In[472]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)


# # Features should be on the same scale:

# In[473]:


X.describe()


# In[474]:


from sklearn.preprocessing import MinMaxScaler


# In[475]:


scaler=MinMaxScaler()


# In[476]:


X_train=scaler.fit_transform(X_train)


# In[477]:


X_test=scaler.transform(X_test)


# # Train the model on the training dataset:

# In[478]:


X.shape


# In[479]:


from sklearn.feature_selection import RFE


# In[480]:


from sklearn.linear_model import LinearRegression


# In[481]:


model=LinearRegression()


# In[482]:


rfe_model=RFE(model,15)


# In[483]:


rfe_model.fit(X_train,y_train)


# In[484]:


rfe_model.support_


# In[485]:


X.columns


# In[486]:


Top15Columns=X.columns[rfe_model.support_]


# In[487]:


Top15Columns


# In[488]:


X_train=pd.DataFrame(X_train,columns=X.columns)


# In[489]:


X_test=pd.DataFrame(X_test,columns=X.columns)


# In[490]:


X_train=X_train[Top15Columns]


# In[491]:


X_test=X_test[Top15Columns]


# In[492]:


import statsmodels.api as sm


# In[493]:


X_train


# In[494]:


y_train=pd.DataFrame(y_train).reset_index(drop=True)


# In[495]:


y_train


# In[496]:


pvalues=sm.OLS(y_train,X_train).fit()


# In[497]:


pvalues.summary()


# In[498]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[499]:


list(range(0,len(X_train.columns)))


# In[500]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[501]:


results


# In[502]:


X_train=X_train.drop("boreratio",axis="columns")


# In[503]:


X_test=X_test.drop("boreratio",axis="columns")


# In[504]:


X_train.shape


# In[505]:


X_test.shape


# In[506]:


pvalues=sm.OLS(y_train,X_train).fit()
pvalues.summary()


# In[508]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[509]:


X_train=X_train.drop("stroke",axis="columns")
X_test=X_test.drop("stroke",axis="columns")


# In[510]:


pvalues=sm.OLS(y_train,X_train).fit()
pvalues.summary()


# In[511]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[513]:


X_train=X_train.drop("cylindernumber",axis="columns")
X_test=X_test.drop("cylindernumber",axis="columns")


# In[514]:


pvalues=sm.OLS(y_train,X_train).fit()
pvalues.summary()


# In[515]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[518]:


X_train=X_train.drop("enginesize",axis="columns")
X_test=X_test.drop("enginesize",axis="columns")


# In[519]:


pvalues=sm.OLS(y_train,X_train).fit()
pvalues.summary()


# In[520]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[521]:


X_train=X_train.drop("curbweight",axis="columns")
X_test=X_test.drop("curbweight",axis="columns")


# In[522]:


pvalues=sm.OLS(y_train,X_train).fit()
pvalues.summary()


# In[524]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[552]:


X_train=X_train.drop("carwidth",axis="columns")
X_test=X_test.drop("carwidth",axis="columns")


# In[553]:


pvalues=sm.OLS(y_train,X_train).fit()
pvalues.summary()


# In[554]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[557]:


VIF=[]
for i in range(len(X_train.columns)):
    vifvalue=variance_inflation_factor(X_train.values,i)
    VIF.append(vifvalue)
results=pd.DataFrame()
results["Features"]=X_train.columns
results["VIF"]=VIF
results.sort_values("VIF",ascending=False)


# In[558]:


X_train.columns


# In[559]:


c_df.head()


# In[560]:


model=LinearRegression()


# In[561]:


model.fit(X_train,y_train)


# In[562]:


model.coef_


# In[563]:


results=pd.DataFrame()
results["Features"]=X_train.columns


# In[564]:


X_train.columns


# In[565]:


X_train.columns


# In[568]:


results["Importance"]=model.coef_.reshape(7,)


# In[569]:


results.sort_values(by="Importance",ascending=False)


# Here, above are the features that makes car price huge among all car drives.
