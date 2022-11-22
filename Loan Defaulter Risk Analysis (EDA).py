#!/usr/bin/env python
# coding: utf-8

# ## import Basic Libaries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


# In[3]:


plt.style.use("fivethirtyeight")


# ## Load Data

# In[4]:



df_raw = pd.read_csv('application_data.csv')
df = df_raw.copy()


# In[10]:


description = pd.read_csv('columns_description.csv' , index_col=0)
description


# In[11]:


df.head(5)


# In[12]:


df.shape


# In[13]:


# Function to describe a columns name
def description_func(row=None):
    descr = description[description['Row']==row]['Description']
    return print(descr.str.cat())


# In[14]:


df.info(all)


# ### Missing Values

# In[15]:


# Number of Missing values in %
def find_missing_values(df):
    df_nan = ((df.isnull().sum()/len(df))*100).sort_values()
    df_miss = df_nan[df_nan > 0]
    return df_miss


# In[16]:


# Plot Missing values
def plot_missing_values(df_miss):
    df_miss.plot(kind='bar', figsize=(17,6))
    plt.title('Missing Values by Column \n', size=22)
    plt.xlabel('column Name', size=17)
    plt.ylabel('Percentage of Missing Values\n', size=17)
    plt.show()
    return None


# In[17]:


df_miss = find_missing_values(df)


# In[18]:


plot_missing_values(df_miss)


# In[19]:


msno.bar(df, label_rotation=90)


# In[20]:


def plot_null_values(df):
    df_null = pd.DataFrame(df.isnull().sum()/len(df)*100).reset_index()
    df_null.columns = ['Coluns Name', 'Null Values Percentage']

    fig = plt.figure(figsize=(22,7))
    ax = sns.pointplot(x='Coluns Name', y='Null Values Percentage', data=df_null, color='blue')
    plt.xticks(rotation=90, size=7)
    ax.axhline(50, ls='--', color='red')
    plt.title("Percenatage of Missing Values in df\n",size=22)
    plt.xlabel("Columns Name\n", size=17)
    plt.ylabel("Percenatge of Missing Values\n", size=17)
    plt.show()
    return None


# In[21]:


plot_null_values(df)


# In[ ]:





# In[22]:


# Column with Missing values greather than 50%

df_null_50 = pd.DataFrame(df_miss[df_miss>50]).reset_index()
df_null_50.columns=['Columns Name','Percentage of Missing Values']
df_null_50.style.background_gradient(cmap='viridis')


# In[23]:


# Drop columnof Missing values more than 50%

#df_null_50.set_index('Columns Name').index

df_50_drop = ['HOUSETYPE_MODE', 'LIVINGAREA_AVG', 'LIVINGAREA_MODE',
       'LIVINGAREA_MEDI', 'ENTRANCES_AVG', 'ENTRANCES_MODE', 'ENTRANCES_MEDI',
       'APARTMENTS_MEDI', 'APARTMENTS_AVG', 'APARTMENTS_MODE',
       'WALLSMATERIAL_MODE', 'ELEVATORS_MEDI', 'ELEVATORS_AVG',
       'ELEVATORS_MODE', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_AVG',
       'NONLIVINGAREA_MEDI', 'EXT_SOURCE_1', 'BASEMENTAREA_MODE',
       'BASEMENTAREA_AVG', 'BASEMENTAREA_MEDI', 'LANDAREA_MEDI',
       'LANDAREA_AVG', 'LANDAREA_MODE', 'OWN_CAR_AGE', 'YEARS_BUILD_MODE',
       'YEARS_BUILD_AVG', 'YEARS_BUILD_MEDI', 'FLOORSMIN_AVG',
       'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'LIVINGAPARTMENTS_AVG',
       'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI', 'FONDKAPREMONT_MODE',
       'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MEDI',
       'NONLIVINGAPARTMENTS_MODE', 'COMMONAREA_MODE', 'COMMONAREA_AVG',
       'COMMONAREA_MEDI']
df = df.drop(df_50_drop, axis=1)


# In[24]:


df.corr().style.background_gradient(cmap='viridis')


# In[25]:


plt.figure(figsize=(25,20), dpi=100)
sns.heatmap(df.corr(), cmap='viridis')
plt.show()


# ### Categorial and missing values

# In[26]:


num_cols = [col for col in df.columns if df[col].dtypes!='object']
cat_cols = [col for col in df.columns if df[col].dtypes=='object']


# In[27]:


num_cols


# In[28]:


df['CNT_CHILDREN'].value_counts()


# In[29]:


sns.countplot(x=df['CNT_CHILDREN'])


# In[30]:


df_AMT = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']


# In[31]:


df_miss = find_missing_values(df)


# In[32]:


df_miss


# In[33]:


plot_missing_values(df_miss)


# In[34]:


abs(df['DAYS_LAST_PHONE_CHANGE'])


# In[35]:


df[df['DAYS_LAST_PHONE_CHANGE'].isnull()]


# In[36]:


df['CNT_FAM_MEMBERS'].unique()


# In[37]:


df[df['CNT_FAM_MEMBERS'].isnull()]


# In[38]:


from sklearn.impute import SimpleImputer


# In[39]:


def SimpleImputer_funct(col,strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df[col] = imputer.fit_transform(df[col].values.reshape(-1,1))[:,0]
    return df[col].isnull().sum()


# In[40]:


df.columns


# In[41]:


df[['EXT_SOURCE_2','EXT_SOURCE_3', 'TARGET']].corr()


# In[42]:


sns.heatmap(df[['EXT_SOURCE_2','EXT_SOURCE_3', 'TARGET']].corr(),cmap='viridis',annot=True)

# Very low correlation between column 'EXT_SOURCE_2','EXT_SOURCE_3' and Target, so we can drop thse columns


# In[43]:


df = df.drop(['EXT_SOURCE_2','EXT_SOURCE_3'], axis=1)


# In[44]:


# Correlation between AMT and Taget

col_FLAG = ['AMT_INCOME_TOTAL','AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']


# In[45]:


df_AMT = df[col_FLAG + ['TARGET']]


# In[46]:


plt.figure(figsize=(8,6),dpi=100)
sns.heatmap(df[col_FLAG + ['TARGET']].corr(), cmap='viridis', annot=True)


# In[47]:


# Correlation between FLAG_DOCUMENT and Taget

col_FLAG = ['FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21']


# In[48]:


df_FLAG = df[col_FLAG + ['TARGET']]


# In[49]:


df_FLAG.loc[:,'TARGET'] = df_FLAG['TARGET'].replace({1:"Defaulter", 0: "Repayer"})


# In[50]:


df_FLAG


# In[51]:


import itertools


# In[52]:


fig = plt.figure(figsize=(22,20))
for cols ,index in itertools.zip_longest(col_FLAG, range(len(col_FLAG))):
    plt.subplot(4 , 5, index+1)
    ax = sns.countplot(x=df_FLAG[cols], hue=df_FLAG["TARGET"], palette=["r","g"])
    plt.yticks(fontsize=8)
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.title(index)


# In[53]:


# only, the FLAG_DOCUMENT_3 column will be kept, because the rest is useless for the continuation of the analysis
col_FLAG.remove('FLAG_DOCUMENT_3')


# In[54]:


# Drop the rest a of columns of FLAG_DOCUMENT_X

df = df.drop(col_FLAG, axis=1)


# In[55]:


# df[col_Name + ['TARGET']]


# In[56]:


num_cols = [col for col in df.columns if df[col].dtypes!='object']
cat_cols = [col for col in df.columns if df[col].dtypes=='object']


# In[57]:


df[num_cols].columns


# In[58]:


# correlation between mobile phone, work phone etc, email, Family members and Region rating
col_contact = ['FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']


# In[59]:


fig = plt.figure(figsize=(8,6),dpi=100)
sns.heatmap(df[col_contact + ['TARGET']].corr(), cmap='viridis', annot=True)


# In[60]:


# There ist no relevant correlations betwenn these Features and the Target variable, drop all these columns

df = df.drop(col_contact, axis=1)


# In[61]:


df.head()


# In[62]:


df.columns


# In[63]:


# box plotting the values of AMT_ANNUITY
sns.boxplot(y=df['AMT_ANNUITY'])
plt.yscale('log')
plt.show()


# In[64]:


print(f"Mean: {round(df['AMT_ANNUITY'].mean(),2)}",'\n')
print(f"Median: {round(df['AMT_ANNUITY'].median(),2)}",'\n')
print(df['AMT_ANNUITY'].describe())


# In[65]:


(df['AMT_ANNUITY']).isnull().sum()


# ## Feature Engineering

# In[66]:


# Convert Negative into positiv Days

col_date = ['DAYS_BIRTH','DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']

for col in col_date:
    df[col] = abs(df[col])


# In[67]:


df[['DAYS_BIRTH','DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']].head(2)


# In[68]:


# 10 biggest AMT_INCOME_TOTAL

df['AMT_INCOME_TOTAL'].sort_values(ascending=False).head(10)


# In[69]:


# Creating a new categorical variable from a continue variable 'AMT_INCOME_TOTAL'

ranges = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k'
         ,'700k-800k','800k-900k','900k-1M', '1M Above']
bins = range(0,12,1)

df['AMT_INCOME_TOTAL_RANGE'] = pd.cut(df['AMT_INCOME_TOTAL']/100000, bins=bins, labels=ranges)


# In[70]:


round(df['AMT_INCOME_TOTAL_RANGE'].value_counts(normalize=True)*100,2)


# In[71]:


# Drop the columns 'AMT_INCOME_TOTAL'
df = df.drop('AMT_INCOME_TOTAL', axis=1)


# In[72]:


# Creating a new categorical variable from a continue variable 'AMT_CREDIT'


# In[73]:


df['AMT_CREDIT'].sort_values(ascending=False)


# In[74]:


bins = range(0,12,1)
ranges = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k'
         ,'700k-800k','800k-900k','900k-1M', '1M Above']
df['AMT_CREDIT_RANGE'] = pd.cut(df['AMT_CREDIT']/100000, labels=ranges, bins=bins)


# In[75]:


df['AMT_CREDIT_RANGE']


# In[76]:


# Filling missing values with median
SimpleImputer_funct(col='AMT_ANNUITY', strategy='median')


# In[77]:


df_miss = find_missing_values(df)
print(df_miss)


# #### Analysis of CNT_FAM_MEMBERS

# In[78]:


df['CNT_FAM_MEMBERS'].value_counts(dropna=False)


# In[79]:


#ploting the data from CNT_FAM_MEMBERS coloumn in a box plot to detect outliners
sns.boxplot(y=df['CNT_FAM_MEMBERS'])
plt.yscale('log')
plt.show()


# In[80]:


print(f"Mean: {round(df['CNT_FAM_MEMBERS'].mean(),2)}",'\n')
print(f"Median: {df['CNT_FAM_MEMBERS'].median()}",'\n')
print(df['CNT_FAM_MEMBERS'].describe())


# In[81]:


df['CNT_FAM_MEMBERS'].isnull().sum()


# #### Analysis of Code gender

# In[82]:


df['CODE_GENDER'].value_counts(dropna=False)


# In[83]:


sns.countplot(x=df['CODE_GENDER'].sort_values())


# In[84]:


# Replace XNA with mode
df.loc[df['CODE_GENDER']=='XNA', 'CODE_GENDER'] = 'F'


# In[85]:


df['CODE_GENDER'].value_counts()


# In[86]:


df_ORGA = pd.DataFrame(df['ORGANIZATION_TYPE'].value_counts()).reset_index()
df_ORGA.columns = ['ORGANIZATION_TYPE', 'Values']
print(df_ORGA)


# In[87]:


print(f"Mode: {df['ORGANIZATION_TYPE'].mode()}",'\n')
print(df['ORGANIZATION_TYPE'].describe())


# In[88]:


df['AMT_INCOME_TOTAL_RANGE'].value_counts(dropna=False)


# In[89]:


plt.figure(figsize=(12,6))
sns.countplot(x=df['AMT_INCOME_TOTAL_RANGE'] )


# In[90]:


df.loc[df['AMT_INCOME_TOTAL_RANGE']=='NaN', 'AMT_INCOME_TOTAL_RANGE'] = '100K-200K'


# In[91]:


df['AMT_INCOME_TOTAL_RANGE'].value_counts(dropna=False).plot(kind='bar')


# In[92]:


df['AMT_INCOME_TOTAL_RANGE'].fillna(value = df['AMT_INCOME_TOTAL_RANGE'].mode(), inplace=True)


# In[93]:


df_miss = find_missing_values(df)
print(df_miss)


# In[94]:


df['AMT_INCOME_TOTAL_RANGE'].value_counts(dropna=False)


# #### Analysis of AMT_GOODS_PRICE

# In[95]:


#box plotting the values of AMT_GOODS_PRICE

sns.boxplot(y=df['AMT_GOODS_PRICE'])
plt.yscale('log')
plt.show()


# In[96]:


print(f"Median: {df['AMT_GOODS_PRICE'].median()}\n")
print(f"Mean: {df['AMT_GOODS_PRICE'].mean()}\n")
print(f"Max: {df['AMT_GOODS_PRICE'].max()}\n")
print(f"Min: {df['AMT_GOODS_PRICE'].min()}\n")
print(df['AMT_GOODS_PRICE'].describe())


# In[97]:


df['AMT_GOODS_PRICE'].value_counts(dropna=False)


# In[98]:


df['AMT_GOODS_PRICE'].fillna(value = df['AMT_GOODS_PRICE'].median(), inplace=True)


# In[99]:


df_miss = find_missing_values(df)
print(df_miss)


# In[100]:


df['AMT_INCOME_TOTAL_RANGE'].value_counts(dropna=False)


# In[101]:


df.info(all)


# In[102]:


df=df.astype({'AMT_INCOME_TOTAL_RANGE': object, 'AMT_CREDIT_RANGE': object})


# ####  Analysis of AMT_REQ_CREDIT_BUREAU_DAY and Outlier detection

# In[103]:


sns.boxplot(y=df['AMT_REQ_CREDIT_BUREAU_DAY'])
plt.show()


# In[104]:


print(df['AMT_REQ_CREDIT_BUREAU_DAY'].describe())


# In[105]:


##----Removing outliers for the column below----##
cols_of_outliers=['AMT_REQ_CREDIT_BUREAU_DAY']
for i in cols_of_outliers:
    percentiles = df[i].quantile([0.01 ,0.99]).values
    df.loc[df[i] <= percentiles[0]] = percentiles[0]
    df.loc[df[i] >= percentiles[1]] = percentiles[1]


# In[106]:


sns.boxplot(y=df['AMT_REQ_CREDIT_BUREAU_DAY'])
plt.show()


# In[107]:


df_miss.index


# #### change the positiv to negativ sign of 'DAYS_LAST_PHONE_CHANGE'

# In[108]:


df['DAYS_LAST_PHONE_CHANGE'] = abs(df['DAYS_LAST_PHONE_CHANGE'])


# In[109]:


df['DAYS_LAST_PHONE_CHANGE']


# #### Columns 'CNT_FAM_MEMBERS'

# In[110]:


df[ 'CNT_FAM_MEMBERS'].unique()


# In[111]:


df[ 'CNT_FAM_MEMBERS'].mode()


# In[112]:


df[ 'CNT_FAM_MEMBERS'] = df[ 'CNT_FAM_MEMBERS'].replace('nan', '0.0')


# In[113]:


df['CNT_FAM_MEMBERS'].value_counts(dropna=False).plot


# In[114]:


# Drop the row with nan, beacuse it ios just one Row
df = df.dropna(subset=['CNT_FAM_MEMBERS'])


# In[115]:


df['CNT_FAM_MEMBERS'].value_counts(dropna=False).plot(kind='bar')


# #### Columns 'AMT_INCOME_TOTAL_RANGE'

# In[116]:


df['AMT_INCOME_TOTAL_RANGE'].value_counts(dropna=False).plot(kind='bar')


# In[117]:


df['AMT_INCOME_TOTAL_RANGE'].value_counts(dropna=False)


# In[118]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR']

for x in amount:
    df.loc[: ,x].fillna(df[x].median(), inplace=True)


# In[119]:


df = df.drop(amount, axis=1)


# In[120]:


df_miss = find_missing_values(df)
print(df_miss)


# In[121]:


floors = ['FLOORSMAX_MEDI', 'FLOORSMAX_MODE', 'FLOORSMAX_AVG']

for x in floors:
    df.loc[: ,x].fillna(df[x].mean(), inplace = True)


# In[122]:


df_miss = find_missing_values(df)
print(df_miss)


# In[123]:


years = ['YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_AVG']

for x in years:
    df.loc[: ,x].fillna(df[x].mean(), inplace = True)


# In[124]:


df_miss = find_missing_values(df)
print(df_miss)


# In[125]:


df['OCCUPATION_TYPE'].value_counts(dropna=False)


# In[126]:


description_func(row='OCCUPATION_TYPE')


# In[127]:


# Replace the mode '0.0' with 'unemployed' and also 'NaN' with mode 'unemployed'
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace({0.0: 'unemployed', np.NaN: 'unemployed'})


# In[128]:


df['OCCUPATION_TYPE'].value_counts(dropna=False)


# In[129]:


df_miss = find_missing_values(df)
print(df_miss)


# #### Columns 'OBS_30_CNT_SOCIAL_CIRCLE'

# In[130]:


df['OBS_30_CNT_SOCIAL_CIRCLE'].unique()


# In[131]:


df['OBS_30_CNT_SOCIAL_CIRCLE'].value_counts(dropna=False)


# In[132]:


description_func(row='OBS_30_CNT_SOCIAL_CIRCLE')


# In[133]:


description_func(row='OBS_60_CNT_SOCIAL_CIRCLE')


# In[134]:


col_mixt = ['OBS_60_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE' , 'DEF_60_CNT_SOCIAL_CIRCLE',
      'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE','AMT_INCOME_TOTAL_RANGE','NAME_TYPE_SUITE','AMT_CREDIT_RANGE'
            ,'EMERGENCYSTATE_MODE','TOTALAREA_MODE']

for x in col_mixt:
    df.loc[:, x] = df[x].replace(to_replace = np.NaN, value='0.0')


# In[135]:


df_miss = find_missing_values(df)
print(df_miss)


# In[137]:


msno.bar(df, label_rotation=90, figsize=(20,10), color='blue')


# In[ ]:




