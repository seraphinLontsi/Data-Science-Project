#!/usr/bin/env python
# coding: utf-8

# # Data Description
Data set -->This is a countrywide car accident dataset, which covers 49 states of the USA. Information. The accident data are collected from February 2016 to December 2021.
Currently, there are about 2.9 million accident records in this dataset.
# # Dataset abbrevations:¶
1-ID - This is a unique identifier of the accident record.

2-Severity - Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay).

3-Start_Time - Shows start time of the accident in local time zone.

4-End_Time - Shows end time of the accident in local time zone. End time here refers to when the impact of accident on traffic flow

5-Start_Lat - Shows latitude in GPS coordinate of the start point.

6-Start_Lng - Shows longitude in GPS coordinate of the start point.

7-End_Lat - Shows latitude in GPS coordinate of the end point.

8-End_Lng - Shows longitude in GPS coordinate of the end point.

9-Distance(mi) - The length of the road extent affected by the accident.

10-Description - Shows natural language description of the accident.

11-Number - Shows the street number in address record.

12-Street - Shows the street name in address record.

13-Side - Shows the relative side of the street (Right/Left) in address record.

14-City - Shows the city in address record.

15-County - Shows the county in address record.

16-State - Shows the state in address record.

17-Zipcode - Shows the zipcode in address record.

18-Country - Shows the country in address record.

19-Timezone - Shows timezone based on the location of the accident (eastern, central, etc.).

20-Airport_Code - Denotes an airport-based weather station which is the closest one to location of the accident.

21-Weather_Timestamp - Shows the time-stamp of weather observation record (in local time).

22-Temperature(F) - Shows the temperature (in Fahrenheit).

23-Wind_Chill(F) - Shows the wind chill (in Fahrenheit).

24-Humidity(%) - Shows the humidity (in percentage).

25-Pressure(in) - Shows the air pressure (in inches).

26-Visibility(mi) - Shows visibility (in miles).

27-Wind_Direction - Shows wind direction.

28-Wind_Speed(mph) - Shows wind speed (in miles per hour).

29-Precipitation(in) - Shows precipitation amount in inches, if there is any.

30-Weather_Condition - Shows the weather condition (rain, snow, thunderstorm, fog, etc.)

31-Amenity - A POI annotation which indicates presence of amenity in a nearby location.

32-Bump - A POI annotation which indicates presence of speed bump or hump in a nearby location.

33-Crossing - A POI annotation which indicates presence of crossing in a nearby location.

34-Give_Way - A POI annotation which indicates presence of give_way in a nearby location.

35-Junction - A POI annotation which indicates presence of junction in a nearby location.

36-No_Exit - A POI annotation which indicates presence of junction in a nearby location.

37-Railway - A POI annotation which indicates presence of railway in a nearby location.

38-Roundabout - A POI annotation which indicates presence of roundabout in a nearby location.

39-Station - A POI annotation which indicates presence of station in a nearby location.

40-Stop - A POI annotation which indicates presence of stop in a nearby location.

41-Traffic_Calming - A POI annotation which indicates presence of traffic_calming in a nearby location.

42-Traffic_Signal - A POI annotation which indicates presence of traffic_signal in a nearby location.

43-Turning_Loop - A POI annotation which indicates presence of turning_loop in a nearby location.

44-Sunrise_Sunset - Shows the period of day (i.e. day or night) based on sunrise/sunset.

45-Civil_Twilight - Shows the period of day (i.e. day or night) based on civil twilight.

46-Nautical_Twilight - Shows the period of day (i.e. day or night) based on nautical twilight.

47-Astronomical_Twilight - Shows the period of day (i.e. day or night) based on astronomical twiligh
# ## Import Basic Libaries

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt


# ## Import oder Libaries

# In[2]:


import matplotlib.patches as mpatches
import matplotlib
import calendar
import gc


# In[3]:


df_raw = pd.read_csv('US_Accidents_Dec20_Updated.csv')


# In[4]:


df = df_raw.copy()


# In[5]:


df.head(5)


# In[6]:


df.tail(5)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


print(f"Number of Row : {df.shape[0]} and Numer of Columns: {df.shape[1]}")


# In[10]:


df = df.drop_duplicates()


# In[11]:


print(f"Number of Row : {df.shape[0]} and Numer of Columns: {df.shape[1]}")


# ## Missing Values

# In[12]:


def find_missing_values(df):
    df_nan = ((df.isnull().sum()/len(df))*100).sort_values()
    df_miss = df_nan[df_nan >0]
    return df_miss


# In[13]:


df_miss = find_missing_values(df)
print(df_miss)


# In[14]:


def plot_missing_values(df):
# Function to count the missing values in df
    plt.figure(figsize=(10,5), dpi=100)
    df_miss.plot(kind='bar')
    plt.title('Percentage Missing Values', fontsize=20)
    plt.xlabel('Columns Name' ,fontsize=14)
    plt.ylabel('Percentage of Missing Values',fontsize=14)


# In[15]:


plot_missing_values(df_miss)


# In[16]:


msno.bar(df, color="dodgerblue", fontsize=12, figsize=(25,20))


# In[17]:


# Columns Name
df.columns


# In[18]:


# Plot the Missing values 
null_df = pd.DataFrame((df.isnull().sum()/len(df)*100)).reset_index()
null_df.columns = ['Columns Name', 'Null Values Percentage']
fig = plt.figure(figsize=(22,8))
ax = sns.pointplot(x='Columns Name', y = 'Null Values Percentage', data=null_df, color='blue')
plt.xticks(rotation=90, fontsize=12)
ax.axhline(50, ls='--', color='red')
plt.title("Percentage of Missind Values")
plt.ylabel("Null Values Percentage")
plt.xlabel("Columns Name")
plt.show()


# ## EDA 

# ### 1. City in US with the most number of accident cases 

# In[19]:


# The 10 cities with the highest number of accidents
cities = df['City'].value_counts()
plt.figure(figsize=(12,7))
plt.title("The 10 cities with the highest number of accidents",size=22,color="grey",y=1.05)
lab = cities[:10].index
plt.pie(cities[:10] ,shadow=True ,explode=(0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1,0.1) ,startangle=90 ,labels=lab)
plt.show()


# In[20]:


plt.figure(figsize=(25,7), dpi=100)
sns.countplot(x='State', data=df)
plt.yscale("log")
plt.title("STATE WITH NUMBER OF ACCICENDS", fontsize=20)
plt.show()


# In[21]:


### Top 10 Street with maximun numer of accidents

streets_top_10  = df['Street'].value_counts().sort_values(ascending=False)[:10].reset_index()
streets_top_10.columns = ["Street Name", "Number of Accidents"]

fig = plt.figure(figsize=(12,6), dpi=100)
sns.barplot(x=streets_top_10['Street Name'], y=streets_top_10['Number of Accidents'])
plt.xticks(rotation=90)
plt.xlabel('Street Name', fontsize=15)
plt.ylabel('Number of Accidents', fontsize=15)
plt.title("Top 10 Street With Maximun of Accidents", fontsize=20)
plt.show()


# In[22]:


street = df['Street'].value_counts()
plt.figure(figsize=(12,7))
plt.title("Top 10 Street With Maximun of Accidents" ,size = 22 ,color="grey" ,y=1.05)
lab = street[:10].index
plt.pie(cities[:10] ,shadow=True ,explode=(0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1,0.1) ,startangle=90 ,labels=lab)
plt.show()


# In[23]:


for x in df['Severity'].unique():
    plt.subplots(figsize=(15,5))
    severity = df[df['Severity']==x]['Weather_Condition'].value_counts().sort_values(ascending=False)[:10]
    severity.plot(kind='bar', align='center')
    plt.xlabel("Weather Condition", fontsize=17)
    plt.ylabel("Accident Count",fontsize=17)
    plt.title('10 of The Main Weather Conditions for Accidents of Severity ' + str(x),fontsize=17)


# In[24]:


import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame


# In[25]:


'''
geometry = [Point(x) for x in zip(df['Start_Lng'], df['Start_Lat'])]
gdf = GeoDataFrame(df, geometry=geometry)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(13,7)), marker='o', color='red', markersize=15)
plt.show()
'''


# In[26]:


'''
plt.figure(figsize=(20,10), dpi=200)
sns.scatterplot(x=df['Start_Lng'], y=df['Start_Lat'], size=0.001, hue='Severity', data=df, alpha=0.5)
plt.xlabel("Start Longitude", fontsize=17)
plt.ylabel("Start Latitude", fontsize=17)
plt.title("US Accidents Start Latitude and Longitude", fontsize=22)
plt.show()
'''


# In[27]:


df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'])
df['End_Time'] = pd.to_datetime(df['End_Time'])


# ### What are the months, weeks, days and hours when accidents are most frequent

# In[28]:


# Just the beginning of the Accident will be considered

df['Start_Time']       = pd.to_datetime(df['Start_Time'])
df['Start_Time_Year']  = df['Start_Time'].dt.year
df['Start_Time_Month'] = df['Start_Time'].dt.month
df['Start_Time_Week']  = df['Start_Time'].dt.isocalendar().week
df['Start_Time_Day']   = df['Start_Time'].dt.day
df['Start_Time_Hour']  = df['Start_Time'].dt.hour


# In[29]:


df_year_severity = df.groupby(['Start_Time_Year']).count()['Severity']


# In[30]:


df.groupby(['Start_Time_Month', 'Severity']).count()


# In[31]:


df_Severity_month=df.groupby(['Start_Time_Year', 'Start_Time_Month', 'Severity']).count()['ID']


# In[32]:


df_Severity_month = df_Severity_month.reset_index()


# In[33]:


df_Severity_month[df_Severity_month['Start_Time_Year']==2016]


# In[34]:


plt.figure(figsize=(12,5), dpi=100)
sns.barplot(x='Start_Time_Month', y='ID', hue='Severity', data=df_Severity_month, ci=None)
plt.ylabel('Accidents Count', size=16)
plt.xlabel('Month', size=16)
plt.show()


# In[35]:


'''
for year in df_Severity_month['Start_Time_Year'].unique():
    df_month = df_Severity_month[df_Severity_month['Start_Time_Year']==year]
    plt.figure(figsize=(10,4), dpi=100)
    sns.barplot(x='Start_Time_Month', y='ID', hue='Severity', data=df_month, ci=None)
    plt.ylabel('Accidents Count', size=16)
    plt.xlabel('Month of Year '+ str(year), size=16)
    plt.show()
    '''


# In[36]:


sns.set_style("whitegrid")


# In[37]:


df['Start_Time_DayOfWeek'] = df['Start_Time'].apply(lambda time : time.dayofweek)


# In[38]:


df_DayOfWeek = df.groupby('Start_Time_DayOfWeek').count()['ID']


# In[39]:


plt.figure(figsize=(10,5), dpi=100)
sns.countplot(x=df['Start_Time_DayOfWeek'])
plt.xlabel('Day of the Week', size=16)


# ### Hours Analysis on Sunday

# In[40]:


Sunday_start_time = df['Start_Time'][df['Start_Time_DayOfWeek']==6]
plt.figure(figsize=(10,5), dpi=100)
sns.countplot(data=df, x=Sunday_start_time.dt.hour)
plt.xlabel('Sunday Start Time', size=16)


# In[41]:


Saturday_start_time = df['Start_Time'][df['Start_Time_DayOfWeek']==5]
plt.figure(figsize=(10,5), dpi=100)
sns.countplot(data=df, x=Saturday_start_time.dt.hour)
plt.xlabel('Saturday Start Time', size=16)


# In[42]:


df_hours = df['Start_Time_Hour'].value_counts().reset_index().rename(columns={'index':'Hours', 'Start_Time_Hour': 'Cases'}).sort_values('Hours')
df_hours


# ### Hours Analysis

# In[43]:


plt.figure(figsize=(10,5), dpi=100)
sns.countplot(data=df, x=df['Start_Time_Hour'])
plt.xlabel('Start Time in Hours', size=16)


# In[44]:


# Plots the histogram for each numerical feature in a separate subplot
df.hist(bins=24, figsize=(15, 12), layout=(-1, 5), edgecolor="black")
plt.tight_layout();


# ## Advance Analysis

# ### Year Analysis

# In[45]:


df_year = pd.DataFrame(df['Start_Time_Year'].value_counts()).reset_index().rename(columns={'index':'Year', 'Start_Time_Year':'Cases'}).sort_values(by='Cases', ascending=True)


# In[46]:


df_year


# In[47]:


fig, ax = plt.subplots(figsize=(10,5), dpi=100)

ax = sns.barplot(y=df_year['Cases'] ,x=df_year['Year'], palette=['#9a90e9', '#5d52de', '#3ee6e0', '#40ff59','#2ee83e'])

for idx in ax.patches:
    ax.text(idx.get_x() + 0.2 , idx.get_height() - 50000, str(round((idx.get_height()/df.shape[0])*100, 2)) + '%'
            , fontsize=15, weight='bold', color='white')

#plt.ylim(10000,1000000)
plt.title('\n Road Accident Percentage \n over past 5 years in US (2016,2020)\n', size=20, color='grey')
plt.ylabel('\nAccident Cases\n', fontsize=15, color='grey')
plt.xlabel('\nYears\n', fontsize=15, color='grey')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)

for i in ['bottom','top','left','right']:
    ax.spines[i].set_color('white')
    ax.spines[i].set_linewidth(1.2)
    
for k in ['top', 'right', 'bottom', 'left']:
    side = ax.spines[k]
    side.set_visible(False)

ax.set_axisbelow(True)
ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=0.3)
MA = mpatches.Patch(color='#2ee88e', label='Year with Maximum\n no. Accidents on Road')
MI = mpatches.Patch(color='#9a90e8', label='Year with Minimun\n no. Accidetns on Road')
ax.legend(handles=[MA, MI], prop={'size': 9.5}, loc='best', borderpad=1, 
          labelcolor=['#2ee88e', '#9a90e8'], edgecolor='white');
plt.show()


# ### Month Analysis

# In[48]:


df_month = pd.DataFrame(df['Start_Time_Month'].value_counts()).reset_index().rename(columns={'index':'Month',
                                                                                           'Start_Time_Month':'Cases'}).sort_values(by='Month')


# In[49]:


df_month['Month'] = list(calendar.month_name)[1:]


# In[50]:


df_month


# In[51]:


fig, ax = plt.subplots(figsize=(13,7), dpi=100)

cmap = matplotlib.cm.get_cmap('plasma', 12)
col = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

ax = sns.barplot(x=df_month['Cases'] ,y=df_month['Month'] ,palette='plasma')

for s in ax.patches:
    plt.text(s.get_width()-10000, s.get_y()+0.4,
            '{:.2f}%'.format((s.get_width()*100/df.shape[0])), ha='center', va='center'
             , fontsize=12, color='white', weight='bold')

plt.title(f'\n Road Accident Percentage for different Month\n', size=17, color='grey')
plt.ylabel('\nMonth of Year\n', fontsize=15, color='grey')
plt.xlabel(f'\nAccident Cases\n', fontsize=15, color='grey')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.xlim(0,450000)

for i in ['top', 'left', 'right']:
    side = ax.spines[i]
    side.set_visible(False)

ax.set_axisbelow(True)
ax.spines['bottom'].set_bounds(0,450000)
ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.5)
    
MA = mpatches.Patch(color=col[-1], label='Month with Maximum\n no. Accidents on Road')
MI = mpatches.Patch(color=col[0], label='Month with Minimun\n no. Accidetns on Road')
    
ax.legend(handles=[MA, MI], prop={'size': 10}, loc='best', borderpad=1, 
              labelcolor=[col[0], 'grey'], edgecolor='white');
plt.show()


# In[52]:


gc.collect()


# # Data Cleaning and Feature Engineering

# In[53]:


num_cols = [col for col in df.columns if df[col].dtypes in ['int64','float64']]
cat_cols = [col for col in df.columns if df[col].dtypes in ['object','bool']]
date_cols = [col for col in df.columns if df[col].dtypes=='datetime64[ns]']


# In[54]:


date_cols


# In[55]:


cat_cols


# In[56]:


num_cols


# In[57]:


find_missing_values(df)


# ### Feature Relationships

# In[58]:


df_corr = df.corr()
df_corr = df_corr.drop('Turning_Loop', axis=1)
df_corr = df_corr.drop('Turning_Loop', axis=0)


# In[59]:


plt.figure(figsize=(22,15), dpi=200)
sns.heatmap(round(df_corr,2), cmap='viridis', annot=True)
plt.show()


# In[60]:


#temperature and wind chill have a 0.99 correlation

plt.figure(figsize=(9,5),dpi=100)
sns.scatterplot(x='Wind_Chill(F)', y='Temperature(F)', data=df)
plt.show()


# In[61]:


# Drop the columns City with just 108 missing values missing and the columns Number with more than 65% miss values
df = df.drop(['City', 'Number'], axis=1)


# In[62]:


find_missing_values(df)


# In[63]:


df[df[['Sunrise_Sunset', 'Astronomical_Twilight', 'Nautical_Twilight',
       'Civil_Twilight']].isnull()]


# In[64]:


df[df[['Sunrise_Sunset', 'Astronomical_Twilight', 'Nautical_Twilight',
       'Civil_Twilight']].isnull()][['Sunrise_Sunset', 'Astronomical_Twilight', 'Nautical_Twilight',
       'Civil_Twilight']]


# In[65]:


df = df.dropna(subset=['Sunrise_Sunset', 'Astronomical_Twilight', 'Nautical_Twilight',
       'Civil_Twilight'])


# In[66]:


df.corr()['Temperature(F)'].sort_values()


# In[67]:


df = df.drop('Turning_Loop', axis=1)


# In[68]:


# Make all Zipcode 5 digit
zipcodes = pd.DataFrame(df[[not a for a in df['Zipcode'].isna()]]['Zipcode'].str[:5])
zipindex = np.array(df[df['Zipcode'].notnull()].index)
df.loc[zipindex, 'Zipcode'] = zipcodes.loc[:, 'Zipcode']


# In[69]:


df['Zipcode']


# In[70]:


df['Side'].value_counts()


# In[71]:


# Replace space with most commont entry
df['Side'] = df['Side'].replace(' ', 'R')


# In[72]:


df['Weather_Condition'].replace('Thunder','Thunderstorm')


# In[73]:


#view unique values of Wind_Direction
df['Wind_Direction'].unique()


# In[74]:


#replace entries in Wind_Direction
df['Wind_Direction'] = df['Wind_Direction'].replace('North','N')
df['Wind_Direction'] = df['Wind_Direction'].replace('Variable','VAR')
df['Wind_Direction'] = df['Wind_Direction'].replace('West','W')
df['Wind_Direction'] = df['Wind_Direction'].replace('East','E')
df['Wind_Direction'] = df['Wind_Direction'].replace('South','S')
df['Wind_Direction'] = df['Wind_Direction'].replace('Calm','CALM')


# In[75]:


df['Wind_Direction'].unique()


# In[76]:


zipcode_Lat_Lng = df.loc[np.array(df[df['Zipcode'].isnull()].index), ['Start_Lat', 'Start_Lng']]
print(zipcode_Lat_Lng)


# In[77]:


timezome_Lat_Lng = df.loc[np.array(df[df['Timezone'].isnull()].index), ['Start_Lat', 'Start_Lng']]
print(timezome_Lat_Lng)


# In[78]:


#initialize Nominatim for finding missing cities and zipcodes based on lat/lng

from geopy.geocoders import Nominatim
geo = Nominatim(user_agent="geoapiExercises")


# In[80]:


# fill missing values in columns Zipcode

for idx in np.array(df[df['Zipcode'].isnull()].index):
    
    location = geo.reverse(zipcode_Lat_Lng.loc[idx,'Start_Lat'].astype('str')+","+ zipcode_Lat_Lng.loc[idx, 'Start_Lng'].astype('str'))
    
    address = location.raw['address']
    zipcode = address.get('postcode')
    df.loc[idx, 'Zipcode'] = zipcode
    
    if zipcode==None:
        df.loc[idx, 'Zipcode'] = df[df['State']==df.loc[idx, 'State']]['Zipcode'].mode(dropna=True)[0]


# In[ ]:


df['Zipcode'].isnull().sum()


# In[ ]:


#

from timezonefinder import TimezoneFinder
instance = TimezoneFinder()

for idx in np.array(df[df['Timezone'].isnull()].index):
    timezone = instance.timezone_at(lng=timezome_Lat_Lng.loc[idx, 'Start_Lng'], lat=timezome_Lat_Lng.loc[idx, 'Start_Lat'])
    df.loc[idx, 'Timezone'] = timezone


# In[ ]:


# 
ac_MissIndex = np.array(df[df['Airport_Code'].isnull()].index)
airpot_code  = df.loc[ac_MissIndex, 'State']

print(airpot_code)


# In[ ]:


#replace missing airport_code data with most common airport_code for each state
ac_states = pd.DataFrame(df.loc[ac_MissIndex, 'State'].unique(), columns=['State'])

for i in range(len(ac_states)):
    ac_states.loc[i, 'Mode'] = df[df['State'] == ac_states.loc[i ,'State']]['Airport_Code'].mode()[0]
    
for i in ac_MissIndex:
    df.loc[i, 'Airport_Code'] = ac_states[ac_states['State'] == df.loc[i ,'State']]['Mode'].tolist()[0] 


# In[ ]:


df['Airport_Code'].isnull().sum()


# In[ ]:


# Copy values values from Start to End
latnull = np.array(df[df['End_Lat'].isnull()].index)
df.loc[latnull, 'End_Lat'] = df.loc[latnull, 'Start_Lat']

lngnull = np.array(df[df['End_Lng'].isnull()].index)
df.loc[lngnull, 'End_Lng'] = df.loc[lngnull, 'Start_Lng']

# Replace missing wind_chill data with temperature
wcnull = np.array(df[df['Wind_Chill(F)'].isnull()].index)
df.loc[wcnull, 'Wind_Chill(F)'] = df.loc[wcnull, 'Temperature(F)']

# Replace missing temperature data with wind chill
tempnull = np.array(df[df['Temperature(F)'].isnull()].index)
df.loc[tempnull, 'Temperature(F)'] = df.loc[tempnull, 'Wind_Chill(F)']

# Replace missing data for when
weather_time_null = np.array(df[df['Weather_Timestamp'].isnull()].index)
df.loc[weather_time_null, 'Weather_Timestamp'] = df.loc[weather_time_null, 'End_Time']


# In[ ]:


find_missing_values(df)


# In[ ]:


# Mode Imputer
df.loc[np.array(df[df['Wind_Direction'].isnull()].index), 'Wind_Direction'] = 'CALM'

df.loc[np.array(df[df['Weather_Condition'].isnull()].index), 'Weather_Condition'] = 'Fair'


# In[ ]:


find_missing_values(df).index


# In[ ]:


df[['Pressure(in)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
       'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']].corr()


# In[ ]:


imputer_median = ['Pressure(in)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
       'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']


# In[ ]:


# Impute the rest of missing values with the median values

from sklearn.impute import SimpleImputer


# In[ ]:


imputer = SimpleImputer(missing_values=np.nan ,strategy='median')
df_median_fit = imputer.fit_transform(df[imputer_median])
df_median = imputer.transform(df_median_fit)
df[imputer_median] = pd.DataFrame(df_median)


# In[ ]:


find_missing_values(df)


# In[ ]:


df = df.dropna(subset=imputer_median, axis=0)


# In[ ]:


# print number and percentage of null entries per variable
find_missing_values(df)


# In[ ]:





# In[ ]:




