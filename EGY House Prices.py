#!/usr/bin/env python
# coding: utf-8

# # 1.0 Preparing Modules and Files

# In[172]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import plotly.express as px


# In[2]:


adf = pd.read_csv(r"C:\Users\Shehab Fekry\Downloads\Egypt_Houses_Price.csv")


# ## Data Dictionary
# 
#  - Type: style of archticture.
#      - 'Duplex', A duplex house plan has two living units attached to each other, either next to each other as townhouses, or above     each other like apartments. one of two properties on a single lot (land).
#      - 'Apartment', a self-contained housing unit that occupies part of a building.
#      - 'Penthouse', a flat (property) on the top floor of a tall building.
#      - 'Studio', non-residential properties of very small areas.
#      - 'Chalet', beach houses (less often pool houses and apartments) built in any style of architecture, (apartments, villas, etc.) usually of small areas.
#      - 'Standalone Villa', large, single-family homes that are detached from other buildings.
#      - 'Twin House', two living units attached to each other side by side, usually are mirror images of each other. despite sharing a wall; each of the properties is built upon its own lot (land).
#      - 'Town House', a unit in a row of joined houses sharing side walls.
# 
# 
# - Price: Price of the property.
# - Area: Area of the property.
# - Furnished: wether the residence has furniture.
# - Payment_Option: Payment method, Cash / Installment.
# - Delivery_Date: Years / months till the property is delivered.
# - Delivery_Term: interior state of the property.

# # 2.0 Exploratory Data Analysis

# In[3]:


adf.describe()


# In[4]:


adf.info()


# ## Insights:
# 
# - The dataset has 27361 entries (rows), and 12 columns.
# - There are dozens of null values within at least 4 columns.
# - All the 12 columns are in 'Object' format, although attributes like 'Price' and 'Area' should be numeric.
# - Columns 'Level' and 'Compound' have no null values, but missing values are represented in strings of 'Unknown', and likely with all columns.
# - Data cleaning should involve dropping 'null' values along with the critical 'Unknown' values in numeric columns.

# # 3.0 Data Cleaning

# ## 3.1 Dropping 'null' Values

# In[5]:


adf.isnull().sum() 
# cells designated 'null' do not express all the missing values,
# bcuz all columns are in 'object' dtype and some missing values are represented as strings e.g. 'unknown', 'missing'


# In[6]:


# A copy for backup
df = adf.copy()


# In[7]:


df = df.dropna()


# In[8]:


df.isnull().sum()


# ## 3.2 Cleaning inconvertible 'object' rows (Strings) in numeric columns
# 
# - do **Price** and **Area** columns have 'Unknown' rows or any similar text for missing values?

# In[9]:


# dtype assignment

df[['Price', 'Area']] = df[['Price', 'Area']].apply(pd.to_numeric, errors='coerce')

# errors='coerce' converts any non numeric Price / Area cell to NaN values


# In[10]:


df.loc[df['Price'].isnull()].head(10)


# In[11]:


df.isnull().sum()


# In[12]:


df = df.dropna()


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# ## 3.3 Checking the Integrity and Accuracy of Non-Numeric Columns

# In[15]:


df.Furnished.unique()


# In[16]:


df.Level.unique()


# In[17]:


com1 = df.Compound.unique() # >> numpy array
sorted(com1)

# sort_values(): Specifically for pandas DataFrames (and Series).
# np.sort(): Specifically for numpy arrays.
# sorted(L): General-purpose, for any iterable (e.g., lists, tuples), Python built-in function.


# In[18]:


com01 = df['Compound'].unique()
np.sort(com01)


# In[19]:


df.Type.unique()


# In[20]:


cit = df.City.unique()
sorted(cit)


# In[21]:


df.Bedrooms.unique()


# In[22]:


df.Bathrooms.unique()


# ## Insights:
# 
# - Column **Type** shows redundancy in naming categories, which causes statistical inaccuracies and misrepresentations.
# - The columns **Bedrooms** and **Bathrooms** show the same problem, but will need a different approach as both are numerical and many mistakes are seen compared to **Type.**
# - Many columns have abundant 'Unknown' values, but since they're categorical and insignificant they will not be processed.
# - Columns **Compound** and **City** had to be checked manually after being cast in an alphabetical arrangement, and both showed no repeated categories.

# ## 3.4 Fixing Misspellings and Typos in Non-Numeric Columns

# ### 3.4.1 Fixing redundancy in **Type**
# 
# - Conditional Assignment using df.loc[]

# In[23]:


df.loc[df['Type']=='Twin house', 'Type'] = 'Twin House'
df.loc[df['Type']=='Stand Alone Villa', 'Type'] = 'Standalone Villa'


# ### 3.4.2 Fixing redundancy in **Bedrooms** and **Bathrooms**
# 
# - converting ['Bedrooms', 'Bathrooms'] to numeric columns would fix the redundancy.
# - but also would result in 'null's in place of '10+' rows, the only category that cannot be represented numerically.
# - which then can be replaced with '10+' using conditional assignment.

# In[24]:


bath = df.loc[df['Bathrooms']=='10+']
bed = df.loc[df['Bedrooms']=='10+']
display(bath)
display(bed)

# it appears there is only one record (residence) where Bedrooms are 10+, same for Bathrooms, and it is actually the same record.


# In[25]:


df[['Bedrooms', 'Bathrooms']] = df[['Bedrooms', 'Bathrooms']].apply(pd.to_numeric, errors='coerce')


# In[26]:


df['Bedrooms'] = df['Bedrooms'].astype(str)
df['Bathrooms'] = df['Bathrooms'].astype(str)


# In[27]:


df.loc[df['Bedrooms']=='nan', 'Bedrooms'] = '10+'
df.loc[df['Bathrooms']=='nan', 'Bathrooms'] = '10+'


# In[28]:


df.Bedrooms.unique()


# In[29]:


df.Bathrooms.unique()


# # 4.0 Invistigating Duplications

# ## 4.1 Grouping duplicates and sorting by occurrences (Counts)

# In[30]:


# Creating a dataframe consisting of the duplicates
dup = df.loc[df.duplicated(keep=False)]

# Grouping by unique records and sizing each group
ss = dup.groupby(list(dup.columns)).size().reset_index(name='Count')

# sorting 
freq = ss.sort_values(by='Count', ascending=False)
freq.head(11)


# keep=False implies that original records are included.


# ## 4.2 Duplicates' Stats

# In[31]:


sum = freq['Count'].sum()
cnt = freq['Count'].count()
print(f'Groups showing duplication: {cnt},\nSum: {sum} records')


# In[32]:


# 'Count' statistics

freq.Count.describe()


# In[33]:


fig, ax = plt.subplots(figsize=(12,3))
#plt.figure()
sns.boxplot(data=freq, x='Count', ax=ax)
ax.set_xticks(np.arange(1, 32, 1))
plt.title('Statistial representation of the Recurrences')
plt.show()


# In[34]:


sns.countplot(data=freq, x='Count') #Sheikh Zayed
plt.xlabel('Recurrences')
plt.ylabel('Groups / original records \nout of 1190')
plt.title('Distribution of Recurrences')
plt.show()


# In[35]:


plt.figure(figsize=(12,4))
cop = freq['Compound'].value_counts()[:20]
plt.bar(cop.index, cop.values)
plt.xticks(rotation=90)
plt.show()


# ## Insights:
# 
# - There are 3074 records showing duplication, 1190 of which are originals.
# - Minimum duplication records show recurrence of 2 times (original included), also they comprise about 70% of the total originals.
# - Maximum duplication records show recurrence of 31 and 15 times, with a single instance for each number.
# - Statistical analysis using boxplot demonstrates the maximum allowed recurrence before it is considered an outlier to be **4 recurrences.**
# - As the total recurrences of a record increases, it is more likely that the duplications are not authentic, and merely the result of data entry errors.
# 
# #### How should the duplicates be processed?
# 
# - Considering the nature of the subject dataset (real estate data), it's possible for different properties to have identical features and prices, especially in large apartment complexes i.e. Compounds and other residential areas.
# - However, the boxplot analysis demonstrated the likelyhood of any recurrence bigger than 4 to be erroneous.
# - So, the best solution to this problem is keeping all the duplicates having less than or equal to 4 recurrences. And cutting down all the records of recurrences bigger than 4 (outliers) to that same number. e.g. 31 records are sized down to 4.

# ## 4.3 Processing Duplicates

# In[36]:


# Adding a cumulative count column to mark unwanted duplicates
df['cumulative'] = df.groupby(list(df.columns)).cumcount()
df.sort_values(by='cumulative', ascending=False).head(5)


# In[37]:


# How many records are expected to be dropped?
excess_records = df.loc[df['cumulative']>3, 'cumulative'].count() # cumulative count starts from index [0]
print(excess_records)


# In[38]:


# Filtering the DataFrame to keep only the desired number of occurrences and dropping the temporary column
df = df[df['cumulative'] < 4]
df = df.drop(columns='cumulative')
df


# ## 4.4 Visual representation of the new duplication subset

# In[39]:


# Creating a dataframe consisting of the new modified duplicates.
neo_dup = df.loc[df.duplicated(keep=False)]

# Grouping by unique records and sizing each group
sz = neo_dup.groupby(list(neo_dup.columns)).size().reset_index(name='Count')

# sorting 
frequency = sz.sort_values(by='Count', ascending=False)

# keep=False implies that original records are included.


# In[40]:


sns.countplot(data=frequency, x='Count')
plt.xlabel('Recurrences')
plt.ylabel('Groups / original records \nout of 1190')
plt.title('Distribution of Recurrences')
plt.show()


# In[41]:


frequency.Count.describe()


# # 5.0 Checking for other Inconsistencies
# 
# - Categorical Inconsistencies might emerge between correlated columns like **Compound** and **City**, for instance, some Compounds are known to be within specific regions, yet associated with others.
# - A logical inconsistency emerges as, for instance, a misfit presents in the intuitive relationship between two attributes. like the **Area** of the property and how many compartments might fit inside of it e.g. **Bedrooms** and **Bathrooms.**
# - Spotting inconsistencies implies making manual observations onto a sample subset of the data.

# ### 5.0.1 Choosing a sample dataset
# 
# - Sample dataset where 'Type'=='Studio'

# In[42]:


df.loc[df['Type']=='Studio'].sort_values(by='Price', ascending=False).head()


# > [Note!]
# > it's well known that Compound Marassi is in the North Coast, yet some records above inaccurately associate it with different cities.

# In[43]:


# NNC = Non North Coast

# all records where Compound == Marassi
all_mar = df.loc[(df['Compound']=='Marassi')]

# all cities associated with Marassi
mar_city = all_mar.City.value_counts()

print(f'Marassi associated Cities: \n\n{mar_city}')


# In[181]:


p = round((17+9+3)*100/335, 2)
print(f'Percentage of mismatching records between "City" and "Compound" in Marassi subframe: {p} %')


# ## Q. Do other compound-subsets show the same partial misfits? if so, Which column is showing the correct values? 
# 
# - Theory: Columns 'Compound' and 'City' partially show logical mismatches for compound-filtered subsets (8.7 % in data above), which one is incorrect?
# - Statistical approach is needed

# In[54]:


# Top 10 Compounds by frequency

top10cpd = df.Compound.value_counts().reset_index(name='Size').head(10)
top10cpd


# In[ ]:


# test to determine the accuracy of associations between columns 'Compound' and 'City'

def cpd_associated_cities(compound):
    comp_frame = df[df['Compound']==f'{compound}']
    cit_size = comp_frame.City.value_counts()
    return cit_size


# In[57]:


cpd_associated_cities('Madinaty')


# In[ ]:


cpd_associated_cities('Mountain View North Coast')


# In[ ]:


cpd_associated_cities('Hyde Park New Cairo')


# In[ ]:


cpd_associated_cities('Rehab City')


# In[185]:


cpd_associated_cities('Maadi V')


# ## 5.1 Investigating Column-to-Column compatibility

# ### 5.1.1 Is 'Price' more consistent with 'Compound' or 'City' in filtered subsets?
# 
# - Proceeding with Compound == Marassi subset.
# - Fixing the factor 'Compound' to measure consistency between 'Price' and 'City'.

# In[45]:


# Price comparison - pivot table
pivot00 = pd.pivot_table(all_mar, index=['Type', 'City'],  columns='Furnished', values=['Price'], aggfunc=['mean', 'count'])


def highlight_cells(val):
     # highlighting single porperties - no true mean
    if val == 1:
        color = 'wheat'
    # mean prices
    elif val > 1.1*10**7 and val < 2*10**7:     # 11 mil to 20 mill
        color = 'yellow'
    elif val > 2*10**7 and val < 3.5*10**7:     # 20 mil to 35 mil
        color = 'lime'
    elif val >= 5*10**6 and val < 9*10**6:     # 5 mil to 9 mil
        color = 'cyan'
    elif val > 3*10**6 and val < 5*10**6:     # 3 mil to 5 mil
        color = 'salmon'
    else:
        color = ''
    return f'background-color: {color}'

styled_pivot = pivot00.style.applymap(highlight_cells)

display(styled_pivot)
print('DATA: All Marassi Records (same compound)')


# In[46]:


# Average Price for each Type of property _ whole data

display(Markdown('### Average price for each Type of property _ whole data'))
for arc_type in df.Type.unique():
    tyframe = df.loc[df['Type']== arc_type]
    mean_price = tyframe.Price.mean()
    print(f'Type: {arc_type}\n Mean: {mean_price}\n')


# In[47]:


# demonstrating that varying areas do account for the spread-out prices

mar_pent = df.loc[(df['Type']=='Penthouse') & (df['Compound']=='Marassi'), ['Price', 'Area', 'City', 'Furnished']].sort_values(by='Area', ascending=False)
display(Markdown('### All marassi penthouses'))
display(mar_pent)


# In[48]:


# All Apartments of the North coast
north_apt = df.loc[(df['City']=='North Coast') & (df['Type']=='Apartment')]
north_apt_cpd = north_apt.Compound.value_counts()
north_apt_cpd


# ## Insights:
# - All Apartments of the North coast are very few (27 out of 4000+ records) and are associated with compounds that are not in the North Coast (or any coast), this consistency that combines **Type** and **Compound** and excludes **City** out suggests that column **City** is incorrect.
# 
# - All "Marassi" records do not have properties of Type "Apartment", which is reasonable because "Chalet" category encompasses all beach / coastal apartments, and "Marassi" is in the "North Coast".
# 
# - within each type of property, there is a convergence in **Price** between "Marassi" properties of different areas, also there is an increase above whole-data mean prices specific for each type.
#  
# - large differences in prices per type and between different cities are scarse and could be explained by gaps in properties' **Area,** a factor not considered in the pivot table. (Check the table above)
#  
#  ### Conclusion:
#  - Column "Compound" is showing high compatiblity with **Type** and **Price.**
#  - incorrectness should likely be ascribed to the column "City", for being incoherent with **Type** and **Price.**

# ### 5.1.2 Price range comparison between NC and NNC Marassi records when limiting the factor 'Type' to Standalone Villas only

# In[49]:


# all marassi villas
marvill = df.loc[(df['Type']=='Standalone Villa') & (df['Compound']=='Marassi'), ['Price', 'Area', 'City']]

# associated cities
city_vill = marvill.City.value_counts()
city_vill


# In[50]:


# marassi villas _ North Coast
marvill_nc =  marvill.loc[marvill['City']=='North Coast']

# scatter plot _ all marassi villas
sns.scatterplot(data=marvill, x='Area', y='Price', hue='City', alpha=0.6, sizes=90)

# mean price of marassi villas _ North Coast only
meanprice = marvill_nc.Price.mean()

plt.axhline(meanprice, color='crimson', alpha=0.6, label='Average Price')
plt.title('Average Price of Marassi Villas')
plt.show()


# In[51]:


# Price stats of marassi villas _ North Coast
marvill_nc.Price.describe()


# In[52]:


sns.boxplot(data=marvill_nc, x='Price' )
plt.show()


# In[53]:


# all Tagamoa villas
tagavill = df.loc[(df['Type']=='Standalone Villa') & (df['City']=='New Cairo - El Tagamoa'), ['Price', 'Area', 'Compound']]
tagavill['new'] = tagavill['Compound'].apply(lambda x: 'Marassi' if x == 'Marassi' else 'Non-Marassi')

sns.scatterplot(data=tagavill, x='Area', y='Price', hue='new', alpha=0.6, sizes=90)
plt.grid(axis='y')
plt.show()


# In[173]:


px.box(tagavill, x='Price')


# ## Insights:
# 
# - with an exception of a single record (of extremely low area), all prices of NNC Marassi Villas lie within the bulk \ normal range of North Coast Marassi Villas, this observation suggests that the attribute **City** of NNC Marassi villas was not assigned correctly. 
# 
# 
#  ### Conclusion:
#  - Column "Compound" is showing high compatiblity with **Price.**
#  - This suggests that the column that's most likely showing wrong values is **City**

# ### 5.1.3 Is 'Level' more consistent with 'Compound' or 'City' ?
# - Does **Level** has any natural or logical correlations with other columns?
#     1. certain compounds might have opted to construct it's multi-story architecture based on schemes of definite story limits.
#     2. certain types of architecture are associated with specific story sets, e.g. Standalone Villas having ground and 1st levels.

# In[55]:


# All compounds where Level == 10+
lev11 = df.loc[df['Level']=='10+']
sns.countplot(data=lev11, x='Compound')
plt.xticks(rotation=90)
plt.show()


# In[187]:


df.loc[df['Compound']=='Maadi V'].sort_values(by='Area')


# In[193]:


df.loc[df['City']=='Shorouk City'].sort_values(by='Area').tail()


# In[163]:


# Mountain View North Coast
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
montview = df.loc[(df['Compound']=='Mountain View North Coast')]
sns.scatterplot(data=montview, x='Area', y='Price', hue='Type', alpha=0.6, ax=axes[0])
axes[0].set_xlim(0, 300)
axes[0].set_ylim(0, 7000000)
sns.scatterplot(data=montview, x='Area', y='Price', hue='Level', alpha=0.6, ax=axes[1])
axes[1].set_xlim(0, 300)
axes[1].set_ylim(0, 7000000)
plt.tight_layout()
plt.show()

hi_area = montview.Area.value_counts().head(8)
type_presentation = montview.Type.value_counts()

print(f'Frequent compound areas:\n{hi_area}')
print(' ')
print(type_presentation)
print(f'City == all: {montview.City.count()}')

# Not in The North Coast
montco = montview.loc[montview['City']!='North Coast', 'City'].count()
print(f'City != North Coast: {montco}')


# In[170]:


hi_area = montview.loc[montview['Area'].isin([92, 125]), 'Area']
pivot01 = pd.pivot_table(montview, index=['Type', 'Level'],  values=['Price'], aggfunc=['count'])
pivot01


# In[ ]:


df[(df['Type']=='Standalone Villa') & (df['Level'].isin(['3']))]


# In[ ]:


top10cpd


# In[183]:


def compound_levels(compound):
    comframe = df[df['Compound']==f'{compound}']
    lev_cnt = comframe.Level.value_counts()
    return lev_cnt

compound_levels('Marassi')

# use composition charts


# In[ ]:


compound_levels('Madinaty')


# In[184]:


compound_levels('Maadi V')


# In[ ]:


compound_levels('Mountain View North Coast')


# In[ ]:


compound_levels('Rehab City')


# In[ ]:


compound_levels('Amwaj')


# In[ ]:


compound_levels('Mountain View iCity')


# ## Q. what should the correct associations between "Compound" and "City" look like?

# In[ ]:


# Prevalence-Based 1:1 associations between compounds and cities

all_cpd = sorted(df.Compound.unique())
for cpd in all_cpd:
    cframe = df[df['Compound']==cpd]
    city_cnt = cframe.City.value_counts().reset_index(name='Counts')
    hi_city = city_cnt.loc[city_cnt['Counts'].idxmax(), 'index']
    length = 28 - len(cpd)
    spaces = '.' * length
    print(f'The Compound: {cpd} {spaces} City: {hi_city}')
    
# one kattameya


# In[106]:


# 
lo_res = df.Compound.value_counts().reset_index(name='num')
cpd_list = lo_res[lo_res['num']==3]['index'].to_list() # list of compounds that represent 3 or less records

cpd_frame = df.loc[df['Compound'].isin(cpd_list), ['Compound', 'City']].sort_values(by='Compound')
print(cpd_frame[44:])


# In[134]:


type(cpd_frame)


# In[105]:


df[df['Compound'].str.contains('Sokh')]


# ## 5.2 City Re-assignment

# In[ ]:


# City RE-assignment
# Prevalence-Based Top associations between compounds and cities

all_cpd = sorted([x for x in df.Compound.unique() if x!='Unknown'])
for cpd in all_cpd:
    cframe = df[df['Compound']==cpd]
    city_cnt = cframe.City.value_counts().reset_index(name='Counts')
    hi_city = city_cnt.loc[city_cnt['Counts'].idxmax(), 'index']
    df.loc[df['Compound']==cpd, 'City'] = hi_city
    
df


# In[ ]:


df.loc[df['Compound']=='Bianchi'].City.unique()


# In[ ]:


df[df['Compound']=='One Kattameya'].City.value_counts()


# In[ ]:


#df[df['Compound']=='Bianchi']
compound_levels('Bianchi')


# In[ ]:


above = df[(df['City']=='North Coast') & (df['Level'].isin(['5', '6', '7', '9', '8', '10', '10+']))].Compound.value_counts()
above


# In[ ]:


above5 = df[(df['City']=='North Coast') & (df['Level'].isin(['5', '6', '7', '8', '10', '10+']))]
#above5
for cpd in above5.Compound.unique():
    #subframe = above5[above5['Compound']==cpd]
    city = above5.City.value_counts()
    print(cpd)
    print(f'{city}\n ---------------')


# In[ ]:


df[df['Compound']=='The Gate']


# # 6.0 Visualization

# In[ ]:


top = df.groupby(df['City']).size().reset_index(name='Top')
top0 = top.sort_values(by='Top', ascending=False)
top0


# In[ ]:


df['City'].value_counts()


# In[ ]:


plt.figure(figsize=(12,4))
plt.bar(top0['City'][:20], top0['Top'][:20])
plt.xticks(rotation=90)
plt.title('Records per City')
plt.show()


# ## 6.1 Investigating the Dependency of Price on property's Area

# In[ ]:


# Get unique categories
categories = df['Type'].unique()

# Create subplots
fig, axes = plt.subplots(4, 2, figsize=(10, 20))
axes = axes.flatten()

# Iterate through categories and create scatter plots
for ax, cat in zip(axes, categories):       #>>> for each ax in axes, and cat in categories.
    subset = df[df['Type'] == cat]
    ax.scatter(subset['Area'], subset['Price'], alpha=0.4)
    ax.set_title(f'Type: {cat}')
    ax.set_xlabel('Area')
    ax.set_ylabel('Price')

plt.tight_layout()
plt.show()


# In[ ]:


# correlation between Area and Price overall
correlation_matrix = df[['Price', 'Area']].corr('pearson')
display(correlation_matrix)
correlation_coefficient = df['Area'].corr(df['Price'])
print(f'Overall correlation_coefficient: {correlation_coefficient: .2f}')


# In[ ]:


# correlation between Area and Price among each type of architecture
for property_type in df.Type.unique():
    subset = df[df['Type']==property_type]
    cormat = subset[['Price', 'Area']].corr()
    print(f'\nType: {property_type}')
    print(cormat)


# In[ ]:





# In[ ]:


plt.figure(figsize=(10,4))
sns.histplot(df.loc[df['Price']<10000000, 'Price'], bins=40, kde=True, color='skyblue')
plt.title('Price Distribution')
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.histplot(data=df, x='Area', kde=False, bins=30, hue='Type', multiple='stack')
plt.title('Area Distribution')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x=df['Type'].sort_values(), data=df, palette='viridis')
plt.xticks(rotation=44)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x=df['Bedrooms'].sort_values(), data=df, palette='viridis')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x=df['Bathrooms'].sort_values(), data=df, palette='viridis')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x=df['Level'], data=df, palette='viridis')
plt.show()


# In[ ]:


aga.groupby(['Type', 'Compound']).size().unstack(fill_value=0)


# In[ ]:


# Linear Regression
# Sample data
x = marvill['Area']
y = marvill['Price']

# Perform linear regression
slope, intercept = np.polyfit(x, y, 1)

# Plot scatterplot and regression line
plt.scatter(x, y, label='Data points')
plt.plot(x, slope * x + intercept, color='red', label=f'Line: y={slope:.2f}x + {intercept:.2f}')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Scatter Plot with Linear Regression Line')
plt.legend()
plt.show()

print(f'Slope: {slope}')
print(f'Intercept: {intercept}')


# In[ ]:


import plotly.express as px


# In[ ]:


px.box(marvill, x='Price')

