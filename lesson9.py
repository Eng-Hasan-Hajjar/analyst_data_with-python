


##anas_data


"""

Data Cleaning
Data cleaning means fixing bad data in your data set.

Bad data could be:

Empty cells
Data in wrong format
Wrong data
Duplicates

"""
import pandas as pd

df = pd.read_csv('anas_data.csv')

new_df = df.dropna()

print(new_df.to_string())
################################

df.dropna(inplace = True)

print(df.to_string())


########

df = pd.read_csv('anas_data.csv')

df.fillna(999, inplace = True)
print(df.to_string())

############
df = pd.read_csv('anas_data.csv')
df.fillna({"Calories": 999}, inplace=True)
print(df.to_string())

###
df = pd.read_csv('anas_data.csv')

x = df["Calories"].mean()

df.fillna({"Calories": x}, inplace=True)
print(df.to_string())




#####
df = pd.read_csv('anas_data.csv')

x = df["Calories"].median()

df.fillna({"Calories": x}, inplace=True)
print(df.to_string())


###
df = pd.read_csv('anas_data.csv')

x = df["Calories"].mode()[0]

df.fillna({"Calories": x}, inplace=True)
print(df.to_string())

###
df = pd.read_csv('anas_data.csv')

x = df["Calories"].mode()

df.fillna({"Calories": x}, inplace=True)
print(df.to_string())



#####
df = pd.read_csv('anas_data.csv')
print(df.to_string())
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')

print(df.to_string())



#####
df = pd.read_csv('anas_data.csv')

df.loc[21,'Duration']= 45

print(df.to_string())


###
df = pd.read_csv('anas_data.csv')


for s in df.index:
    if df.loc[s,"Duration"] > 59:
        df.loc[s,'Duration']= 12


print(df.to_string())




###
df = pd.read_csv('anas_data.csv')


for s in df.index:
    if df.loc[s,"Duration"] > 59:
        df.drop(s,inplace=True)


print(df.to_string())