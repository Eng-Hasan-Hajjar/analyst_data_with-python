##lesson8

## analysing data
##ex1
import pandas as pd

df = pd.read_csv('data2.csv')

print(df.head(6))

print(df.head())




print(df.tail(3)) 


##ex2
print(df.info()) 


"""

Data Cleaning
Data cleaning means fixing bad data in your data set.

Bad data could be:

Empty cells
Data in wrong format
Wrong data
Duplicates

"""
