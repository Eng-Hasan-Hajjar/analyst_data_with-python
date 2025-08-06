##lesson7

##pandas
##ex1
##Load a CSV file into a Pandas DataFrame:
import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 


##ex2
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

myvar = pd.DataFrame(mydataset)

print(myvar)


##ex3
print(pd.__version__)
##ex4
a = [1, 7, 2]

myvar = pd.Series(a)

print(myvar)

##ex5
##Return the first value of the Series:
print(myvar[0])
##ex6

a = [1, 7, 2]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar)
##ex7
print(myvar["y"])

##ex8

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories)

print(myvar)
##ex9

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories, index = ["day1", "day2"])

print(myvar)

##ex10
##DataFrame
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myvar = pd.DataFrame(data)

print(myvar)
##ex11
##Return row 0:
#refer to the row index:
print(df.loc[0])
print(myvar.loc[0])
##Return row 0 and 1:
print(myvar.loc[[0,1,2]])

##ex12
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df) 
print(df.loc["day2"])


import pandas as pd

df = pd.read_csv('data2.csv')

print(df.to_string()) 



import pandas as pd

df = pd.read_csv('data2.csv')

print("new ex",df) 

"""
pd.options.display.max_rows = 3
print(pd.options.display.max_rows) 
print("new ex",df) 
"""





import pandas as pd

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)

print(df) 