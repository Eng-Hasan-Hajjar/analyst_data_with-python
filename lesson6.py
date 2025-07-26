## lesson 6
##Searching Arrays
##ex1
##Find the indexes where the value is 4:
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)


##ex2
##Find the indexes where the values are even:
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 0)

print(x)

##ex3
#Find the indexes where the values are odd:

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 1)

print(x)


#Search Sorted
##ex4
#Find the indexes where the value 7 should be inserted:

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 10)

print(x)



#ex5
##Find the indexes where the value 7 should be inserted, starting from the right:
arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7, side='right')

print(x)

#ex6
#Find the indexes where the values 2, 4, and 6 should be inserted:
arr = np.array([1, 3, 5, 7])

x = np.searchsorted(arr, [2, 4, 6])

print(x)


#NumPy Sorting Arrays
##ex7
arr = np.array([3, 2, 0, 1])

print(np.sort(arr))
##ex8
#Sort the array alphabetically:
arr = np.array(['banana', 'cherry', 'apple'])

print(np.sort(arr))


#ex9
#Sort a boolean array:
arr = np.array([True, False, True])

print(np.sort(arr))

##ex10
arr = np.array([[3, 2, 4], [5, 0, 1]])
print(arr)
print(np.sort(arr))



##NumPy Filter Array
##ex11
##Create an array from the elements on index 0 and 2:
arr = np.array([41, 42, 43, 44])

x = [True, False, False, False]

newarr = arr[x]

print(newarr)

#ex12
#Create a filter array that will return only values higher than 42:

arr = np.array([41, 42, 43, 44])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)


##ex13
##Create a filter array that will return only even elements from the original array:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)



#ex14
#Create a filter array that will return only values higher than 42:

arr = np.array([41, 42, 43, 44])

filter_arr = arr > 42

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)


#ex15

#Create a filter array that will return only even elements from the original array:
arr = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr = arr % 2 == 0

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)



##ex16
#Generate a random integer from 0 to 100:
from numpy import random
x = random.randint(100)
print(x)


##ex17
#Generate a random float from 0 to 100:
from numpy import random
x = random.rand(3)
print(x)
##ex18
#Generate a 1-D array containing 5 random integers from 0 to 100:

x=random.randint(100, size=(5))

print(x)

#ex19
#Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:
x=random.randint(100, size=(5,2))

print(x)

#ex20
##Generate a 1-D array containing 5 random floats:
x = random.rand(5)

print(x)
##ex21
#Generate a 2-D array with 3 rows, each row containing 5 random numbers:
x = random.rand(3,5)

print(x)

##ex22
#Return one of the values in an array:
x = random.choice([3, 5, 7, 9])

print(x)

##ex23
#Generate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9):
x = random.choice([3, 5, 7, 9], size=(3, 5))

print(x)

##ex24

"""

Generate a 1-D array containing 100 values, where each value has to be 3, 5, 7 or 9.

The probability for the value to be 3 is set to be 0.1

The probability for the value to be 5 is set to be 0.3

The probability for the value to be 7 is set to be 0.6

The probability for the value to be 9 is set to be 0

"""
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))

print(x)

##Ex25
##Same example as above, but return a 2-D array with 3 rows, each containing 5 values.
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))

print(x)
#ex26
#Randomly shuffle elements of following array:
arr = np.array([1, 2, 3, 4, 5])

random.shuffle(arr)

print(arr)
##ex27
##Generate a random permutation of elements of following array:
arr = np.array([1, 2, 3, 4, 5])

print(random.permutation(arr))
print(arr)


#ex28


import matplotlib.pyplot as plt
import seaborn as sns
"""
sns.displot([0, 1, 2, 3, 4, 5,6,7])

plt.show()


sns.displot([0, 1, 2, 3, 4, 5], kind="kde")

plt.show()

"""

##Normal Distribution


x = random.normal(size=(2, 3))

print(x)

"""

sns.displot(random.normal(size=1000), kind="kde")

plt.show()

"""

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
x = random.binomial(n=10, p=0.5, size=10)

sns.displot(random.binomial(n=10, p=0.5, size=1000))
print(x)
plt.show()


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

data = {
  "normal": random.normal(loc=50, scale=5, size=1000),
  "binomial": random.binomial(n=100, p=0.5, size=1000)
}

sns.displot(data, kind="kde")

plt.show()