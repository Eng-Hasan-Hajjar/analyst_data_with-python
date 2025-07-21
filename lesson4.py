
##lesson4

##ex1
import numpy as np

##ex 1
# 
# Iterate through the array as a string:
arr = np.array([1, 2, 3])
print("ex1")
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)       

print(arr.dtype)

##ex2
##Iterate through every scalar element of the 2D array skipping 1 element:
##تكرار كل عنصر قياسي في المصفوفة ثنائية الأبعاد مع تخطي عنصر واحد:
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x)

##ex3
##Enumerate on following 1D arrays elements:

##قم بإحصاء عناصر المصفوفات أحادية الأبعاد التالية:

arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
  print(idx, x)


##ex4
#   Enumerate on following 2D array's elements:

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)

print(arr) 

