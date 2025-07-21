##lesson3
##Data Types in Python


"""

i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - string
U - unicode string
V - fixed chunk of memory for other type ( void )

"""


##ex1
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr.ndim)
print(arr.dtype)

##ex2

arr = np.array(['apple', 'banana', 'cherry'])

print(arr.dtype)
##EX3
arr = np.array([1, 2, 3, 4], dtype='S')

print(arr)
print(arr.dtype)
##ex4
##Create an array with data type 4 bytes integer:
arr = np.array([1, 2, 3, 4], dtype='i4')

print(arr)
print(arr.dtype)
##ex5
##A non integer string like 'a' can not be converted to integer (will raise an error):
###   arr = np.array(['a', '2', '3'], dtype='i')   ##error

##ex6
##Change data type from float to integer by using 'i' as parameter value:
arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')

print(newarr)
print(newarr.dtype)




##ex7
##Change data type from float to integer by using 'i' as parameter value:
arr = np.array([1.1, 2.1, 3.1], dtype='i1')
print(arr)
print(arr.dtype)

##ex8
##Change data type from integer to boolean:
arr = np.array([1, 0, 3])
newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)


##ex9
##Make a copy, change the original array, and display both arrays:
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)


#ex10
##vMake a view, change the original array, and display both arrays:
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42

print(arr)
print(x)


##ex11
##Make a view, change the view, and display both arrays:
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31

print(arr)
print(x)

##ex12
##Print the value of the base attribute to check if an array owns it's data or not:
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)

##ex13
##Print the shape of a 2-D array:
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr)
print(arr.shape)
##ex14
##Create an array with 5 dimensions using ndmin using a vector with values 1,2,3,4 and verify that last dimension has value 4:

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)

##ex15
"""
Convert the following 1-D array with 12 elements into a 2-D array.

The outermost dimension will have 4 arrays, each with 3 elements:
"""
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)
print(newarr.ndim,"New arr 4 * 3 with reshap()")
newarr = arr.reshape(3, 4)

print(newarr)


##ex16
"""
Convert the following 1-D array with 12 elements into a 3-D array.

The outermost dimension will have 2 arrays that contains 3 arrays, each with 2 elements:

"""

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print("---------------------------------- 3 d dime reshap()")
newarr = arr.reshape(2, 3, 2)

print(newarr)
print("---------------------------------- 3 d dime reshap()")
##ex17

##Try converting 1D array with 8 elements to a 2D array with 3 elements in each dimension (will raise an error):
"""
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(3, 3)

print(newarr)
"""


##Check if the returned array is a copy or a view:

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr.reshape(2, 4))
print(arr.reshape(2, 4).base)

##ex18
#Convert 1D array with 8 elements to 3D array with 2x2 elements:
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(2, 2, -1)

print(newarr)
print(newarr.ndim)
print(newarr.shape)


##ex19
##Convert the array into a 1D array:
arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)


##ex20
##Iterate on the elements of the following 1-D array:
arr = np.array([1, 2, 3])

for x in arr:
  print(x)


  ##ex21
##Iterate on the elements of the following 2-D array: 
# 

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)
##Ex22
# Iterate on each scalar element of the 2-D array: 
 
arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  for y in x:
    print(y) 



##23 ex
# Iterate on the elements of the following 3-D array:

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)    


##24 ex
# Iterate down to the scalars:
# 

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  for y in x:
    for z in y:
      print(z) 


##25 ex
# Iterate through the following 3-D array:
# 
# 

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
  print(x)  


##ex 26
# 
# Iterate through the array as a string:
arr = np.array([1, 2, 3])
print("ex26")
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)       