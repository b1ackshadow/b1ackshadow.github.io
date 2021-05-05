## Numpy basics for Data Science

NumPy is short for numerical python. As the name dictates, it is used for numerical computing like linear algebra, statistical computations .

Most notable features of numpy:-

* ndarray - supports efficient multidimensional array for fast parallelised arithmetic operations (vectorized SIMD operations [[1]](https://towardsdatascience.com/decoding-the-performance-secret-of-worlds-most-popular-data-science-library-numpy-7a7da54b7d72/) )
* No loop - you can directly operate on the array without having to use python loops
* r/w files - supports directly saving arrays into disk and memory-mapped files(virtual memory space)
* maths - Linear algebra, statistical methods, Fourier transform etc
* C api - Numpy is built on C, C++ and FORTRAN . You can interact with such libs. 

You can learn more about numpy by heading over to [numpy docs](https://numpy.org/doc/stable/user/whatisnumpy.html)

#### Things we are going to learn:

* Fast vectorized array operations like cleaning, munging, filtering, transformations etc
* sorting, unique and set operations
* descriptive stats, summarizing data
* merging and joining datasets
* conditional logic as array exp
* Group-wise data manipulation



Just to give you a taste of why you would choose numpy over python sequences for array based operations, you can refer the below ipython test:


```python
# Note we are not including these inside the time block 
#because we want to test the speed of the operation not the data initialization

import numpy as np

arr = np.arange(100000)

lst = list(range(100000))
```


```python
%%timeit -n10
    
result = arr * 2
```

    178 µs ± 25.2 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%%timeit -n10
    
result = [each * 2 for each in lst]
```

    5.12 ms ± 777 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


To put it into perspective,

1 millisecond is equal to 1000 microsecond.

So python 5000 >>> numpy 178 microseconds

## 1.1 Numpy ndarray

Numpy homogeneous ndarray allows us to perform efficient array operations directly on the whole block of array as opposed to loops as if the array is just a scalar element(like in pure python)


```python
data_np = np.arange(10,dtype=int)
data_py = list(range(10))

#python list operation
squares_py = [ e ** 2 for e in data_py]
print("Python list operation:", squares_py)

#numpy scalar looking array operation
squares_np = data_np ** 2
print("Numpy array operation:",squares_np)

# similarly we can perform add, subtract, multiply and division etc.

#print(data_np - 2)
#print(data_np * 2)
#print(data_np / 2)
```

    Python list operation: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    Numpy array operation: [ 0  1  4  9 16 25 36 49 64 81]


Every numpy ndarray has attributes to check the dimension and the data type. 


```python
data = np.random.randn(10,20)
print("Dimension/shape of the data:",data.shape)
print("Datatype:",data.dtype)
```

    Dimension/shape of the data: (10, 20)
    Datatype: float64


I hope the above introduction was enough to get you excited about numpy. Now we will be going through basic numpy concepts one bye one. 

### Creating ndarrays

`array` function can be used to create an ndarray using any sequence(list, another array etc). This function returns a new array so you can use it to make copies of arrays as well.


```python
my_list = [1,2,3,4,5]
print(my_list)

my_arr = np.array(my_list)
print(my_arr)
```

    [1, 2, 3, 4, 5]
    [1 2 3 4 5]


Hmm. What if I have nested lists ?


```python
my_list = [list(range(10)),list(range(10))]
print(my_list,"\n")

my_arr = np.array(my_list)
print(my_arr)
```

    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] 
    
    [[0 1 2 3 4 5 6 7 8 9]
     [0 1 2 3 4 5 6 7 8 9]]


Generally, when working with matrices, ndarrys, people tend to use common array initializations like identity matrix, all-ones matrix, zero matrix or just an empty array without any values. For this reaons, numpy provides us with some helper functions to achieve the same.


```python
print(np.eye(4,4),"\n")

print(np.ones((4,2)),"\n")

print(np.zeros((4,2)),"\n")

print(np.empty([2, 2]),"\n")
```

    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]] 
    
    [[1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]] 
    
    [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]] 
    
    [[1. 1.]
     [1. 1.]] 
    



```python
# arange is similar to python range

print(np.arange(10))
```

    [0 1 2 3 4 5 6 7 8 9]


Numpy datatypes help it allocate memory and efficient work with it unlike python's dynamic memory allocation. You can use "dtype" attribute to work with datatypes in numpy


```python
arr = np.array([1,2,3])
print(arr.dtype)

np.array([1,2,3,4],dtype=np.int32).dtype
```

    int64





    dtype('int32')



You can covert numpy arrays from one datatype to another as long as they are compatible. If floating point numbers are converted to int, the decimal part is truncated. Also it's worth to note that astype always returns a copy of the data even if type is unchanged.


```python
arr = np.array([1,2,3])
print(arr.dtype)

print(arr.astype(np.int8).dtype)

arr = np.array([1.1,2.34,3.14])

print(arr.dtype)

print(arr.astype(int))
```

    int64
    int8
    float64
    [1 2 3]


You can covert array of string representing numbers to numeber dtype and vice verse using np.string_. However, since numpy works only with fixed sizes, it may truncate string input data.


```python
arr = np.array(["1","2","3"],dtype=np.string_)

print(arr)

print(arr.astype(int))

arr = arr.astype(int)

print(arr.astype(np.string_))
```

    [b'1' b'2' b'3']
    [1 2 3]
    [b'1' b'2' b'3']


### Arithmetic with numpy arrays

Without the ability to perform arithmetic operations, numpy would not be that useful. Any arithmetic operations between two equal sized arrays will yield elementwise operation. This involves batch operations on elements rather than using loops which is termed as *vectorization*.


```python
arr = np.array([[1,2,3],[4,5,6]])

print(arr)

print("\nAddition:\n", arr + arr)

print("\nSubtraction\n", arr - arr)

print("\nMultiplication:\n", arr * arr)

print("\nDivision:\n", arr / arr)
```

    [[1 2 3]
     [4 5 6]]
    
    Addition:
     [[ 2  4  6]
     [ 8 10 12]]
    
    Subtraction
     [[0 0 0]
     [0 0 0]]
    
    Multiplication:
     [[ 1  4  9]
     [16 25 36]]
    
    Division:
     [[1. 1. 1.]
     [1. 1. 1.]]


References:-

[1] https://towardsdatascience.com/decoding-the-performance-secret-of-worlds-most-popular-data-science-library-numpy-7a7da54b7d72
