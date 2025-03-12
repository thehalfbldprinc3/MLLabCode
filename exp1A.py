import numpy as np

# 1. Create Arrays
arr1 = np.array([1, 2, 3, 4, 5])  # 1D array
print("1D Array:", arr1)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
print("\n2D Array:\n", arr2)

arr3 = np.zeros((2, 3))  # Array of zeros
print("\nArray of zeros:\n", arr3)

arr4 = np.ones((3, 3))  # Array of ones
print("\nArray of ones:\n", arr4)

arr5 = np.eye(4)  # Identity matrix
print("\nIdentity Matrix:\n", arr5)

arr6 = np.linspace(0, 10, 5)  # 5 equally spaced points between 0 and 10
print("\nLinspace:\n", arr6)

arr7 = np.arange(0, 10, 2)  # Array with a step
print("\nArange:\n", arr7)

# 2. Array Properties
print("\nArray shape:", arr2.shape)
print("Array data type:", arr2.dtype)
print("Array size:", arr2.size)

# 3. Indexing and Slicing
print("\nElement at index 2 of arr1:", arr1[2])
print("Row 1 of arr2:", arr2[1])
print("Slice of arr1 [1:4]:", arr1[1:4])
print("Last element of arr1:", arr1[-1])

# 4. Reshaping Arrays
arr8 = np.arange(1, 13).reshape(3, 4)  # Reshape 1D to 2D
print("\nReshaped Array (3x4):\n", arr8)

# 5. Mathematical Operations
arr9 = np.array([1, 2, 3, 4])
arr10 = np.array([5, 6, 7, 8])
print("\nElement-wise addition:", arr9 + arr10)
print("Element-wise multiplication:", arr9 * arr10)
print("Square of arr9:", np.square(arr9))
print("Sum of arr9:", np.sum(arr9))
print("Mean of arr9:", np.mean(arr9))

# 6. Broadcasting
arr11 = np.array([[1], [2], [3]])
arr12 = np.array([10, 20, 30])
print("\nBroadcasting example:\n", arr11 + arr12)

# 7. Stacking and Concatenation
stacked = np.vstack((arr9, arr10))  # Vertical stacking
print("\nVertically stacked arrays:\n", stacked)

concatenated = np.concatenate((arr9, arr10))  # Concatenation
print("\nConcatenated arrays:", concatenated)

# 8. Logical Operations
print("\nElements greater than 2 in arr9:", arr9 > 2)

# 9. Random Arrays
random_arr = np.random.rand(2, 3)  # Uniform distribution
print("\nRandom Array (Uniform Distribution):\n", random_arr)

random_int_arr = np.random.randint(0, 10, (3, 3))  # Random integers
print("\nRandom Integer Array:\n", random_int_arr)

# 10. Transposing and Matrix Operations
print("\nOriginal arr2:\n", arr2)
print("Transposed arr2:\n", arr2.T)

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
dot_product = np.dot(matrix1, matrix2)  # Matrix multiplication
print("\nDot Product of matrices:\n", dot_product)

# 11. Flattening Arrays
flat_array = arr8.flatten()  # Convert 2D to 1D
print("\nFlattened Array:", flat_array)

# 12. Sorting
unsorted = np.array([3, 1, 4, 1, 5, 9])
sorted_array = np.sort(unsorted)  # Sort array
print("\nSorted Array:", sorted_array)

# 13. Universal Functions (ufunc)
arr13 = np.array([1, 2, 3, 4, 5])
print("\nExponential:", np.exp(arr13))
print("Logarithm:", np.log(arr13))
print("Sine values:", np.sin(arr13))