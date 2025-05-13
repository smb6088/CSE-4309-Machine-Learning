import sys
import math

pathname = sys.argv[1]

file_read = open(pathname,"r")
nums = []

for num in file_read:
    row = num.split()
    row = [float(num) for num in row]
    nums.append(row)

col_mean = []
col_len = len(nums[0])
row_len = len(nums)

for col in range(col_len):
    column = [nums[row][col] for row in range(row_len) ]
    mean = sum(column)/ row_len
    col_mean.append(mean)

col_std_dev = []
for col in range(col_len):
    column = [nums[row][col] for row in range(row_len) ]
    mean = col_mean[col]
    variance = sum((num - mean)**2 for num in column) / (row_len - 1)
    std_dev = math.sqrt(variance)
    col_std_dev.append(std_dev)

for i in range(col_len):
    print(f"Column {i+1}: mean = {col_mean[i]: .4f}, std = {col_std_dev[i]: .4f}" )