"""
Given an array of integers arr, return true if and only if it is a valid mountain array.

Recall that arr is a mountain array if and only if:

arr.length >= 3
There exists some i with 0 < i < arr.length - 1 such that:
arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
arr[i] > arr[i + 1] > ... > arr[arr.length - 1]

Example 1:

Input: arr = [2,1]
Output: false

Example 2:

Input: arr = [3,5,5]
Output: false

Example 3:

Input: arr = [0,3,2,1]
Output: true
 
Constraints:

1 <= arr.length <= 104
0 <= arr[i] <= 104

"""

def validMountainArray(arr):
    if len(arr)<3:
        return False 
    if arr[0]<arr[1]:
        increasing=True 
    else:
        return False 
        
    decreasing=False 
    for i in range(1,len(arr)-1):
        if arr[i]>arr[i+1] and decreasing==False:
            decreasing=True
            increasing=False
        if arr[i]<arr[i+1] and increasing==False:
            return False 
        if arr[i]==arr[i+1]:
            return False 
    if increasing==False and decreasing==True:
        return True 
    return False 
            

        