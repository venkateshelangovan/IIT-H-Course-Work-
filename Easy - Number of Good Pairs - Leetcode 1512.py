"""
Given an array of integers nums.

A pair (i,j) is called good if nums[i] == nums[j] and i < j.

Return the number of good pairs.

Example 1:

Input: nums = [1,2,3,1,1,3]
Output: 4
Explanation: There are 4 good pairs (0,3), (0,4), (3,4), (2,5) 0-indexed.

Example 2:

Input: nums = [1,1,1,1]
Output: 6
Explanation: Each pair in the array are good.
Example 3:

Input: nums = [1,2,3]
Output: 0
 
Constraints:

1 <= nums.length <= 100
1 <= nums[i] <= 100
"""
from scipy.special import comb

def NumIdenticalPairs(nums):
    dicti={}
    for x in nums:
        if x in dicti:
            dicti[x]+=1
        else:
            dicti[x]=1
    out=0
    for k,v in dicti.items():
        if v>1:
            out+=int(comb(v,2))
    return out
            

# get inputs 
# size of array of size n
n=int(input())
A=list(map(int,input().strip().split()[:n]))
print(NumIdenticalPairs(A))

        