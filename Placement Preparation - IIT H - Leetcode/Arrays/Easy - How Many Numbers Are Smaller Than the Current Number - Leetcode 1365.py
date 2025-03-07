"""
Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it. That is, 
for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].

Return the answer in an array.

Example 1:

Input: nums = [8,1,2,2,3]
Output: [4,0,1,1,3]
Explanation: 
For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3). 
For nums[1]=1 does not exist any smaller number than it.
For nums[2]=2 there exist one smaller number than it (1). 
For nums[3]=2 there exist one smaller number than it (1). 
For nums[4]=3 there exist three smaller numbers than it (1, 2 and 2).

Example 2:

Input: nums = [6,5,4,8]
Output: [2,1,0,3]

Example 3:

Input: nums = [7,7,7,7]
Output: [0,0,0,0]
 
Constraints:

2 <= nums.length <= 500
0 <= nums[i] <= 100
"""

def smallerNumbersThanCurrent(nums):
    temp=sorted(nums)
    t=temp[0]
    out=[0]*len(temp)
    for i in range(1,len(temp)):
        if temp[i]>=temp[i-1]:
            out[i]=out[i-1]+1
    dicti={}
    for x,y in zip(temp,out):
        if x not in dicti:
            dicti[x]=y
    for i in range(len(nums)):
        nums[i]=dicti[nums[i]]
    return nums
            

# get inputs 
# size of array of size n
n=int(input())
A=list(map(int,input().strip().split()[:n]))
print(smallerNumbersThanCurrent(A))

        