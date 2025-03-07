"""
Link : https://leetcode.com/problems/non-decreasing-array/ 

Given an array nums with n integers, your task is to check if it could become non-decreasing by modifying 
at most one element.

We define an array is non-decreasing if nums[i] <= nums[i + 1] holds for every i (0-based) such that 
(0 <= i <= n - 2).

Example 1:

Input: nums = [4,2,3]
Output: true
Explanation: You could modify the first 4 to 1 to get a non-decreasing array.

Example 2:

Input: nums = [4,2,1]
Output: false
Explanation: You can't get a non-decreasing array by modify at most one element.
"""
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        l=len(nums)
        count=0
        for i in range(l-1):
            if nums[i]>nums[i+1]:
                count+=1 
                if count>1:
                    return False 
                if (i>0 and nums[i-1]>nums[i+1]):
                    nums[i+1]=nums[i]
                else:
                    nums[i]=nums[i+1]
        return True 