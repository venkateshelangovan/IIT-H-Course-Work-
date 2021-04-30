"""
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule.

Example 1:

Input: flowerbed = [1,0,0,0,1], n = 1
Output: true

Example 2:

Input: flowerbed = [1,0,0,0,1], n = 2
Output: false

Constraints:

1 <= flowerbed.length <= 2 * 104
flowerbed[i] is 0 or 1.
There are no two adjacent flowers in flowerbed.
0 <= n <= flowerbed.length

"""

def canPlaceFlowers(flowerbed,n):
    possible_flower=0 
    if n==0:
        return True 
    if len(flowerbed)==1:
        if flowerbed[0]==0:
            possible_flower=1
        return False if n>possible_flower else True 
            
    for i in range(0,len(flowerbed)):
        if i==0:
            if flowerbed[i+1]==0 and flowerbed[i]==0:
                possible_flower+=1 
                flowerbed[i]=1
        elif i==len(flowerbed)-1:
            if flowerbed[i-1]==0 and flowerbed[i]==0:
                possible_flower+=1 
                flowerbed[i]=1
        else:
            if flowerbed[i]==0 and flowerbed[i-1]==0 and flowerbed[i+1]==0:
                possible_flower+=1 
                flowerbed[i]=1
                    
    if n>possible_flower:
        return False 
    return True 
            

        