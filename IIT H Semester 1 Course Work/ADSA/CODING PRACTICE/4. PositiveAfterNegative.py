# Question Link : https://www.geeksforgeeks.org/move-negative-numbers-beginning-positive-end-constant-extra-space/

"""
An array contains both positive and negative numbers in random order. Rearrange the array elements so that 
all negative numbers appear before all positive numbers.
Examples : 
Input: -12, 11, -13, -5, 6, -7, 5, -3, -6
Output: -12 -13 -5 -7 -3 -6 11 6 5
Note: Order of elements is not important here.
"""

# function to sort 0,1,2 
def PositiveAfterNegative(input_list):
    """
    Function Description : 
                         This function takes the input list and returns the output list such that negative numbers
                         appears before positive numbers.
    Inputs : input_list - given input list of size n 
    Returns : input_list with negative numbers appear before positive numbers
    """
    n=len(input_list)
    neg_ind=0 
    for i in range(n):
        if input_list[i]<0:
            input_list[i],input_list[neg_ind]=input_list[neg_ind],input_list[i]
            neg_ind+=1 
    return input_list
    
# getting the inputs 

# number of test cases 
t=int(input())

while t>0:
    # size of the array 
    n=int(input())
    # getting the array input 
    input_list=list(map(int, input().split(' ')[:n]))
    print(PositiveAfterNegative(input_list))
    t=t-1

# TIME COMPLEXITY : O(n)
# SPACE COMPLEXITY : O(1)
        
        
                
    


