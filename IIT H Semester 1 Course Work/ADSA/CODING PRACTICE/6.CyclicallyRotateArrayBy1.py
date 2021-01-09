# Question Link : https://www.geeksforgeeks.org/c-program-cyclically-rotate-array-one/

"""
Given an array, cyclically rotate the array clockwise by one.

Examples:

Input:  arr[] = {1, 2, 3, 4, 5}
Output: arr[] = {5, 1, 2, 3, 4}
"""

# function to get cyclically rotated array by 1
def cyclicRotationBy1(inp_list):
    """
    Function Description : 
                         This function takes the input list and returns the cyclically rotated array by one
    Inputs : inp_list 
    Returns : cyclically rotate the inp_list by 1 
    """
    n=len(inp_list)
    rotated_last_element=inp_list[n-1]
    for i in range(n-1,0,-1):
        inp_list[i]=inp_list[i-1]
    inp_list[0]=rotated_last_element
    return ",".join(map(str,inp_list))
# getting the inputs 

# number of test cases 
t=int(input())

while t>0:
    # size of the array 1
    n=int(input())
    # getting the array 1 input 
    input_list=list(map(int, input().split(' ')[:n]))
    print(cyclicRotationBy1(input_list))
    t=t-1

# TIME COMPLEXITY : O(n)

        
                
    


