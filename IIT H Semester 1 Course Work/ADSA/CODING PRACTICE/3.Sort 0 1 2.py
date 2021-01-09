# Question Link : https://www.geeksforgeeks.org/sort-an-array-of-0s-1s-and-2s/


"""
Given an array A[] consisting 0s, 1s and 2s. The task is to write a function that sorts the given array. 
The functions should put all 0s first, then all 1s and all 2s in last.
"""

# function to sort 0,1,2 
def Sort012(input_list):
    """
    Function Description : 
                         This function takes the input list and returns the output list such that all 0s first,
                         then all 1s and all 2s in last.
    Inputs : input_list - given input list of size n 
    Returns : sorted innput_list
    """
    n=len(input_list)
    first=0
    middle=0 
    last=n-1
    while middle<=last:
        if input_list[middle]==0:
            input_list[first],input_list[middle]=input_list[middle],input_list[first]
            middle+=1 
            first+=1 
        elif input_list[middle]==1:
            middle+=1  
        else:
            input_list[middle],input_list[last]=input_list[last],input_list[middle]
            last-=1 
    return input_list
    
    
# getting the inputs 

# number of test cases 
t=int(input())

while t>0:
    # size of the array 
    n=int(input())
    # getting the array input 
    input_list=list(map(int, input().split(' ')[:n]))
    print(Sort012(input_list))
    t=t-1

# TIME COMPLEXITY : O(n)
# SPACE COMPLEXITY : O(1)
        
        
                
    


