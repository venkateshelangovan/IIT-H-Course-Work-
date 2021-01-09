# Question Link : https://www.geeksforgeeks.org/check-if-a-key-is-present-in-every-segment-of-size-k-in-an-array/

def keyInKSegments(input_list,k,x):
    """
    Function Description : 
                         This function takes the input list , window size k and key x as input and return True
                         if the key is found in input list of every window size k and False otherwise.
    Inputs : input_list - given input list of size n 
                     k - window size 
                     x - key to be searched in every window size
    Returns : True - if the key x is found in input list in every window size k 
              False - Otherwise 
    """
    i=0
    n=len(input_list)
    while i<n:
        j=0
        while j<k:
            if input_list[i+j]!=x:
                j+=1 
            else:
                break
        if j==k:
            # if this satisfies then key x is not in one particular window of size k 
            return False
        i+=k 
        if i+k-1>=n:
            break
    if i==n:
        """
        if this condition satisfies we reached last index and there are no more elements to form window of 
        size k.This will happen when n is one of the multiples of k
        """
        return True 
    
    # if n is not the multiple of k 
    j=i-k 
    while j<n:
        if input_list[j]!=x:
            j+=1 
        else:
            break
    
    if j==n:
        # occurs when element missing in final window 
        return False
    
    return True # indicates element is found in last window aswell
    
    
# getting the inputs 

# number of test cases 
t=int(input())

while t>0:
    # size of the array 
    n=int(input())
    # getting the array input 
    input_list=list(map(int, input().split(' ')[:n]))
    # segment size
    k=int(input())
    # key element to be searched 
    x=int(input())
    print(keyInKSegments(input_list,k,x))
    t=t-1

# TIME COMPLEXITY : O(n)

"""
Sample Input :
2
7
12 31 32 11 13 12 13
3
12
6
18 19 20 18 20 18
2
19

Sample Output :
True
False
"""          
    


