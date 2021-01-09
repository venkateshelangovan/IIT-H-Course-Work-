# Question Link : https://www.geeksforgeeks.org/union-and-intersection-of-two-sorted-arrays-2/

"""
Given two sorted arrays, find their union and intersection.
Example:

Input : arr1[] = {1, 3, 4, 5, 7}
        arr2[] = {2, 3, 5, 6} 
Output : Union : {1, 2, 3, 4, 5, 6, 7} 
         Intersection : {3, 5}

Input : arr1[] = {2, 5, 6}
        arr2[] = {4, 6, 8, 10} 
Output : Union : {2, 4, 5, 6, 8, 10} 
         Intersection : {6}
"""

# function to get Union
def getUnion(arr1,arr2):
    """
    Function Description : 
                         This function takes the arr1 and arr2 and return the Union of arr1 and arr2
    Inputs : arr1 - given input list of size n1 
             arr2 - given input list of size n2
    Returns : Union of arr1 and arr2 
    """
    union_out=[]
    n1=len(arr1)
    n2=len(arr2)
    i=0 
    j=0 
    
    while i<n1 and j<n2:
        if arr1[i]<arr2[j]:
            union_out.append(arr1[i])
            i+=1 
        elif arr1[i]>arr2[j]:
            union_out.append(arr2[j])
            j+=1 
        else:
            union_out.append(arr1[i])
            i+=1 
            j+=1 
    
    while i<n1:
        union_out.append(arr1[i])
        i+=1 
    while j<n2:
        union_out.append(arr2[j])
        j+=1 
    return ",".join(map(str, union_out)) 
    
# function to get Intersection
def getIntersection(arr1,arr2):
    """
    Function Description : 
                         This function takes the arr1 and arr2 and return the Intersection of arr1 and arr2
    Inputs : arr1 - given input list of size n1 
             arr2 - given input list of size n2
    Returns : Intersection of arr1 and arr2 
    """
    intersection_out=[]
    n1=len(arr1)
    n2=len(arr2)
    i=0 
    j=0 
    
    while i<n1 and j<n2:
        if arr1[i]<arr2[j]:
            i+=1 
        elif arr1[i]>arr2[j]:
            j+=1 
        else:
            intersection_out.append(arr1[i])
            i+=1 
            j+=1 
    
    return ",".join(map(str, intersection_out))
# getting the inputs 

# number of test cases 
t=int(input())

while t>0:
    # size of the array 1
    n1=int(input())
    # getting the array 1 input 
    input_list1=list(map(int, input().split(' ')[:n1]))
    # size of the array 2
    n2=int(input())
    # getting the array 2 input 
    input_list2=list(map(int, input().split(' ')[:n2]))
    print(getUnion(input_list1,input_list2))
    print(getIntersection(input_list1,input_list2))
    t=t-1

# TIME COMPLEXITY : O(max(n1,n2))
# SPACE COMPLEXITY : O(n+m)
        
        
                
    


