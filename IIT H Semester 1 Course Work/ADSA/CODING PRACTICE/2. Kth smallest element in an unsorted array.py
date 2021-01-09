# Question Link : https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/


"""
Given an array arr[] and a number K where K is smaller than size of array, the task is to find the 
Kth smallest element in the given array. It is given that all array elements need not be distinct.
"""

# importing the libraries 
import heapq

# implementing the max heap 
class MyMaxHeap:
    def __init__(self,val=None):
        if val==None:
            self.val=[]
        else:
            self.val=[-i for i in val]
            heapq.heapify(self.val)
    
    def get_top(self):
        return -self.val[0]
        
    def replace_element(self,element):
        return heapq.heapreplace(self.val,-element)

def KthSmallest(input_list,k):
    """
    Function Description : 
                         This function takes the input list and k and returns the kth smallest integer in the
                         input_list
    Inputs : input_list - given input list of size n 
                     k - 1<=k<=len(input_list)
    Returns : kth smallest element in the input_list
    """
    k_max_heap=MyMaxHeap(input_list[:k])
    n=len(input_list)
    for i in range(k,n):
        if input_list[i]<k_max_heap.get_top():
            k_max_heap.replace_element(input_list[i])
    return k_max_heap.get_top()
    
    
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
    print(KthSmallest(input_list,k))
    t=t-1

# TIME COMPLEXITY : O(nlogk)
        
        
                
    


