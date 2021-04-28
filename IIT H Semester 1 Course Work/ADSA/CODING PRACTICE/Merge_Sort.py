"""
Merge Sort 
"""
def Merge_Sort(A):
    """
    Sorts the given array 
    Input : Array A of size n 
    Returns : Sorted Array 
    """
    if len(A)<=1:
        return A 
    m=len(A)//2 
    # dividing the array into equal halves 
    L=A[:m]
    R=A[m:]
    Merge_Sort(L)
    Merge_Sort(R)
    
    i=0
    j=0
    k=0 
    
    nl=len(L)
    nr=len(R)
    
    while i<nl and j<nr:
        if L[i]<R[j]:
            A[k]=L[i]
            i+=1 
        else:
            A[k]=R[j]
            j+=1 
        k+=1 
        
    # copy the leftout elements in each array L/R to array A 
    while i<nl:
        A[k]=L[i]
        i+=1 
        k+=1 
    
    while j<nr:
        A[k]=R[j]
        j+=1 
        k+=1 
        
    return A
    
# get inputs 
# size of array n 
n=int(input())
A=list(map(int,input().strip().split()[:n]))
print(Merge_Sort(A))

        