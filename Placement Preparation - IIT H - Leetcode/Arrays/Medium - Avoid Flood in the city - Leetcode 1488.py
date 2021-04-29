"""
Avoid Flood in the city 
"""
import bisect 

def avoidFlood(rains):
    n=len(rains)
    out=[1]*n
    dry_lakes=[]
    fill_lakes={}
        
    for i in range(n):
        lake=rains[i]
        if lake>0: # rain day 
            out[i]=-1 
            if lake in fill_lakes:
                k=fill_lakes[lake]
                dry0=bisect.bisect_right(dry_lakes,k) # get upperbound 
                if dry0>=len(dry_lakes):
                    return []
                out[dry_lakes[dry0]]=lake 
                dry_lakes.remove(dry_lakes[dry0])
            fill_lakes[lake]=i
        else:
            dry_lakes.append(i)
    return out

# get inputs 
# size of array n 
n=int(input())
A=list(map(int,input().strip().split()[:n]))
print(avoidFlood(A))

        