
def Division3Contests(n,k,d,inp_list):
    """
    inputs : n,k,d - number of problem setters, k problems in a division 3 contest, number of days d 
             inp_list of length n - each element inp_list[i] contains the number of problems setted by person Pi 
    """
    total_problems_set=sum(inp_list)
    max_days=min(total_problems_set//k,d)
    return max_days 

# get the input test cases 
t=int(input())
while t>0:
    n,k,d=map(int,input().split())
    problem_list=list(map(int, input().split(' ')[:n]))
    print(Division3Contests(n,k,d,problem_list))
    t-=1
