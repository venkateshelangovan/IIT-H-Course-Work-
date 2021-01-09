def Search(alphabets,k):
    """
    input : alphabets - list of alphabets 
            k - either 0 or 1 
    returns : if k==1 : return second half of list 
              else : return first half of list 
    """
    if k=='0':
        return alphabets[:(len(alphabets)//2)]
    else:
        return alphabets[(len(alphabets)//2):]

def getAlphabet(encoding):
    """
    input : encoding - string of length 4 
    returns : corresponding alphabet for the given encoding 
    """
    alphabets=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
    count=1
    for i in range(len(encoding)):
        alphabets=Search(alphabets,encoding[i])
    return alphabets[0]
        

def encode_string(string,n):
    """
    input : string - input string given 
            n - size of string 
    returns : decoded strings - expected output 
    """
    output=""
    for i in range(0,n,4):
        output+=getAlphabet(string[i:i+4])
    return output 
        
        
t=int(input())
while t>0:
    n=int(input())
    input_string=str(input())
    print(encode_string(input_string,n))
    t-=1