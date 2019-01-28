n=int(input())
b = True
i=2;
while(i<n):
    if(n%i==0):
        b=False
        break
    i+=1
if(n<2 or not b):
    print("Not Prime")
else:
    print("Prime")