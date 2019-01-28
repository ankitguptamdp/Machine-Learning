t=int(input())
for k in range(t):
    n=int(input())
    i=0
    while(i<=n):
        j=i
        while((j*j)<=n):
            if((i*i + j*j)==n):
                print('',end='(')
                print(i,end=',')
                print(j,end=')')
                print('',end=' ')
            j+=1
        i+=1
    print('')