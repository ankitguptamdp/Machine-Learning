n=int(input())
for i in range(n):
    for j in range(n-i):
        print(int(j+1),end=' ')
    for j in range((i-1)*2+1):
        print('*',end=' ')
    print('')