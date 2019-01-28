n=int(input())
a=[int(x) for x in input().split()]
a=sorted(a)
for i in range(n):
    print(a[i],end=' ')