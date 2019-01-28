N=int(input())
a=[]
for i in range(N):
  a.append(int(input()))
index1=N-1
index2=0
for i in range(N-1):
  if a[i]<=a[i+1]:
    index1=i
    break
for i in range(N-1,0,-1):
  if a[i]<=a[i-1]:
    index2=i
    break
flag=True
if index1!=index2:
  flag=False

if flag:
  print("true")
else:
  print("false")