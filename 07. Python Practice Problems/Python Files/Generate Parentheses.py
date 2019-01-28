def parenthesis(opened,closed,n,s=[]):
    if closed==n:
        print(''.join(s))
        return
    if closed<opened:
        s.append(')')
        parenthesis(opened,closed+1,n,s)
        s.pop()
    if opened<n:
        s.append('(')
        parenthesis(opened+1,closed,n,s)
        s.pop()

n=int(input())
parenthesis(0,0,n)