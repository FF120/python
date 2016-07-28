# -*- coding: utf-8 -*-
"""
À¨ºÅµÄÆ¥Åä£¬×î³¤×Ó´®
"""

s = ")(((()()()))))"

def sumString(i,s):
    
    lists = []
    sums = 0
    while True:
        if s[i] == '(':
            lists.append(s[i])
            i += 1
        if s[i] == ')':
            if len(lists)==0:
                break
            lists.pop()
            sums += 2
            i += 1
        if i >= len(s):
            break
    return sums
sums = 0
for i in xrange(len(s)):
    ss = sumString(i,s)
    if  ss > sums:
        sums = ss
        
print sums