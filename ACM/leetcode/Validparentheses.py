# -*- coding: utf-8 -*-
"""
À¨ºÅµÄÆ¥Åä
"""

s = "[[[[[[[[[]]]]]]]]])))))))))"
s = s.lower() #ºöÂÔ´óĞ¡Ğ´

lists = []
i = 0
result = True
count = 0
def check(a,b):
    if a == '(' and b == ')':
        return True
    if a == '[' and b == ']':
        return True
    if a == '{' and b == '}':
        return True
    return False
    
while i<len(s):
    if s[i] in ['(','[','{']:
        lists.append(s[i])
        
    elif s[i] in [')',']','}']:
        if len(lists) == 0:
            break
        sig = lists.pop()
        re = check(sig,s[i])
        if re:
            count += 2
        if not re:
            break
    i += 1

if count == len(s):
    print True 
else:
    print False