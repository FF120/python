# -*- coding: utf-8 -*-
"""
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring
cases.
For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.
Note: Have you consider that the string might be empty? This is a good question to ask during an
interview.
For the purpose of this problem, we define empty string as valid palindrome.
"""

s = "A man, a plan, a canal: Panama"
s = s.lower() #ºöÂÔ´óÐ¡Ð´
result = True
if len(s) == 0:
    print result
i = 0
j = len(s)-1
while i < j:  
    if s[i].isalnum() and s[j].isalnum() and s[i] == s[j]:
        i += 1
        j -= 1
    elif not s[i].isalnum():
        i += 1
    elif  not s[j].isalnum():
        j -= 1
    else:
        result = False
        break

print result        