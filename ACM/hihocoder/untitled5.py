# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:48:02 2016

@author: FF120
"""

while True:
    try:
        (x, y) = (int(x) for x in raw_input().split())
        print x + y
    except EOFError:
        break