# -*- coding: utf-8 -*-
import string
try:
    lists = []
    while True:
        line = raw_input().split()
        num = string.atoi(line[0]) + string.atoi(line[0])
        print num
except EOFError:
    pass