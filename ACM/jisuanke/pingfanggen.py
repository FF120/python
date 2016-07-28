import string
import math
try:
    lists = []
    while True:
        line = raw_input()
        num = string.atoi(line)
        lists.append( int( math.sqrt(num) ))
except EOFError:
    for i in lists:
        print i