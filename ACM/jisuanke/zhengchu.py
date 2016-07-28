
#整除问题
import string
a = raw_input()
ll = a.split()
if ( string.atoi(ll[0]) % string.atoi(ll[1]) == 0):
	print "YES"
else:
	print "NO"
	