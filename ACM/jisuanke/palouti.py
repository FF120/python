# -*- coding: utf-8 -*-
# 爬楼梯  --动态规划
import string
a = raw_input()  #获取用户输入
a = string.atoi(a)

dp = []
dp.append(0)
dp.append(1)
dp.append(2)
for i in xrange(3,a+1):
    dp.append( (dp[i-1] + dp[i-2]) )

print dp[a]