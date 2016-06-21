#Ordinary Least Squares
###################################################|Ordinary Least Squares|#####################################################
#1. 最小二乘法 
#求解线性回归的参数，本质上是求估计出来的y和实际的y的差的平方和的最小值，是一个最优化问题
#该方法只适用于各个自变量独立的时候，当各个自变量之间相互有关联的时候，性能不好
from sklearn import linear_model
clf = linear_model.LinearRegression()
x = [[0,0],[1,1],[2,2]]
y = [0,1,2]
clf.fit (x,y)
predicted = clf.predict(x)
sum_array = (y - predicted) ** 2
mean = sum(sum_array) / len(sum_array)
print clf.coef_
##计算出的模型是y = 0.5x1 + 0.5x2,  估计值和实际值的误差最小，是0.
##残差
print mean


from sklearn import linear_model
clf = linear_model.LinearRegression()
x = [[0,0],[1,1],[2,2]]
y = [0,0.9999,2.0001]
clf.fit (x,y)
print clf.coef_
##[ 0.500025  0.500025]
#################################################################Ridge Regression#################################
#2. Ridge Regression
#################################################################
#当矩阵X有近似线性关系的时候，会发生多重共线性的问题，用传统的最小二乘法，这时候可以考虑
#Ridge Regression
#在最小化残差平方和的时候加入一个惩罚项，用来控制缩放的比例，为了消除
#collinearity的影响
from sklearn import linear_model
clf = linear_model.Ridge (alpha = .5) #控制惩罚参数
clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print  clf.coef_
print  clf.intercept_ 

##实际使用中如果设置一个最好的alpha值呢，scikit实现了一个验证alpha值的方法
from sklearn import linear_model
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
clf.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
print clf.alpha_    

##############################################################################Lasso###############################
#3. 当参数非常稀疏，也就是有很多w都等于0 的时候，用Lasso的效果会比较好
#Lasso也可以用来做特征选择，稀疏特征选择
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel  
# C 越小，选择的特征越少。
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)     
model = SelectFromModel(lsvc, prefit=True)   
X_new = model.transform(X) #X_new就是特征选择之后的特征。  

##L1-based sparse models的缺陷是当面对一组相似的特征的时候，只会选择其中的一个；为了解决这个问题；
##可以使用随机策略多次选择之后再合并，RandomizedLasso 实现了这样一个接口

#randomized sparse models to be more powerful than standard F statistics at detecting non-zero features
#在非零特征检测的时候，随机稀疏模型比F检验 更好
#如果实际的情况是 只有一小部分特征和Y是关联的，而已知的数据又非常少，需要通过已知
#的X和y预测其中的线性相关的参数，使用稀疏线性模型 要 好于标准的统计测试                  