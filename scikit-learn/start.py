#Ordinary Least Squares
###################################################|Ordinary Least Squares|#####################################################
#1. ��С���˷� 
#������Իع�Ĳ�����������������Ƴ�����y��ʵ�ʵ�y�Ĳ��ƽ���͵���Сֵ����һ�����Ż�����
#�÷���ֻ�����ڸ����Ա���������ʱ�򣬵������Ա���֮���໥�й�����ʱ�����ܲ���
from sklearn import linear_model
clf = linear_model.LinearRegression()
x = [[0,0],[1,1],[2,2]]
y = [0,1,2]
clf.fit (x,y)
predicted = clf.predict(x)
sum_array = (y - predicted) ** 2
mean = sum(sum_array) / len(sum_array)
print clf.coef_
##�������ģ����y = 0.5x1 + 0.5x2,  ����ֵ��ʵ��ֵ�������С����0.
##�в�
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
#������X�н������Թ�ϵ��ʱ�򣬻ᷢ�����ع����Ե����⣬�ô�ͳ����С���˷�����ʱ����Կ���
#Ridge Regression
#����С���в�ƽ���͵�ʱ�����һ���ͷ�������������ŵı�����Ϊ������
#collinearity��Ӱ��
from sklearn import linear_model
clf = linear_model.Ridge (alpha = .5) #���Ƴͷ�����
clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print  clf.coef_
print  clf.intercept_ 

##ʵ��ʹ�����������һ����õ�alphaֵ�أ�scikitʵ����һ����֤alphaֵ�ķ���
from sklearn import linear_model
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
clf.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
print clf.alpha_    

##############################################################################Lasso###############################
#3. �������ǳ�ϡ�裬Ҳ�����кܶ�w������0 ��ʱ����Lasso��Ч����ȽϺ�
#LassoҲ��������������ѡ��ϡ������ѡ��
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel  
# C ԽС��ѡ�������Խ�١�
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)     
model = SelectFromModel(lsvc, prefit=True)   
X_new = model.transform(X) #X_new��������ѡ��֮���������  

##L1-based sparse models��ȱ���ǵ����һ�����Ƶ�������ʱ��ֻ��ѡ�����е�һ����Ϊ�˽��������⣻
##����ʹ��������Զ��ѡ��֮���ٺϲ���RandomizedLasso ʵ��������һ���ӿ�

#randomized sparse models to be more powerful than standard F statistics at detecting non-zero features
#�ڷ�����������ʱ�����ϡ��ģ�ͱ�F���� ����
#���ʵ�ʵ������ ֻ��һС����������Y�ǹ����ģ�����֪�������ַǳ��٣���Ҫͨ����֪
#��X��yԤ�����е�������صĲ�����ʹ��ϡ������ģ�� Ҫ ���ڱ�׼��ͳ�Ʋ���                  