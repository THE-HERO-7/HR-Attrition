import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import pickle
df=pd.read_csv(r'C:\Users\Asus\Documents\python\assignment\Dataset - HR Employee Attrition.csv')
df
hr=pd.DataFrame(df)
hr.info()
df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
df_num=hr.select_dtypes(np.number)
df_num=pd.DataFrame(df_num)
for i in df_num:
    plt.figure(figsize=(2,2))
    sns.boxplot(y=i,data=df_num)
    plt.show()
####################Handling Outliers#########################
df_quantile=df_num.quantile([.25,.5,.75])
#monthly income
df_quantile['MonthlyIncome']
min=2911-(1.5*5469)
max=8380+(1.5*5459)
print(min,max)
hr['MonthlyIncome']=df_num['MonthlyIncome']=np.clip(df_num['MonthlyIncome'],-5292.5,16568.5)
sns.boxplot(df_num.MonthlyIncome)
#NUmCompainesWorked
df_quantile['NumCompaniesWorked']
min=1-(1.5*3)
max=4+(1.5*3)
print(min,max)
hr['NumCompaniesWorked']=df_num['NumCompaniesWorked']=np.clip(df_num['NumCompaniesWorked'],-3.5,8.5)
sns.boxplot(hr['NumCompaniesWorked'])
#performance
df_quantile['PerformanceRating']
hr['PerformanceRating']=df_num['PerformanceRating']=np.clip(df_num['PerformanceRating'],3,3)
sns.boxplot(hr['PerformanceRating'])
#stocklevel
df_quantile['StockOptionLevel']
min=0-(1.5*1)
max=1+(1.5*1)
print(min,max)
hr['StockOptionLevel']=df_num['StockOptionLevel']=np.clip(df_num['StockOptionLevel'],-1.5,2.5)
sns.boxplot(hr['StockOptionLevel'])
#total working hours
df_quantile['TotalWorkingYears']
min=6-(1.5*9)
max=15+(1.5*9)
print(min,max)
hr['TotalWorkingYears']=df_num['TotalWorkingYears']=np.clip(df_num['TotalWorkingYears'],-7.5,28.5)
sns.boxplot(hr['TotalWorkingYears'])
df_quantile['TrainingTimesLastYear']
min=2-(1.5*1)
max=3+(1.5*1)
print(min,max)
hr['TrainingTimesLastYear']=df_num['TrainingTimesLastYear']=np.clip(df_num['TrainingTimesLastYear'],0.5,4.5)
sns.boxplot(hr['TrainingTimesLastYear'])
#companyyears
df_quantile['YearsAtCompany']
min=3-(1.5*6)
max=9+(1.5*6)
print(min,max)
hr['YearsAtCompany']=df_num['YearsAtCompany']=np.clip(df_num['YearsAtCompany'],-6.0,18.0)
sns.boxplot(hr['YearsAtCompany'])
#yearsincurrent
df_quantile['YearsInCurrentRole']
min=2-(1.5*5)
max=7+(1.5*5)
print(min,max)
hr['YearsInCurrentRole']=df_num['YearsInCurrentRole']=np.clip(df_num['YearsInCurrentRole'],-5.5,14.5)
sns.boxplot(hr['YearsInCurrentRole'])
#lastpromotion
df_quantile['YearsSinceLastPromotion']
min=-(1.5*3)
max=3+(1.5*3)
print(min,max)
hr['YearsSinceLastPromotion']=df_num['YearsSinceLastPromotion']=np.clip(df_num['YearsSinceLastPromotion'],-4.5,7.5)
sns.boxplot(hr['YearsSinceLastPromotion'])
#currmanagmer
df_quantile['YearsWithCurrManager']
min=2-(1.5*5)
max=7+(1.5*5)
print(min,max)
hr['YearsWithCurrManager']=df_num['YearsWithCurrManager']=np.clip(df_num['YearsWithCurrManager'],-5.5,14.5)
sns.boxplot(hr['YearsWithCurrManager'])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in hr.columns:
    hr[i]=le.fit_transform(hr[i])
##############Visualization###################
plt.figure(figsize=(20, 20))
sns.heatmap(hr.corr(),fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant 
X = add_constant(hr)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)
for i in hr.columns:
    if(i!='Attrition'):
        sns.boxplot(x=df['Attrition'],y=hr[i],hue=df.Gender)
        plt.show()
    else:
        continue    
###############modelbuilding##############################
hr.info()
#selecting only varying variables
hr1=hr.iloc[:,[0,1,2,4,6,7,8,10,12,13,14,15,16,17,18,19,20,22,23,25,28,30,31,32,33,34]]
hr1.info()
##############rest were not that good###############
###############DECISION TREE########################
#distributing predictors and response
x= hr1.loc[:, hr1.columns!="Attrition"]  
y= hr.Attrition
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0) 
print(x.shape,y.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train) 
x_test=st_x.fit_transform(x_test)
#Fitting Decision Tree classifier to the training set  
from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy')  
print(classifier.fit(x_train, y_train))
#Predicting the test  
y_pred= classifier.predict(x_test)  
#Predicting the train
y1_pred= classifier.predict(x_train)  
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred) 
cm
from sklearn.metrics import accuracy_score
print('accurace:',accuracy_score(y_test,y_pred))
fpr2,tpr2,_=metrics.roc_curve(y_test,y_pred)
plt.plot(fpr2,tpr2)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()
from sklearn.metrics import confusion_matrix  
cm1= confusion_matrix(y_train, y1_pred) 
cm1
from sklearn.metrics import accuracy_score
print('accurace:',accuracy_score(y_train,y1_pred))
fpr3,tpr3,_=metrics.roc_curve(y_train,y1_pred)
plt.plot(fpr3,tpr3)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()
from sklearn import tree 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
tree.plot_tree(classifier,rounded=True,filled=True,proportion=True)
plt.show()
##########overfitted ,doing pre pruning#############
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth=4)
regtree.fit(x_train, y_train)
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)
from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, test_pred),
r2_score(y_test, test_pred))
print(mean_squared_error(y_train, train_pred),
r2_score(y_train, train_pred))
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 7)
regtree2.fit(x_train, y_train)
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)
print(mean_squared_error(y_test, test_pred2),
r2_score(y_test, test_pred2))
print(mean_squared_error(y_train, train_pred2),
r2_score(y_train, train_pred2))
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)
print(mean_squared_error(y_test, test_pred3),
r2_score(y_test, test_pred3))
print(mean_squared_error(y_train, train_pred3),
r2_score(y_train, train_pred3))
regtree4 = tree.DecisionTreeRegressor(max_depth = 9,min_samples_split = 5,min_samples_leaf = 4)
regtree4.fit(x_train, y_train)
test_pred4 = regtree4.predict(x_test)
train_pred4 = regtree4.predict(x_train)
print(mean_squared_error(y_test, test_pred4),
r2_score(y_test, test_pred3))
print(mean_squared_error(y_train, train_pred4),
r2_score(y_train, train_pred3))
###########still overfitted doing post pruning##############
path=classifier.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas,impurities=path.ccp_alphas,path.impurities
print("ccp alpha wil give list of values :",ccp_alphas)
print("----------------------------------------------------")
print("Impurities in Decision Tree :",impurities)
clfs=[]   #will store all the models here
for ccp_alpha in ccp_alphas:
    clf=DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    clf.fit(x_train,y_train)
    clfs.append(clf)
print("Last node in Decision tree is {} and ccp_alpha for last node is {}".format(clfs[-1].tree_.node_count,ccp_alphas[-1]))
train_scores = [classifier.score(x_train, y_train) for classifier in clfs]
test_scores = [clf.score(x_test, y_test) for classifier in clfs]
acc_df1=pd.DataFrame({'ccp_alpha':ccp_alphas,'impurities':impurities,'Train_ACC':train_scores,'Test_ACC':test_scores})
acc_df1['Error']=acc_df1['Train_ACC']-acc_df1['Test_ACC']
print(acc_df1.ccp_alpha[6:25])
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",drawstyle="steps-post")
ax.legend()
plt.show()
#0.002054,.004-.009999
classifier=DecisionTreeClassifier(random_state=0,ccp_alpha=0.002054)
classifier.fit(x_train,y_train)
pred1=classifier.predict(x_train) 
from sklearn.metrics import accuracy_score
print('accurace:',accuracy_score(y_train,pred1))
pred_1= classifier.predict(x_test) 
print('accurace:',accuracy_score(y_test,pred_1))
from sklearn.metrics import classification_report
cm_class_test=classification_report(y_test,pred_1)
print(cm_class_test)
fpr,tpr,_=metrics.roc_curve(y_test,pred_1)
fpr1,tpr1,_=metrics.roc_curve(y_train,pred1)
plt.plot(fpr,tpr)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()
plt.plot(fpr1,tpr1)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.show()
from sklearn import tree 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
tree.plot_tree(classifier,rounded=True,filled=True,proportion=True)
plt.show()
#dumping into pkl file
pickle.dump(classifier, open("model.pkl","wb"))

import os
os.getcwd()