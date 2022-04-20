#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[86]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")


# # Importing Dataset
# ##### dataset link:
# https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients?select=column_2C_weka.csv
# ## Content
# Field Descriptions:
# 
# Each patient is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (each one is a column):
# 
# pelvic incidence
# pelvic tilt
# lumbar lordosis angle
# sacral slope
# pelvic radius
# grade of spondylolisthesis

# In[4]:


dataset=pd.read_csv('Dataset/orthopedic dataset.csv')
dataset.head(10)


# In[5]:


print("column Names:",dataset.columns.values)


# In[6]:


dataset.dtypes


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


dataset['class'].unique()


# # Data Analysis

# In[10]:


dataset_num=dataset.select_dtypes(include=['float64','int64'])
# print(dataset_num.head())


# In[11]:


sns.set_style("whitegrid")
sns.pairplot(dataset,hue="class",diag_kind = "kde",kind = "scatter")
plt.show()


# In[12]:


for name in dataset.columns.values[:-1]:
    sns.FacetGrid(dataset, hue="class",height=5,aspect=2).map(sns.distplot, name).add_legend()
plt.show()


# In[13]:


dataset.boxplot(column=['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle',
 'sacral_slope' ,'pelvic_radius', 'degree_spondylolisthesis' ],figsize=(20,7))
plt.title("Boxblot of data")
plt.show()


# In[15]:


print(dataset['class'].value_counts())
plt.figure(figsize=(10,5))
sns.countplot('class',data=dataset)
plt.title("class distribution")
plt.show()


# In[17]:


dataset=dataset.replace({
    'class':{'Normal':1,'Abnormal':0}
})
dataset.head()


# In[18]:


corr=dataset.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr,annot=True,cmap='cubehelix_r',square=True)
plt.show()


# # Cheking: Missing values

# In[19]:


dataset.isnull().sum(axis=0).plot.bar()
plt.yticks(np.arange(0,5))
plt.show()


# # Splitting dataset

# In[20]:


data_input=dataset.drop(columns='class')
data_output=dataset['class']
data_input.head()


# In[21]:


data_output.head()


# In[22]:


X, X_test, y, y_test = train_test_split(data_input, data_output, test_size=0.30, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=0)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('------------------------')
print('X_val:', X_val.shape)
print('y_val:', y_val.shape)
print('------------------------')
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# In[23]:


y_train.value_counts()


# In[27]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=1.0, random_state=0)
X_train_balanced, y_train_balaned = ros.fit_resample(X_train, y_train)
y_train_balaned.value_counts()


# # Helper function: Evaluate model

# In[28]:


def eval_model(model,X_train,y_train,X_val,y_val):
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    
    return(acc_train,acc_val)


# # Applying KNN

# # Hyperparameter tuning
# ### Tuning n_neighbors

# In[51]:


n_nighbors_values=np.arange(1,31)
accuracy_train_values = []
accuracy_val_values = []

for n_nigh in n_nighbors_values:
    knn_model=KNeighborsClassifier(n_neighbors=n_nigh)
    knn_model.fit(X_train_balanced,y_train_balaned)
    y_pred_train=knn_model.predict(X_train_balanced)
    y_pred_val=knn_model.predict(X_val)
    acc_tarin=accuracy_score(y_train_balaned,y_pred_train)
    acc_val=accuracy_score(y_val,y_pred_val)
    accuracy_train_values.append(acc_tarin)
    accuracy_val_values.append(acc_val)
    
    
print("Best accuracy of validation is {} with K = {}".format(max(accuracy_val_values),1+accuracy_val_values.index(max(accuracy_val_values))))
print("Best accuracy is of train {} with K = {}".format(max(accuracy_train_values),1+accuracy_train_values.index(max(accuracy_train_values))))


# In[52]:


results_knn=pd.DataFrame({
    'n_neighbours': n_nighbors_values,
    'accuracy_train': accuracy_train_values,
    'accuracy_val': accuracy_val_values
})
results_knn


# In[54]:


results_knn.plot(x='n_neighbours', y=['accuracy_train', 'accuracy_val'], figsize=(15, 7))
plt.title('Train "vs" Validation Accuracy\n',fontsize=25)
plt.xlabel('The number of Neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(np.arange(1,31))
plt.grid()
plt.show()


# According to the previous plot, we select n_neighbours = `11` as the best value for n_neighbours

# In[90]:


weights_values = ['uniform', 'distance']
accuracy_train_values = []
accuracy_val_values = []

for w in weights_values:
    knn_model=KNeighborsClassifier(n_neighbors=11,weights=w)
    knn_model.fit(X_train_balanced,y_train_balaned)
    y_pred_train=knn_model.predict(X_train_balanced)
    y_pred_val=knn_model.predict(X_val)
    acc_tarin=accuracy_score(y_train_balaned,y_pred_train)
    acc_val=accuracy_score(y_val,y_pred_val)
    accuracy_train_values.append(acc_tarin)
    accuracy_val_values.append(acc_val)
    
    
    
results_knn=pd.DataFrame({
    'weights': weights_values,
    'accuracy_train': accuracy_train_values,
    'accuracy_val': accuracy_val_values
})
results_knn


# In[91]:


results_knn.plot.bar(x='weights', y=['accuracy_train', 'accuracy_val'],color=['red','blue'])
plt.title('Test "vs" Validation Accuracy\n',fontsize=25)
plt.xlabel('The Weights',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(rotation=360)
plt.legend(loc='lower center')
plt.grid(axis='y')
plt.show()


# According to the previous plot we will select the weight = `uniform`

# # Final Knn Test

# In[92]:


best_knn_model=KNeighborsClassifier(n_neighbors=11,weights='uniform')
best_knn_model.fit(X_train_balanced,y_train_balaned)
y_pred_test=best_knn_model.predict(X_test)
knn_acc_final=accuracy_score(y_test,y_pred_test)
print(round(knn_acc_final,3)*100,"%")


# In[60]:


print("the Real first 10 value \n")
for i in y_test[:10]:
    print(i,end=" - ")
print('\n---------------------------------------\n')
print("the prediction first 10 value \n")
for i in y_pred_test[:10]:
    print(i,end=" - ")


# In[61]:


#calculate confusion matrix
plt.figure(figsize=(10,8))
cm_knn=confusion_matrix(y_test, y_pred_test)
print('confusion matrix is \n',cm_knn)
sns.heatmap(cm_knn,annot=True,center=True)
plt.show()


# In[82]:


plt.style.use('classic')
plt.pie([knn_acc_final,1-knn_acc_final],labels=('Classification True','Classification False')
        ,explode=[0.1,0.1],autopct="%1.1f%%",shadow=True,colors=['green','red'])
plt.axis('equal')
plt.title(' KNN Classification Accuracy\n',fontsize=35)
plt.show()


# # Applying naive bayes NB

# ## Model 1: Gaussian Naive Bayes

# In[63]:


nb_model1 = GaussianNB() 
acc_train1, acc_val1 = eval_model(nb_model1, X_train, y_train, X_val, y_val)
print(acc_train1)
print(acc_val1)


# ## Model 2:  Bernoulli Naive Bayes

# In[64]:


nb_model2 = BernoulliNB()
acc_train2, acc_val2 = eval_model(nb_model2, X_train, y_train, X_val, y_val)
print(acc_train2)
print(acc_val2)


# In[97]:


results = pd.DataFrame({
    'model': ['GaussianNB',  'BernoulliNB'],
    'acc_train': [acc_train1, acc_train2],
    'acc_val': [acc_val1, acc_val2],
})

results


# In[98]:


results.plot.bar(x='model',y=['acc_train','acc_val'])
plt.title('Test "vs" Validation Accuracy of NB\n',fontsize=25)
plt.xlabel('The models',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(rotation=360)
plt.show()


# # final NB test

# In[76]:


best_nb_model=GaussianNB()
best_nb_model.fit(X_train,y_train)
y_pred_test=best_nb_model.predict(X_test)
model_nb_final=accuracy_score(y_test,y_pred_test)
print(round(model_nb_final,3)*100,"%")


# In[77]:


print("the Real first 10 value \n")
for i in y_test[:10]:
    print(i,end=" - ")
print('\n---------------------------------------\n')
print("the prediction first 10 value \n")
for i in y_pred_test[:10]:
    print(i,end=" - ")


# In[78]:


#calculate confusion matrix
plt.figure(figsize=(10,8))
cm_nb=confusion_matrix(y_test, y_pred_test)
print('confusion matrix is \n',cm_nb)
sns.heatmap(cm_nb,annot=True,center=True)
plt.show()


# In[81]:


plt.style.use('classic')
plt.pie([model_nb_final,1-model_nb_final],labels=('Classification True','Classification False')
        ,explode=[0.1,0.1],autopct="%1.1f%%",shadow=True,colors=['green','red'])
plt.axis('equal')
plt.title(' NB Classification Accuracy\n',fontsize=35)
plt.show()


# # Decision tree classifier

# In[87]:


max_depth_values = list(range(1, 21))
acc_train_values = []
acc_val_values = []

for max_depth in max_depth_values:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    acc_train_values.append(acc_train)
    acc_val_values.append(acc_val)
    
    
    
print("Best accuracy of validation is {} with Mac Depth = {}".format(max(acc_val_values),1+acc_val_values.index(max(acc_val_values))))
print("Best accuracy is of train {} with Max Depth = {}".format(max(acc_train_values),1+acc_train_values.index(max(acc_train_values))))


# In[88]:


results = pd.DataFrame({
    'max_depth': max_depth_values,
    'acc_train': acc_train_values,
    'acc_val': acc_val_values,
})

results


# In[89]:


results.plot(x='max_depth', y=['acc_train', 'acc_val'], figsize=(10, 6))
plt.xticks(np.arange(1, 21))
plt.title('Train "vs" Validation Accuracy of "Max Depth" \n',fontsize=25)
plt.xlabel('The Max Depth num',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.grid(axis='both')
plt.show()


# We select max_depth = `4`

# In[100]:


criterion_values = ['gini', 'entropy']
acc_train_values = []
acc_val_values = []

for criterion in criterion_values:
    model = DecisionTreeClassifier(max_depth=4, criterion=criterion, random_state=0)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    acc_train_values.append(acc_train)
    acc_val_values.append(acc_val)
    
results = pd.DataFrame({
    'criterion': criterion_values,
    'acc_train': acc_train_values,
    'acc_val': acc_val_values,
})

results


# In[101]:


results.plot.bar(x='criterion', y=['acc_train', 'acc_val'], figsize=(10, 6))
plt.grid(axis='y')
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.title('Test "vs" Validation Accuracy of DT\n',fontsize=25)
plt.xlabel('The criterion',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(rotation=360)
plt.legend(loc='lower center')
plt.grid(axis='y')
plt.show()


# We select criterion = `entropy`

# # Final Test

# In[104]:


best_dt_model=DecisionTreeClassifier(max_depth=4,criterion='entropy',random_state=0)
best_dt_model.fit(X_train,y_train)
y_pred_test=best_dt_model.predict(X_test)
model_dt_final=accuracy_score(y_test,y_pred_test)
print(round(model_dt_final,3)*100,"%")


# In[105]:


print("the Real first 10 value \n")
for i in y_test[:10]:
    print(i,end=" - ")
print('\n---------------------------------------\n')
print("the prediction first 10 value \n")
for i in y_pred_test[:10]:
    print(i,end=" - ")


# In[106]:


#calculate confusion matrix
plt.figure(figsize=(10,8))
cm_dt=confusion_matrix(y_test, y_pred_test)
print('confusion matrix is \n',cm_dt)
sns.heatmap(cm_dt,annot=True,center=True)
plt.show()


# In[107]:


plt.style.use('classic')
plt.pie([model_dt_final,1-model_dt_final],labels=('Classification True','Classification False')
        ,explode=[0.1,0.1],autopct="%1.1f%%",shadow=True,colors=['green','red'])
plt.axis('equal')
plt.title(' DT Classification Accuracy\n',fontsize=35)
plt.show()


# In[114]:


plt.figure(figsize=(20, 10))
plt.bar(X_train.columns, best_dt_model.feature_importances_)
plt.title("Analysing feature importances\n",fontsize=25)
plt.xticks(rotation=90)
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# In[115]:


from sklearn.tree import plot_tree

plt.figure(figsize=(20, 5))
plot_tree(best_dt_model, feature_names=X_train.columns, class_names=['Normal', 'Abnormal'])
plt.show()


# # End the code 
