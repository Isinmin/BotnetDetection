import pandas as pd
import numpy as npfrom sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_csv('Datasets/trainset.csv')
df.head()
df.info()
df.drop(['Unnamed: 0','srcip', 'srcport', 'dstip', 'dstport', 'proto', 'std_active', 'min_idle', 
        'mean_idle', 'max_idle', 'std_idle','furg_cnt', 'burg_cnt','sflow_bpackets','sflow_bbytes',
        'sflow_fpackets','sflow_fbytes','dscp'],axis=1,inplace=True)
        
df.columns
len(df.columns)
from sklearn.model_selection import train_test_split
X = df.drop(['label'],axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


from sklearn.lineral_model import LinearRegression
from sklearn.metrics import accuracy_score

linearReg = LinearRegression()
linear

import itertools

feat_sets = []
for T in range(4, 41):
    for subset in itertools.combinations(names_columns, T):
        array_names = [name for name in subset]
        feat_sets.append(array_names)
        
len(feat_sets)
from time import time
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_predict
classifiers = [DecisionTreeClassifier(),MultinomialNB(),GaussianNB(),BernoulliNB(),LinearSVC(),AdaBoostClassifier(),RandomForestClassifier()]
names = ['Decision Tree','Multinominal NB','Gaussian NB','Bernoulli NB','Linear SVC','Adaboost','Random Forest Classifier']

results = []

for i,subset in enumerate(feat_sets):
    print('Running subset {}'.format(str(subset)))
    X = df[subset]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
    
    for j, c in enumerate(classifiers):
        start = time()
        print('Classifier {}'.format(names[j]))
        predictions = c.fit(X_train,y_train).predict(X_test)
        precision,recall,fscore,support = precision_recall_fscore_support(y_test,predictions)
        acc = accuracy_score(y_test,predictions)
        print('Accuracy: {}'.format(acc))
        c_matrix = confusion_matrix(y_test,predictions)
        results.append([i,names[j],acc,precision,recall,fscore,c_matrix,(time() - start)])
 results_df = pd.DataFrame(results,columns = ['Subset','Classifier','Accuracy','Precision','Recall','F-score','Confusion Matrix','Runtime'])
 rfstats = results_df[results_df['Classifier'] == 'Random Forest Classifier']
adastats = results_df[results_df['Classifier'] == 'Adaboost']
dtstats = results_df[results_df['Classifier'] == 'Decision Tree']
mnbstats = results_df[results_df['Classifier'] == 'Multinominal NB']
gnbstats = results_df[results_df['Classifier'] == 'Gaussian NB']
bnbstats = results_df[results_df['Classifier'] == 'Bernoulli NB']
svcstats = results_df[results_df['Classifier'] == 'Linear SVC']

dtstats[dtstats['Accuracy'] == dtstats['Accuracy'].max()]
feat_sets[182]
rfstats[rfstats['Accuracy'] == rfstats['Accuracy'].max()]
feat_sets[855]
adastats[adastats['Accuracy'] == adastats['Accuracy'].max()]
feat_sets[3471]
mnbstats[mnbstats['Accuracy'] == mnbstats['Accuracy'].max()]
feat_sets[77]
gnbstats[gnbstats['Accuracy'] == gnbstats['Accuracy'].max()]
feat_sets[320]
bnbstats[bnbstats['Accuracy'] == bnbstats['Accuracy'].max()]
feat_sets[3779]
svcstats[svcstats['Accuracy'] == svcstats['Accuracy'].max()]
feat_sets[686]
results_df.to_csv('bruteforce_results.csv')
import pandas as pd
from pandas import DataFrame
classifiers = [DecisionTreeClassifier(),MultinomialNB(),BernoulliNB(),LinearSVC(),AdaBoostClassifier(),RandomForestClassifier()]
names = ['Decision Tree','Multinominal NB','Bernoulli NB','Linear SVC','Adaboost','Random Forest Classifier']

for i,clf in enumerate(classifiers):
    
    print('Testing classifier {}'.format(names[i]))
    
    rfecv = RFECV(clf,verbose=3)
    
    rfecv.fit(X_train,y_train)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    rankings = list(rfecv.ranking_)
    best = []
    for index,content in enumerate(rankings):
        if(content == 1):
            best.append(all_cols[index])
            
    print('Optimal subset: ' + str(best))
    
    rf = clf

    Xaux_train = X_train[best]
    Xaux_test = X_test[best]

    rf.fit(Xaux_train,y_train)
    pred = rf.predict(Xaux_test)
    print(accuracy_score(y_test,pred))
    all_cols = ['min_fpktl', 'mean_fiat', 'max_fiat', 'std_fiat',
            'mean_biat', 'max_biat', 'std_biat', 'bpp', 'avg_iat', 'pct_packets_pushed',
            'avg_payload_length', 'iopr']
X = df[all_cols]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100]}

grid = GridSearchCV(SVC(),param_grid,verbose=3,n_jobs=2)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_

grid.best_params_
pred = grid.predict(X_test)
accuracy_score(y_test,pred)
pred = grid.predict(X_test)
accuracy_score(y_test,pred)grid.best_estimator_
pred = grid.predict(X_test)
accuracy_score(y_test,pred)
param_grid = {'C':[0.1,1,10,100],'gamma':[0.1,0.01,0.001,0.0001],'degree':[0,1]}

grid_poly = GridSearchCV(SVC(kernel='poly'),param_grid,verbose=3,n_jobs=2)
grid_poly.fit(X_train,y_train)
grid_poly.best_params_

param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001,0.0001],'coef0':[-1,0,1]}

grid_sig = GridSearchCV(SVC(kernel='sigmoid'),param_grid,verbose=3,n_jobs=2)
grid_sig.fit(X_train,y_train)
grid_sig.best_params_
pred = grid_sig.predict(X_test)
accuracy_score(y_test,pred)
from sklearn.lineral_model import LinearRegression
from sklearn.metrics import accuracy_score

linearReg = LinearRegression() # создание модели
linearReg.fit(data_train, values_train) # применение модели к набору данных для обучения
predictions = linearReg.predict(data_test) # получение прогнозов на основе обученной модели

accuracy_score(predictions, values_test) # получение значения точности обучаемой модели
