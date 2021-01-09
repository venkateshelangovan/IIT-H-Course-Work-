
"""
Authors: Arkadipta De (ai20mtech14002)
         Venkatesh E  (ai20mtech14005)

# **FML HACKATHON :**

## **OVERVIEW OF THE DATASET :**

### **Importing the Libraries :**
"""

# importing the libraries
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings("ignore")

'''
Import this library only if running file in Google Colab
'''
#from google.colab import files

random_state = 100

# to display entire rows and columns of dataframe 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

"""### **Reading the Data :**"""

'''
Put train_input.csv and test_input.csv in Current Working Directory 
'''
path = os.getcwd()

# read the train data
train_df = pd.read_csv(path + '/' + "train_input.csv")
# read the test data 
test_df = pd.read_csv(path + '/' + "test_input.csv")

# storing the original data 
train_org=train_df.copy()
test_org=test_df.copy()

# sample of train data
train_df.head(10)

# shape of datasets 
print("\nTrain dataset shape : ",train_df.shape)
print("\nTest dataset shape : ",test_df.shape)

# info about the dataset 
#train_df.info(verbose=True)

"""**NOTE :** 

1. Here there were 24 input features among which 13 features were of datatype int and remaining 11 were of type float. Target variable is a Discrete feature.

2. Feature (9-18) and feature 24 has missing values and it is needed to be taken care of.
"""

# statistics about the dataset
#train_df.describe(percentiles=[0.01,0.1,0.9,0.95,0.99])

# target variable 
#print(train_df['Target Variable (Discrete)'].value_counts())
#train_df['Target Variable (Discrete)'].value_counts().plot(kind='barh',ylabel="Output Categories")

'''

"""**Note :**

1. Class 1,0,2,6,5 have some good number of occurences.

2. Remaining classes 3,4,7,8,9,10,11,12,13,14,15,16,17 have very few occurences.

## **DATA VISUALIZATION :**

### **Correlation Plot -- HEATMAP:**
"""

plt.figure(figsize=(20,10))
sns.heatmap(train_df.drop(columns='Target Variable (Discrete)',axis=1).corr(),annot=True)

"""**Note :** 

1. Here the correlation between feature 1 and feature 23 is 1. So having both the feature is not so useful for further analysis and anyone could be removed.

2. Similarly Correlation between features(11,14) and feature(15,16) also very high and so anyone could be removed from both sets as having both will lead to redundant feature.
"""

# get the columns from train dataset 
cols_train=train_df.columns.tolist()
# removing the target varibale 
cols_train.remove('Target Variable (Discrete)')
# getting the discrete columns 
discrete_cols=[x for x in cols_train if "Discrete" in x]
# getting the continuous columns 
continuous_cols=[x for x in cols_train if "Discrete" not in x]


print("Number of columns : ",len(cols_train))
print("Number of Discrete Columns : ",len(discrete_cols))
print("Number of Continuous Columns : ",len(continuous_cols))

def discrete_function(cols):
  """
  Input : cols - features of given input data
  returns : 1 if it is categorical feature 
            0 if it is not a categorical feature 
  """
  l=len(train_df[cols].value_counts())
  if l<=20:
    return 1
  else:
    return 0

discrete_cat=[]
discrete_cont=[]
for x in discrete_cols:
  d=discrete_function(x)
  if d==1:
    discrete_cat.append(x)
  else:
    discrete_cont.append(x)

"""**Note :**

1. The discrete features (5,6,7,8,21) were the categorical features.

2. Other discrete features(1,2,3,4,19,20,22,23) were not like categorical features.

### **Univariate and Bivariate Analysis:**
"""

def count_plot(discrete_cat):
  """
  input : discrete_cat : list containing the discrete column 
  plots : count plots of all the features given in the discrete_cat list 
  """
  fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(25,12)) # figure of 2 rows and 3 columns subplots 
  # count plots 
  s2=sns.countplot(x=discrete_cat[0],data=train_df,ax=ax1,palette='magma')
  s3=sns.countplot(x=discrete_cat[1],data=train_df,ax=ax2,palette='crest')
  s4=sns.countplot(x=discrete_cat[2],data=train_df,ax=ax3,palette='cubehelix')
  s5=sns.countplot(x=discrete_cat[3],data=train_df,ax=ax4,palette='Spectral')
  # annotating the labels 
  for p in s2.patches:
    s2.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width()/2 , p.get_height()), ha = 'center', va = 'center', xytext = (0,10),rotation=0, textcoords = 'offset points',fontsize=12)
  for p in s3.patches:
    s3.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width()/2 , p.get_height()), ha = 'center', va = 'center', xytext = (0,10),rotation=0, textcoords = 'offset points',fontsize=12)
  for p in s4.patches:
    s4.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width()/2 , p.get_height()), ha = 'center', va = 'center', xytext = (0,10),rotation=45, textcoords = 'offset points',fontsize=12)
  for p in s5.patches:
    s5.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width()/2 , p.get_height()), ha = 'center', va = 'center', xytext = (0,10),rotation=45, textcoords = 'offset points',fontsize=12)

count_plot(discrete_cat)

"""**Note :**

In most of categorical features some category entries were very less (in single digit count).

"""

def scatterplot(col1,col2,c):
  """
  input : col1 : along x axis 
          col2 : along y axis 
             c : color to plot the scatter plot
  plots : scatter plot of given two features col1 and col2
  """
  plt.figure(figsize=(10,5))
  sns.scatterplot(x=col1,y=col2,data=train_df,color=c)
color=['black','red','green','orange','blue','purple','grey']
c=0
for i in range(len(discrete_cont)-1):
  for j in range(i+1,len(discrete_cont)):
    scatterplot(discrete_cont[i],discrete_cont[j],color[c])
    c+=1
    if c>6:
      c=0

for i in range(len(continuous_cols)-1):
  for j in range(i+1,len(continuous_cols)):
    scatterplot(continuous_cols[i],continuous_cols[j],color[c])
    c+=1
    if c>6:
      c=0

"""**Note :**

1. We could not see much correlation among continuous features.

2. Features (15,16),(15,17) are correlated among all the combinations of continuous features
"""

# non_categorical cols
non_cat=continuous_cols
non_cat.extend(discrete_cont)

def histogram(col):
  """
  input : col - feature
  plots : distplot,rug plot and kde plot for given input feature(col)
  """
  plt.figure(figsize=(10,5))
  fig, ((ax1,ax2,ax3)) = plt.subplots(nrows=1,ncols=3,figsize=(20,5))
  sns.distplot(train_df[col],kde=False,rug=False,ax=ax1,color='black')
  sns.distplot(train_df[col],kde=True,rug=False,ax=ax2,color='red')
  sns.distplot(train_df[col],kde=True,rug=True,ax=ax3,color='green')

for x in non_cat:
  histogram(x)

"""**Note :**

1. Some of the features such as Feature 2,17,19 etc were Right Skewed.

2. Feature 24 has left skewed distribution.

3. Features 1,23 must be a random distribution. 

4. Some features (22,20,24) have bimodal distribution 
"""

def BoxViolinPlots(col):
  """
  input : col - feature 
  plots : boxplots and violin plots to visualize outliers 
  """
  plt.figure(figsize=(10,5))
  fig, ((ax1,ax2)) = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
  sns.boxplot(y=train_df[col],ax=ax1,color='orange')
  sns.violinplot(y=train_df[col],ax=ax2,color='red')
  ax1.set_xlabel(col)
  ax1.set_ylabel("Observed Values")
  ax2.set_xlabel(col)
  ax2.set_ylabel("Observed Values")

for x in cols_train:
  BoxViolinPlots(x)

"""**Note :**

1. Both Box Plots and Violin Plots used to visualize the outliers.

2. Features (4,7,9,12,17,20) has many outlier points and we should check its impact on models while model building process.
'''

"""
## **DATA PREPROCESSING :**

### **Remove Highly Correlated Features:**
"""

# removing highly correlated data 
def return_uncorrelated_dataset(df,test,threshold):
  """
  input : df  - train data 
          test- test data 
          threshold - maximum correlation between two features. If the correlation between two features is more than 
                      threshold, remove one feature
  return : train and test data with correlated features removed
  """
  correlation_matrix=df.corr()
  cols=df.columns.tolist()
  l=len(cols)
  threshold_limit_columns=[True]*l
  threshold_limit_columns[l-1]=False
  for i in range(l):
    for j in range(i+1,l):
      if correlation_matrix.iloc[i,j]>threshold:
        if threshold_limit_columns[j]!=False:
          threshold_limit_columns[j]=False
  uncorrelated_features=df.columns[threshold_limit_columns]
  return df[uncorrelated_features],test[uncorrelated_features]

"""### **Missing Values Handling and Imputation :**"""

# identifying the missing values 
def Missing_Value_Checker(df,threshold=0):
  """
  input : df - train data 
          threshold - default as 0 --- to display all features with missing values 
                      can be any value - to display all features with missing values above the threshold
  prints : number of features with missing values greater than the threshold 
           features along with their missing values 
  """
  #print("\n Check for NaN values in each features :\n")
  missing_val_percent=round(df.isnull().sum()/len(df)*100,2)
  missing_val_percent=missing_val_percent.sort_values(ascending=False).where(missing_val_percent>threshold)
  #print(missing_val_percent[~missing_val_percent.isnull()])
  #print("\n Number of features with missing values :",missing_val_percent[~missing_val_percent.isnull()].count())

#drop the missing value column above the threshold mentioned 
def drop_missing_values(df,threshold):
  """
  input : df- train data 
          threshold - percentage of missing values 
  returns : df with features have the missing values less than threshold
  """
  missing_val_percent=round(df.isnull().sum()/len(df)*100,2)
  missing_val_percent=missing_val_percent.sort_values(ascending=False).where(missing_val_percent>threshold)
  remove_cols=list(missing_val_percent[~missing_val_percent.isnull()].index)
  df_new=df.drop(remove_cols,axis=1)
  return df_new

# imputing the feature with median as imputing median is not affected by outliers
def imputing_missingvalues(df,test,missing_col,imputed_val):
  """
  input : df- train 
          test - test data
          missing_col - column which has missing value to be imputed 
          imputed_val - value to be imputed inplace of NA in given missing_col
  returns : df,test with missing value imputed for the column passed with given imputed_val
  """
  df[missing_col].fillna(imputed_val,inplace=True)
  test[missing_col].fillna(imputed_val,inplace=True)
  return df,test

"""### **Handling Data *Duplications*:**"""

def handling_duplicate_entries(df):
  """
  input : df-train 
  returns : df with duplicates of rows and columns will be removed
  """
  # to check if any row in the train data duplicated ------ DUPLICATE ROWS 
  #print("\nTrain data before duplicates removed : ",df.shape)
  df=df.drop_duplicates()
  # DUPLICATE COLUMNS 
  df=df.T.drop_duplicates().T
  #print("Train data after duplicates removed : ",df.shape)
  return df

"""### **Removing Outlier Points :**"""

def remove_outlier(df):
  """
  input : df-train 
  returns : df with target variable count >3 
  """
  df_new=df.groupby("Target Variable (Discrete)").filter(lambda x: len(x) >3)
  return df_new

"""### **Train Validation Split :**"""

def TrainTestData(X,y):
  """
  input : X  - input features 
          y  - output target variable 
  returns : X_train,X_val,y_train,y_val
  """
  X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.3,random_state=42)
  return X_train,X_val,y_train,y_val

"""### **Data Normalization :**"""

def NormalizeData(train,val,test,normalizer):
  """
  input : train , val ,test ----> train , validation and test data 
          normalizer : 1    MinMaxScaler
                      -1    No normalization 
                       else StandardScaler
  returns : train,validation and test data normalized versions
  """
  if normalizer==1:
    normalizer=MinMaxScaler()
  elif normalizer==-1:
    return train,val,test
  else:
    normalizer=StandardScaler()
  train=normalizer.fit_transform(train)
  val=normalizer.transform(val)
  test=normalizer.transform(test)
  return train,val,test

"""### **Download the Dataset :**

Download the data to be given to input model if the modelling is done in seperate python notebook. Otherwise we could omit the download section.
"""
'''
def DownloadData(data,filename):
  """
  input : data - data to be downloaded
         filename- filename to be downloaded in local storage 
  Downloads : data in local storage
  """
  np.save(filename,data)
  from google.colab import files
  files.download(filename)

def DownloadAll(train,trainname,val,valname,test,testname,ytrain,ytrainname,yval,yvalname):
  """
  input : train,val,test,ytrain,yval - data to be downloaded
         trainname,valname,testname,ytrainname,yvalname- filename to be downloaded in local storage 
  Downloads : data in local storage
  """
  DownloadData(train,trainname)
  DownloadData(val,valname)
  DownloadData(test,testname)
  DownloadData(ytrain,ytrainname)
  DownloadData(yval,yvalname)
  print("DATASETS DOWNLOADED")
'''

"""## **CREATING DATASET :**"""

def getData(train,test,correlation_threshold,missing_value_threshold,imputation_method,normalize,outlier=1):
  """
  input    :     train --- train data 
                 test ---- test data 
                 correlation_threshold - value above which correlation between two features is removed
                 missing_value_threshold -feature having missing value above threshold will be removed
                 imputation_method : mean or median or any imputation technique 
                 normalize : 1 - MinMax 
                             else -StandardScaler 
                 Outlier : default 1 -remove outlier 
                            else - dont remove outlier
  returns   :    X_train,X_val,X_test,y_train,y_val
  """
  output_data=train["Target Variable (Discrete)"]
  train,test=return_uncorrelated_dataset(train,test,correlation_threshold)
  train["Target Variable (Discrete)"]=output_data
  Missing_Value_Checker(train)
  train=drop_missing_values(train,missing_value_threshold)
  test=test[train.drop("Target Variable (Discrete)",axis=1).columns]
  missing_cols=train.columns[train.isnull().any()]
  for cols in missing_cols:
    if imputation_method=="mean":
      imputed_val=train[cols].mean()
    elif imputation_method=="median":
      imputed_val=train[cols].median()
    train,test=imputing_missingvalues(train,test,cols,imputed_val)
  train=handling_duplicate_entries(train)
  if outlier==1:
    train=remove_outlier(train)
  y=train["Target Variable (Discrete)"]
  X=train.drop("Target Variable (Discrete)",axis=1)
  X_train,X_val,y_train,y_val=TrainTestData(X,y)
  X_train,X_val,X_test=NormalizeData(X_train,X_val,test,normalize)
  return X_train,X_val,X_test,y_train,y_val

"""
Strategy of Preprocessing Used:
             remove features which are 95% correlated 
             remove feature which has missing values more than 50%
             impute missing values using mean 
             remove outlier in target variable
             MinMaxScaler
"""             
X_train,X_val,X_test,y_train,y_val = getData(train_df,test_df,0.95,50,"mean",1)

"""## **MODELLING :**

#### **Machine Learning Models -- Hyperparameter Tuning :**
"""
'''
def BestParams_GridSearchCV(algo,X,y,hyperparams,folds):
  """
  input   : algo : classification algorithm 
             X : X_train 
             y : y_train 
            param_grid: hyperparameters for the ML algorithm passed
  prints  : best hyperparameters 
  """
  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = algo, param_grid =hyperparams, 
                          cv =folds,scoring="accuracy",n_jobs = -1,verbose = 1,return_train_score=True)
  # Fit the grid search to the data
  grid_search.fit(X, y)
  results=pd.DataFrame(grid_search.cv_results_)
  return results 

def HyperParamsResultsPlot(results,modelname):
  """
  input : results_acc : grid search results got using accuracy as performance measure 
          modelname   : Machine Learning Model used 
  prints : best hyperparameters 
  plots  : hyperparameters plots for given model 
  """
  print("---------------------------------------------------------------------------------------------------------")
  print(modelname)
  print("---------------------------------------------------------------------------------------------------------")
  params=results.sort_values("mean_test_score",ascending=True)
  params['ID']=[i for i in range(1,len(params['mean_test_score'])+1)]
  plt.figure(figsize=(8,5))
  fig, ((ax1)) = plt.subplots(nrows=1,ncols=1,figsize=(20,5))
  ax1.plot(params['ID'],params['mean_test_score'],color='green')
  ax1.plot(params['ID'],params['mean_train_score'],color='red')
  ax1.set_title("Accuracy score using Hyperparameters of "+modelname)
  ax1.set_xlabel("Hyperparameter ID")
  ax1.set_ylabel("Accuracy score")
  best_acc_params=params['params'].iloc[-1]
  print("Best hyparameters using accuracy as performance measure : ",best_acc_params)
  return best_acc_params
'''

def main_results(best_m,train,val,test,ytrain,yval):
  """
  inputs : best_m - best classifier (may be best random forest model,xgboost model etc)
           train,val,test,ytrain,yval- data after preprocessesing 
  prints : validation accuracy 
  returns: test predictions 
  """
  best_m.fit(train,ytrain)
  y_pred = pd.Series(best_m.predict(val))
  print("Validation Accuracy : ",accuracy_score(yval,y_pred))
  test_pred=pd.Series(best_m.predict(test))
  return test_pred 

def submission(ytest,download=0,main_submission=0):
  """
  inputs : ytest - ytest predicted data 
           download - default 0 - don't download the submission file (.csv) for kaggle (just for visualization purpose)
                      else - download the submission file (.csv) for kaggle 
           main_submission -  default 0 - we don't consider that submission as final submission
                              otherwise - take as main submission file and return the submission file 
  returns : predicted target variable count of each classes
  """
  submission_file=pd.DataFrame()
  submission_file['Id']=range(1,len(X_test)+1)
  submission_file['Category']=ytest 
  submission_file['Category']=submission_file['Category'].astype(int)
  submission_file.to_csv("test_output.csv",index=False)   # test_output.csv is stored in Current Working Directory
  if download!=0:
    files.download('test_output.csv')
  if main_submission!=0:
    return submission_file
  #print("Test Predicted Category counts : ")
  #return submission_file['Category'].value_counts()

"""##### **RANDOM FOREST MODEL:**"""

folds = KFold(n_splits = 5, shuffle = True, random_state = random_state)

'''
rf=RandomForestClassifier(random_state=random_state)
hyper_params_rf={
    'max_depth': [8,10,12],
    'min_samples_leaf':[1,2,3,4],
    'min_samples_split': [5,10,20],
    'n_estimators': [200,300], 
    'max_features': [15,20],
     'class_weight':['balanced']
}

best_acc_params_rf=HyperParamsResultsPlot(BestParams_GridSearchCV(rf,X_train,y_train,hyper_params_rf,folds),"RANDOM FOREST")

"""##### **XGBoost :**"""

xg=XGBClassifier(random_state=random_state)
hyper_params_xg={
    'max_depth': [3,8,10],
    'min_child_weight':[1,3,5],
    'min_samples_split': [5,10],
    'n_estimators': [200,300], 
    'max_features': [10,15,20],
     'class_weight':['balanced']
}

best_acc_params_xg=HyperParamsResultsPlot(BestParams_GridSearchCV(xg,X_train,y_train,hyper_params_xg,folds),"XGBoost")

"""##### **GBDT :**"""

gbdt=GradientBoostingClassifier(random_state=random_state)
hyper_params_gbdt={
    'max_depth': [3,8,10],
    'min_samples_leaf':[1,3,5],
    'min_samples_split': [5,10],
    'n_estimators': [200,300], 
    'max_features': [10,15,20]
    }

best_acc_params_gbdt=HyperParamsResultsPlot(BestParams_GridSearchCV(gbdt,X_train,y_train,hyper_params_gbdt,folds),"GBDT")

"""##### **SVM :**"""

svm=SVC(kernel='rbf',random_state=random_state)
hyper_params_svm={
    'C':[0.01,0.1,0.5,1,10,100,1000],
    'gamma':['auto','scale'],
    'class_weight':['balanced']
}

best_acc_params_svm=HyperParamsResultsPlot(BestParams_GridSearchCV(svm,X_train,y_train,hyper_params_svm,folds),"SVM")

"""##### **MULTILAYER PERCEPTRON :**"""

mlp=MLPClassifier()
hyper_params_mlp={
    'hidden_layer_sizes':[(100,),(200,),(300,)],
    'activation':['tanh','logistic','relu'],
    'alpha':[0.0001,0.001,0.01]
}

best_acc_params_mlp=HyperParamsResultsPlot(BestParams_GridSearchCV(mlp,X_train,y_train,hyper_params_mlp,folds),"Multilayer Perceptron")
'''

"""#### **Best Models :**

##### **Best Random Forest Model :**
"""

print('Running Best Candidate Models after Hyperparameter Tunning')

# best rf model 
best_rf=RandomForestClassifier(random_state=random_state,max_depth=12,max_features=15,min_samples_leaf=1,min_samples_split=5,n_estimators=200)
#rf_test_pred=main_results(best_rf,X_train,X_val,X_test,y_train,y_val)
#submission(rf_test_pred)

"""##### **Best XGBoost Model :**"""

# best xgboost model 
best_xg=XGBClassifier(class_weight='balanced',max_depth=8,max_features=10,min_child_weight=1,min_samples_split=5,n_estimators=300,random_state=random_state)
#xg_test_pred=main_results(best_xg,X_train,X_val,X_test,y_train,y_val)
#submission(xg_test_pred)

"""##### **Best GBDT Model :**"""

# best gbdt model 
best_gbdt=GradientBoostingClassifier(max_depth=10,max_features=15,min_samples_leaf=1,min_samples_split=10,n_estimators=200,random_state=random_state)
#gbdt_test_pred=main_results(best_gbdt,X_train,X_val,X_test,y_train,y_val)
#submission(gbdt_test_pred)

"""##### **Best SVM Model :**"""

# best svm model 
best_svm=SVC(C=1000,class_weight='balanced',gamma='scale',probability=True,random_state=random_state)
#svm_test_pred=main_results(best_svm,X_train,X_val,X_test,y_train,y_val)
#submission(svm_test_pred)

"""##### **Best MLP Model :**"""

# best mlp model 
best_mlp=MLPClassifier(activation='relu',alpha=0.01,hidden_layer_sizes=(300,),random_state=random_state)
#mlp_test_pred=main_results(best_mlp,X_train,X_val,X_test,y_train,y_val)
#submission(mlp_test_pred)

"""##### **Voting Classifier (Type - HARD) :**"""
'''
# hard voting classifier 
vcl_hard=VotingClassifier(estimators=[('rf',best_rf),('xgb',best_xg),('gbdt',best_gbdt)],voting='hard')
vcl_hard_test_pred=main_results(vcl_hard,X_train,X_val,X_test,y_train,y_val)
submission(vcl_hard_test_pred)

"""##### **Voting Classifier (Type - SOFT) :**"""

# soft voting classifier 
vcl_soft=VotingClassifier(estimators=[('rf',best_rf),('xgb',best_xg),('gbdt',best_gbdt)],voting='soft',weights=[0.35,0.35,0.3])
vcl_soft_test_pred=main_results(vcl_soft,X_train,X_val,X_test,y_train,y_val)
submission(vcl_soft_test_pred)
'''

"""## **RESULTS :**

### **Final Model :**
"""

print('Running Soft Voting Classifier')
# soft voting classifier 
final_clf=VotingClassifier(estimators=[('rf',best_rf),('xgb',best_xg),('gbdt',best_gbdt)],voting='soft',weights=[0.35,0.35,0.3])
test_pred=main_results(final_clf,X_train,X_val,X_test,y_train,y_val)

"""### **Submission File Generation :**"""

# downloading the main submission file for kaggle
# submission(test_pred,1)

# main kaggle submission file 
'''
Final Submission File Generation
'''
submission(test_pred,0,1)
#final_submission

print('test_output.csv has been created and located in the Current Working Directory')