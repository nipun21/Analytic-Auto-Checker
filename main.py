# Import All required Libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from warnings import simplefilter
# Read Dataset via pandas 
fpdf= pd.read_csv('diabetes.csv')
# Create Dataset copy
frm=fpdf.copy()
dbaray=frm.values
ln=len(frm.columns)
# Range of Dataset
rng=ln-2
# Target of Dataset
trgt=ln-1
final=[]
# for loop runs on all features of DataFrame
for i in range(trgt):
    # Newfrm is list of all combinations of features. ** Always it removes 1 feature from DataFrame **
    newfrm=[]
    if(i>0):
        for x in range(0,i):
            newfrm.append(frm.columns[x])
        if(i==(trgt-1)):
            for y in range(i,trgt):
                newfrm.append(frm.columns[y])
        else:
            for y in range(i+1,trgt):
                newfrm.append(frm.columns[y])
    elif(i==0):
        for z in range(1,trgt):
            newfrm.append(frm.columns[z])
            
    # Create New Dynamic DataFrame from "newfrm" list
    nwfm=frm[newfrm]
    # Get values from New DataFrame
    dbaraynw=nwfm.values
    lns=len(nwfm.columns)
    rngs=lns-2
    trgts=lns-1
    # X axis data for train and validate
    dbrxnw=dbaraynw[:,:rngs]
    # Y axis data for train and validate
    dbrynw=dbaraynw[:,trgts]
    valid_size=0.2
    seed=rngs
    scores="accuracy"
    # Out is list to store all possible results from multiple algorithms
    out=[]
    # Train and validate from new DataFrame
    x_train, x_validate, y_train, y_validate = model_selection.train_test_split(dbrxnw,dbrynw,test_size=valid_size, random_state=seed)
    nwmodel=[]
    # Apply Train and Validate data on multiple Algorithm
    nwmodel.append(("D-Tree",DecisionTreeClassifier()))
    nwmodel.append(("Naive_Base",GaussianNB()))
    nwmodel.append(("KNN",KNeighborsClassifier()))
    nwmodel.append(("LRegression",LogisticRegression()))
    #nwmodel.append(("LDA",LinearDiscriminantAnalysis()))
    nwmodel.append(("SVM", SVC()))
    names=[]
    result=[]
    simplefilter(action='ignore', category='FutureWarning')
    # Use KFold 
    for name, model in nwmodel:
        kfold=model_selection.KFold(n_splits=5, random_state=5)
        cv_result=model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scores)
        result.append(cv_result)
        names.append(name)
        msg="%s: %f (%f)" %(name, cv_result.mean(), cv_result.std())
        # Store all possible results in "out" list
        out.append((name, cv_result.mean()))
    final.append(out)
# Create multiple list as per algorithms to store possible results
DTree=[]
NBase=[]
KNN=[]
LRegression=[]
LDA=[]
SVM=[]
# For loop to get all results and append them in above lists
for m in range(0,len(final)):
    for n in range(0,5):
        if((final[m][n][0])=='D-Tree'):
            DTree.append(final[m][0][1])
        elif((final[m][n][0])=='Naive_Base'):
            NBase.append(final[m][0][1])
        elif((final[m][n][0])=='KNN'):
            KNN.append(final[m][n][1])
        elif((final[m][n][0])=='LRegression'):
            LRegression.append(final[m][n][1])
        elif((final[m][n][0])=='LDA'):
            LDA.append(final[m][n][1])
        elif((final[m][n][0])=='SVM'):
            SVM.append(final[m][n][1])
# Create New DataFrame from all possible reults to check which feature is useful in our dataset
df=pd.DataFrame(list(zip(DTree,NBase,KNN,LRegression,SVM)),columns=['D-Tree','Naive_Base','KNN','LRegression','SVM'])
# Print New data set to evaluate role of every feature in our dataset
print(df)
