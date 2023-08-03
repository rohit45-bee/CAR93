#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Read data 

# In[2]:


train=pd.read_csv("C:/Users/barsh/Downloads/train.csv")
test=pd.read_csv("C:/Users/barsh/Downloads/test.csv")


# In[3]:


train.nunique()


# In[4]:


train=train.drop(labels=['id','Model','Make'],axis=1)


# # Missing data treatment

# In[5]:


train.isna().sum()


# In[6]:


for i in train.columns:
    
    if train[i].dtypes=="object":
        t=train[i].mode()[0]
        train[i]=train[i].fillna(t)
    else: 
        t=train[i].mean()
        train[i]=train[i].fillna(t)
    
        


# In[7]:


train.isna().sum()


# In[8]:


cat=[]
con=[]
for i in train.columns:
    if train[i].dtypes=="object":
        cat.append(i)
        
    else:
        con.append(i)


# # Outliers

# In[9]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X1=pd.DataFrame(ss.fit_transform(train[con]),columns=con)


# In[10]:


outliers=[]
for i in con:
    outliers.extend(X1[(X1[i]<-3)|(X1[i]>3)].index)


# In[11]:


outliers


# In[12]:


from numpy import unique
out=unique(outliers)


# In[13]:


train=train.drop(index=out,axis=0)
train.shape


# In[14]:


train.index=range(0,64)


# In[15]:


train.shape


# # EDA

# In[16]:


Y=train[["Weight"]]
X=train.drop(labels="Weight",axis=1)


# In[17]:


cat=[]
con=[]
for i in X.columns:
    if train[i].dtypes=="object":
        cat.append(i)
        
    else:
        con.append(i)


# In[18]:


import seaborn as sb
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
plt.figure(figsize=(12,12))
x=1
for i in X.columns:
    if X[i].dtypes=="object":
        plt.subplot(6,5,x)
        sb.boxplot(X[i],Y["Weight"])
        x=x+1
    else:
        plt.subplot(6,5,x)
        sb.scatterplot(X[i],Y["Weight"])
        x=x+1
    


# # Preprocessing

# In[19]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X1=pd.DataFrame(ss.fit_transform(train[con]),columns=con)


# In[20]:


X2=pd.get_dummies(train[cat])


# In[21]:


Xnew=X1.join(X2)
Xnew.head()


# # model1(All )

# In[22]:


Xnew=Xnew
Y=train[["Weight"]]


# In[23]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)


# In[24]:


from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[25]:


ol.rsquared_adj


# In[26]:


ol.pvalues.sort_values()


# In[27]:


col_to_del=ol.pvalues.sort_values().index[-1]


# In[28]:


Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # MOdel2

# In[29]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[30]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model3

# In[31]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[32]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model4

# In[33]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[34]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model5

# In[35]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[36]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model6

# In[37]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[38]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model7

# In[39]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[40]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model8

# In[41]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[42]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model9

# In[43]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[44]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model10

# In[45]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[46]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model11

# In[47]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[48]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model12

# In[49]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[50]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model13

# In[51]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[52]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model14

# In[53]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[54]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model15

# In[55]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[56]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model16

# In[57]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[58]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model17

# In[59]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[60]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model18

# In[61]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[62]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model19

# In[63]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[64]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model20

# In[65]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[66]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model21

# In[67]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[68]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model22

# In[69]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[70]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # MOdel21

# In[71]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[72]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model22

# In[73]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[74]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model23

# In[75]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[76]:


len(Xnew.columns)


# In[77]:


Xnew.columns


# In[78]:


col_to_del=ol.pvalues.sort_values().index[-1]
Xnew=Xnew.drop(labels=col_to_del,axis=1)


# # Model24

# In[79]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)

from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# # Testing data set

# In[80]:


Xnew.columns


# In[81]:


test.head()


# In[82]:


test.nunique()


# In[83]:


test=test.drop(labels=['id',"Model","Make"],axis=1)


# In[84]:


test.isna().sum()


# In[85]:


test["Luggage.room"].dtype


# In[86]:


t=test["Luggage.room"].mean()
test["Luggage.room"]=test["Luggage.room"].fillna(t)


# In[87]:


test.isna().sum()


# # outliers

# In[88]:


cat=[]
con=[]
for i in test.columns:
    if train[i].dtypes=="object":
        cat.append(i)
        
    else:
        con.append(i)


# In[92]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

Y1['Manufacturer']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['Type']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['AirBags']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['DriveTrain']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['Cylinders']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['Man.trans.avail']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['Origin']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1['Weight']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Y1 = pd.DataFrame(ss.fit_transform(test[con]),columns=con)
Y1.shape


# In[93]:


Y1=pd.DataFrame(ss.fit_transform(test[con]),columns=con)


# In[94]:


Y1


# In[95]:


Y1[(Y1['Price']>3) |(Y1['Price']<-3)]


# In[96]:


outliers=[]
for i in Y1.columns:
    outliers.extend(Y1[(Y1[i]>3) | (Y1[i]<-3)].index)


# In[97]:


outliers


# NO outliers

# In[98]:


Y2=pd.get_dummies(test[cat])


# In[99]:


Xnew=Y1.join(Y2)


# In[ ]:


Y=Xnew[["Weight"]]
X=Xnew.drop(labels=["Weight"],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=41)


# In[ ]:



from statsmodels.api import add_constant
xconst=add_constant(xtrain)


# In[ ]:



from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[ ]:





# In[ ]:




