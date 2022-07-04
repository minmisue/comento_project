#!/usr/bin/env python
# coding: utf-8

# In[26]:



import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from lightgbm import LGBMClassifier

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')
train = train.drop(['index','FLAG_MOBIL'],axis = 1)
test = test.drop(['index','FLAG_MOBIL'],axis = 1)
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')


# In[2]:


display(train.head(2),train.shape,train.isna().sum(),test.head(2),test.shape,test.isna().sum())


# In[3]:


train['credit_category'] = train['credit'].replace([0.0,1.0,2.0],['credit_high:0','credit_mid:1','credit_low:2'])


# In[4]:


#가정: 차랑 부동산이 있으면 신용도가 높을것이다(재산)
fig=plt.figure(figsize=(10,5))
ax = fig.subplots(1,2)

sns.countplot(data=train, x="credit_category",hue='car',palette="pink",ax = ax[0])
sns.countplot(data=train, x="credit_category",hue='reality',palette="pink",ax = ax[1])

fig.tight_layout()

#차를 가진사람보다 안가진 사람이 모든 등급에서 비율이 높았다 가정안맞음
#부동산은 가진사람이 안가진 사람에 비해 모든 등급에서 비율이 높았음 가정맞음

#연간소득이 높을수록 신용등급이 올라가지는 않는다

display('연간소득과 신용등급 상관계수: '+str(train['income_total'].corr(train['credit'])))


# In[5]:


#가족 규모가 클수록 신용이 낮을것이다 사람이 많으면 많을수록 돈이 많이 들기때문에
display(train['family_size'].value_counts()) #7부터 이상치로 판단 6으로 바꿔줌
train['family_size'] = train['family_size'].apply(lambda x: 6.0 if x >6.0 else x)
display(train['family_size'].value_counts())
display('가족규모와 신용등급 상관계수: '+str(train['family_size'].corr(train['credit'])))
#상관없음


# In[6]:


#가정 업무시작일이 별로 안된사람(기간제 ) or 고용되지 않은 사람의 신용이 낮을것이다
#기간제 기준 
#①사용자는 2년을 초과하지 아니하는 범위 안에서(기간제 근로계약의 반복갱신 등의 경우에는 그 계속근로한 총기간이 2년을 초과하지 아니하는 범위 안에서) 기간제근로자를 사용할 수 있다. 
train['year_DAYS_EMPLOYED'] = train['DAYS_EMPLOYED']//365
DAYS_EMPLOYED_credit=(train['year_DAYS_EMPLOYED']>=-2)|(train['year_DAYS_EMPLOYED']>0)
df = train[DAYS_EMPLOYED_credit]
df
display('DAYS_EMPLOYED_credit과 신용등급 상관계수: '+str(df['year_DAYS_EMPLOYED'].corr(df['credit'])))

#관계없음


# In[7]:


train.head()
sns.violinplot(x="income_type", y="income_total", data=train)
plt.show()


# In[27]:


for df in [train,test]:
    # 연간소득변수로 월 소득 구하기
    df['income_total_m'] = (df['income_total']/12)
    
    #occyp_type 변수 결측값 처리 
    #occyp_type변수의 일하지 않거나(train['DAYS_EMPLOYED']>0) 결측값인 값들에 No_job 을 넣어줌.
    #일을 하는데(train['DAYS_EMPLOYED']<0) 결측값으로 되있는 것들은 최빈값인 Laborers를 넣어줌
    df['occyp_type'] = df['occyp_type'].fillna('Nan')
    df.loc[(train['DAYS_EMPLOYED']>0) & (df['occyp_type']=='Nan'),'occyp_type'] = 'No_job'
    df.loc[(train['DAYS_EMPLOYED']<0) & (df['occyp_type']=='Nan'),'occyp_type'] = 'Laborers'# 최빈값
    df.loc[train['DAYS_EMPLOYED']>=0,'DAYS_EMPLOYED'] = 0 # 일을하지 않는 사람은 0으로 바꿔줌
    df['DAYS_EMPLOYED_log'] = abs(df['DAYS_EMPLOYED']).apply(np.log1p) # 로그변환 0 이 있기때문에 np.log1p를 해줌 x 가 0을 가지면 무한대로 변환되어서 값이 무한대를 가짐
    
    #아까 시각화할때 가족규모가 7이상인 가족은 이상치로 판별하여 6으로 넣어줌
    df.loc[(train['family_size']>6),'family_size'] = 6
    
    #카드 발급월 양수변환
    df['begin_month'] = abs(df['begin_month'])
    
    # 몇년일했는지 파생변수 생성
    df['year_DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']//365)
    
    #로그변환
    df['income_total'] = df['income_total'].apply(np.log1p)
    
    #나이관련 변수 생성 (연,월,주로 나눌수있는 수치형 변수가 있으면 나눠서 파생변수 생성하는 것이 대체로 좋음)
    df['Age'] = abs(df['DAYS_BIRTH']//365)
    df['DAYS_BIRTH_m'] = np.floor(abs(df['DAYS_BIRTH']) / 30) - ((np.floor(abs(df['DAYS_BIRTH']) / 30) / 12).astype(int) * 12)
    df['DAYS_BIRTH_w'] = np.floor(abs(df['DAYS_BIRTH']) / 7) - ((np.floor(abs(df['DAYS_BIRTH']) / 7) / 4).astype(int) * 4)
    
    
    
    # 신청자를 구분시켜줄 정보가 부족하여 job변수에서는 직업유형 소득분류 연간소득을 이용하여 변수 생성
    df['job'] = df['occyp_type'].astype(str) + '_' + df['income_type'].astype(str)+'_'+df['income_total'].astype(str)
    
    # 신청자의 개인정보를 이용하여 파생변수 생성
    df['Privacy'] = df['gender'].astype(str) + '_' + df['car'].astype(str)+'_'+df['reality']+'_'+df['Age'].astype(str)+'_'+df['DAYS_BIRTH_m'].astype(str)+'_'+df['DAYS_BIRTH_w'].astype(str)
    


# In[24]:


#train.isna().sum()
test.isna().sum()


# In[28]:


train = train.drop(['DAYS_EMPLOYED','DAYS_BIRTH'],axis=1)
test = test.drop(['DAYS_EMPLOYED','DAYS_BIRTH'],axis=1)


# In[29]:


train_dum  = train.select_dtypes(include = 'object').columns.to_list()
train_dum


# In[31]:


for col in list(train_dum):
    le = LabelEncoder()
    train[col]=le.fit_transform(train[col])
    test[col]=le.fit_transform(test[col])


# In[32]:


train_x=train.drop('credit', axis=1)
train_y=train[['credit']]
train_x.head()


# In[33]:


X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, 
                                                    stratify=train_y, test_size=0.3,
                                                    random_state = 2022)

print("Train set: ")
print(X_train.shape)
print(y_train.shape)
print("===========")
print("Validation set: ")
print(X_val.shape)
print(y_val.shape)


clf=RandomForestClassifier()
clf.fit(X_train, y_train)
train_y_pred = clf.predict(X_train)
y_proba=clf.predict_proba(X_val)
y_pred = clf.predict(X_val)
print(f"log_loss: {log_loss(to_categorical(y_val['credit']), y_proba)}")


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from lightgbm import LGBMClassifier

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, 
                                                    stratify=train_y, test_size=0.3,
                                                    random_state = 2022)

print("Train set: ")
print(X_train.shape)
print(y_train.shape)
print("===========")
print("Validation set: ")
print(X_val.shape)
print(y_val.shape)


clf=LGBMClassifier(learning_rate=0.15,scale_pos_weight=5,num_leaves=64
                   , objective = 'multiclass')
clf.fit(X_train, y_train)
train_y_pred = clf.predict(X_train)
y_proba=clf.predict_proba(X_val)
y_pred = clf.predict(X_val)
print(f"log_loss: {log_loss(to_categorical(y_val['credit']), y_proba)}")


# In[35]:


from lightgbm import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(clf, ax=ax)

