# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

data=pd.read_csv('data.csv',error_bad_lines=False)
# %%
data.head()

# %%
data.shape
# %%
data.info()
# %%
data.describe()
# %%
data['strength'].value_counts()
# %%
data = data.dropna(axis=0)
# %%
data.shape
# %%
sns.countplot(x=data['strength'])
# %%
data['password'] = data['password'].astype('str')
# %%
data.info()

# This parts are for test parts that the datas are working properly.
# 위 파트들은 데이터셋이 제대로 작동하는지 테스트하는 부분들입니다.
# %%
def cal_len(x):
    x=str(x)
    return len(x)

def capL(x):
    x=str(x)
    cnt=0
    for i in x:
        if(i.isupper()):
            cnt+=1
    return cnt

def smL(x):
    x=str(x)
    cnt=0
    for i in x:
        if(i.islower()):
            cnt+=1
    return cnt
def spc(x):
    x=str(x)
    x=re.findall('[\W]',x)
    cnt=0
    for i in x:
        if(i != None):
            cnt+=1
    return cnt

def alp(x):
    x=str(x)
    cnt=0
    for i in x:
        if(i.isalpha()):
            cnt+=1
    return cnt
def dig(x):
    x=str(x)
    cnt=0
    for i in x:
        if(i.isdigit()):
            cnt+=1
    return cnt

length=lambda x:cal_len(x)
alphbet=lambda x:alp(x)
digit=lambda x:dig(x)
capital=lambda x:capL(x)
small=lambda x:smL(x)
special=lambda x:spc(x)

data['N']=pd.DataFrame(data.password.apply(length))
data['A']=pd.DataFrame(data.password.apply(alphbet))
data['D']=pd.DataFrame(data.password.apply(digit))
data['U']=pd.DataFrame(data.password.apply(capital))
data['L']=pd.DataFrame(data.password.apply(small))
data['S']=pd.DataFrame(data.password.apply(special))
# %%
data.head()
# %%
print("Q1")
# %%
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes[0,0].hist(x=data['N'],bins=30)
axes[0,0].set_title("Histogram for Length of Password")
axes[0,0].set_xlabel('LENGTH')
axes[0,0].set_ylabel('COUNT')

axes[0,1].hist(x=data['A'],bins=30)
axes[0,1].set_title("Histogram for the number of Alphabets of Password")
axes[0,1].set_xlabel('the number of Alphabets')
axes[0,1].set_ylabel('COUNT')

axes[1,0].hist(x=data['D'],bins=30)
axes[1,0].set_title("Histogram for the number of Digits of Password")
axes[1,0].set_xlabel('the number of Digits')
axes[1,0].set_ylabel('COUNT')

axes[1,1].hist(x=data['U'],bins=30)
axes[1,1].set_title("Histogram for the number of Upper-case Alphabets of Password")
axes[1,1].set_xlabel('the number of Upper-case Alphabets')
axes[1,1].set_ylabel('COUNT')

axes[2,0].hist(x=data['L'],bins=30)
axes[2,0].set_title("Histogram for the number of Lower-case Alphabets of Password")
axes[2,0].set_xlabel('the number of Lower-case Alphabets')
axes[2,0].set_ylabel('COUNT')

axes[2,1].hist(x=data['S'],bins=30)
axes[2,1].set_title("Histogram for the number of Special characters of Password")
axes[2,1].set_xlabel('the number of Special characters')
axes[2,1].set_ylabel('COUNT')

plt.tight_layout()
plt.show()
# %%
print("Q2")
# %%
f, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
sns.histplot(data['N'],kde=True,ax=axes[0,0])
axes[0,0].set_title("Distribution for Length of Password")
axes[0,0].set_xlabel('LENGTH')
axes[0,0].set_ylabel('COUNT')

sns.histplot(data['A'],kde=True,ax=axes[0,1])
axes[0,1].set_title("Distribution for the number of Alphabets of Password")
axes[0,1].set_xlabel('the number of Alphabets')
axes[0,1].set_ylabel('COUNT')

sns.histplot(data['D'],kde=True,ax=axes[1,0])
axes[1,0].set_title("Distribution for the number of Digits of Password")
axes[1,0].set_xlabel('the number of Digits')
axes[1,0].set_ylabel('COUNT')

sns.histplot(data['U'],kde=True,ax=axes[1,1])
axes[1,1].set_title("Distribution for the number of Upper-case Alphabets of Password")
axes[1,1].set_xlabel('the number of Upper-case Alphabets')
axes[1,1].set_ylabel('COUNT')

sns.histplot(data['L'],kde=True,ax=axes[2,0])
axes[2,0].set_title("Distribution for the number of Lower-case Alphabets of Password")
axes[2,0].set_xlabel('the number of Lower-case Alphabets')
axes[2,0].set_ylabel('COUNT')

sns.histplot(data['S'],kde=True,ax=axes[2,1])
axes[2,1].set_title("Distribution for the number of Special characters of Password")
axes[2,1].set_xlabel('the number of Special characters')
axes[2,1].set_ylabel('COUNT')

plt.tight_layout()
plt.show()
# In Q1 and Q2, all y label 'COUNT' means that the number of password that satified with x labels
# Q1과 Q2 에서, 모든 y 라벨에 해당하는 COUNT가 뜻하는 것은 x 축 값을 만족하는 password의 수입니다.
# %%
print("Q3")
# %%
corr_N_A = np.corrcoef(data['N'], data['A'])[0, 1]
print(f"Pearson correlation between N and A: {corr_N_A}")

corr_N_U = np.corrcoef(data['N'], data['U'])[0, 1]
print(f"Pearson correlation between N and U: {corr_N_U}")

corr_N_L = np.corrcoef(data['N'], data['L'])[0, 1]
print(f"Pearson correlation between N and L: {corr_N_L}")
# %%
