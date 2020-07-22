---
title: Titanic: Machine Learning from Disaster
category: Kaggle
tag: [Kaggle, Data, ML]
---

데이터 분석을 공부하면서 Kaggle로 문제 풀면서 공부하면 좋을 것 같아서 하나씩 풀어보려고 합니다.
다양한 풀이가 존재하고, 공부하는 입장에서 정말 많은 코드 및 설명을 참고했습니다. 본인의 풀이를 공유해주신 많은 분들께 정말 감사드립니다. 


# 문제 설명

기계학습을 사용하여 타이타닉호에서 살아남은 승객을 예측하는 모델을 만드는 것이다. 

1912년 4월 15일, 첫 항해 중 빙산과 충돌한 후 침몰했다. 불행히도 탑승객 모두를 위한 구명보트는 존재하지 않았고, 2224명의 승객 및 승무원 중 1502명이 사망했다. 

살아남는 것에는 행운의 요소가 있었지만, 몇몇 그룹의 사람들은 다른 사람들보다 살아남을 가능성이 더 높았던 것 같다. 

승객 데이터 (이름, 나이, 성별, 사회적 계층 등)를 활용하여 " 어떠한 사람이 살아남을 가능성이 높았는가 " 에 대해 예측하는 모델을 만들어보자. 


# Code

~~~
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

## 데이터는 탑승객들의 신상정보를 series로 가지는 dataframe으로 구성
## 승객별로 생존여부가 모두 라벨링되어있으므로 지도학습에 해당
## 여러개의 변수를 이용하여 최종적으로 생존(1)과 사망(0)을 나눠야하는 이진 분류 문제 

# 각 데이터에 대한 설명은 다음과 같다.

#survival    생존여부                 0=No, 1=yes
#pclass      사회-경제적 지위           1=1st, 2=2nd, 3=3rd
#sex         성별
#Age         나이
#sibsp       같이 탑승한 형제-자매 수
#parch       같이 탑승한 부모-자녀 수
#ticket      티켓 번호
#fare        탑승 요금
#cabin       방 번호
#embarked    탑승 지역(항구 위치)        C=cherbourg, Q=Queenstown, S=Southampton

# 결측치 파악 

## 데이터의 형태 파악

train.shape
test.shape

## 훈련 데이터는 891개의 행과 12개의 열로 이루어져있다. 
## 테스트 데이터는 학습시킨 모델을 통해 라벨링을 해야하므로 타깃(라벨)에 해당하는 Survived 열이 빠져있다.

## info()는 dataframe에 대해 각 series(column)의 타입과 결측치 개수의 정보를 요약해서 알려준다. 
### pandas의 info()는 dataframe에만 적용할 수 있고, series에는 쓸 수 없다.

train.info()
test.info()

## 결측치의 갯수 파악

train.isnull().sum()
test.isnull().sum()


def bar_chart(feature):
    
    # 각 column(=feature)에서 생존자 수 count
    survived = train[train['Survived']==1][feature].value_counts()
    
    # 각 column에서 사망자 수 count
    dead = train[train['Survived']==0][feature].value_counts()
    
    # 생존자 수, 사망자 수를 하나의 dataframe으로 묶는다.
    df = pd.DataFrame([survived, dead])
    
    # 묶은 dataframe의 인덱스명(행 이름)을 지정한다.
    df.index = ['Survived', 'Dead']
    
    #plot을 그린다.
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('SibSp')
bar_chart('Parch')

## 정제가 필요한 column에 대해 전처리 진행

### Sex
# 성별 데이터는 결측치가 없고 분류도 이미 끝나있다. 학습이 잘 진행될 수 있도록 각 문자들을 숫자에 대응시킨다.

train_test_data = [train, test]

sex_mapping = {"male":0, "female":1}

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

### SibSP, Parch

for dataset in train_test_data:
    
    # 가족수 = 형제자매 + 부모 + 자녀 + 본인
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #혼자일 경우 
    
    # 가족수가 1보다 크면 동승자가 있다.
    dataset.loc[dataset['Family'] > 1, 'IsAlone'] =0

bar_chart('IsAlone')

### Embarked
## S승객 사망 비율이 높음을 위에서 확인할 수 있었다. 
## 거주지역의 차이가 경제적 지표를 나타낼 수 있음

class_list=[]

for i in range(1,4):
    series = train[train['Pclass'] == i]['Embarked'].value_counts()
    class_list.append(series)
    
df = pd.DataFrame(class_list)
df.index = ['1st', '2nd', '3rd']
df.plot(kind="bar", figsize=(10,5))

## Q지역이 비교적으로 다른 지역에 비해 못하는 것을 확인할 수 있다. 
## 대부분의 승객이 S지역에서 탑승했음을 확인할 수 있다. 

## 탑승 지역을 숫자에 매핑하고 결과를 확인 

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0, 'C':1, 'Q':2}

for dataset in train_test_data:
    dataset['embarked'] = dataset['Embarked'].map(embarked_mapping)
    
train.head()

## 이름의 경우 생존율과 유의미한 관계가 있지 않을 것으로 추정
## 그러나 서양인의 경우 이름은 그 사람의 성별, 혼인 여부를 포함하므로 이 부분만 추출
## 해당 내용은 정규표현식을 사용하여 진행
## https://wikidocs.net/21703

## 정규표현식 : https://regexr.com/

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
    
train['Title'].value_counts()
test['Title'].value_counts()

## 확인 결과 mr, miss, mrs, master가 대다수를 차지하고있으므로, 나머지는 하나로 취급한다. 

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)
    
## pandas의 apply 메서드는 series, dataframe의 각 entry에 접근하여 특별한 조작을 하고 싶을 때 많이 사용한다. 
## series에 메서드를 적용하면 리턴값은 보통 series이지만 어쩐 조작을 하느냐에 따라 dataframe이 출력될 수도 있다. 
## dataframe에 대해서도 결과 값은 series이거나 dataframe이다. 
## apply에서는 lambda를 사용할 수 있는데 lambda 변수명 : 변수에 대한 조작 내용 으로 이용한다. 
## 위의 코드를 기준으로 lambda 다음에 오는 x는 우리가 접근하고자 하는 title series의 성분 하나하나를 의미힌다. 

## 숫자에 매핑한 것을 bar_chart로 시각화한다. 

bar_chart('Title')

## 성인 남성인 Mr이 압도적으로 사망률이 높으며, 여성에 해당하는 Miss, Mrs가 많이 생존하였으나 가족이 없는 Miss가 더 생존율이 낮음을 확인할 수 있다. 
## Master는 남성이지만 어린아이가 많아서 사망률보다 생존률이 더 높다. 

## 매핑 확인
train.head()

train['Cabin'].value_counts()
train['Cabin'] = train['Cabin'].str[:1]
class_list=[]
for i in range(1,4):
    a = train[train['Pclass'] == i]['Cabin'].value_counts()
    class_list.append(a)

df = pd.DataFrame(class_list)
df.index = ['1st', '2nd', '3rd']
df.plot(kind="bar", figsize=(10,5))

## 가족 단위로 비슷한 방번호를 부여받았으니, 선실이 같으면 같은 등급의 클래스일 가능성이 높다. 
## 알파벳과 클래스와의 상관관계를 시각화함

## 1등급과 3등급은 공유하는 알파벳이 없다. 
## 그러나 cabin에 대한 정보는 결측치가 과반수 이상이고 그 마저도 1등급에 매우 편향되어있다.

## Age 
## 나이정보에는 결측치가 존재 
## 승객들의 title이 나이대를 어느정도 반영하고 있으므로 title에 해당하는 그룹의 중간값으로 결측치를 메꾼다.

for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
    
    

## pandas의 fillna 메서드에서 파라미터 inplace=True로 할 경우 해당 series에 결측치가 직접 채워진다. 
## inplace=Flase일 경우 결측치가 채워진 새로운 series가 리턴된다. 
## fillna는 dataframe, series에 둘 다 사용 가능하다. 

## 결측치를 채운 후 나이대별로 그룹화한다. 
## 그룹화 기준을 판단하기 위해 그래프를 그려서 분포를 확인한다. 

g = sns.FacetGrid(train, hue="Survived", aspect=4)
g = (g.map(sns.kdeplot, "Age").add_legend()) # add_legend()는 범주를 추가하는 파라미터이다.

## 청소년, 청년, 중년, 장년, 노년 5개의 집단으로 나눈다.

for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
    
## Fare 
## 탐승요금은 높은 등급의 승객일수록 높다. 
## 따라서 fare의 결측치는 각 승객 등급별 중간값으로 채운다. 

for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    
## 그래프를 그려서 그룹화의 기준을 찾는다. 

g = sns.FacetGrid(train, hue="Survived", aspect=4)
g = (g.map(sns.kdeplot, "Fare")
     .add_legend() # 범주 추가
     .set(xlim=(0, train['Fare'].max()))) # x축 범위 설정
     
## 승객 별로 탑승요금의 편차가 굉장히 크고 분포는 우측 꼬리가 길게 편향됨.
## 즉, 데이터를 그룹화 할 때 길이가 아닌 개수를 기준으로 나눈 다음 Farebin이라는 열에 저장 

for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])

## 전처리가 끝났으니 train에 사용하지 않을 열(column)은 삭제한다. 

drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
    
## pandas의 drop메서드는 Index(행) 혹은 series(열) 이름을 인수로 받아 해당하는 행 또는 열을 삭제한다. 
## axis 파라미터는 0일 경우 행, 1일 경우 열을 의미

# 전처리 종료 

## 결측치가 제대로 채워졌는지 확인

train.info()
test.info()

## 데이터 학습 시작
## PassengerID는 승객들의 번호에 불과하므로 학습시키지않는다. 
## Survived 또한 결과에 해당하므로 학습시키지않는다. 
## 불필요한 내용은 삭제하고, 나머지 유의미한 것만 target에 저장하여 훈련 데이터에선 삭제한다. 

drop_column2 = ['PassengerId', 'Survived']
train_data = train.drop(drop_column2, axis=1)
target = train['Survived']

## 데이터 학습, 모델 생성 및 평가에 필요한 패키지들을 import한다. 
## 의사결정나무, 랜덤포레스트, 나이브 베이즈 분류, 서포트 벡터 머신, 로지스틱 회귀 

from sklearn.tree import DecisionTreeClassifier # 의사결정나무
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트
from sklearn.naive_bayes import GaussianNB # 나이브 베이즈 분류
from sklearn.svm import SVC # 서포트 벡터 머신
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀

# 의사결정나무
clf = DecisionTreeClassifier()
clf.fit(train_data, target)
clf.score(train_data, target)
## 0.8787878787878788

# 랜덤 포레스트
clf = RandomForestClassifier()
clf.fit(train_data, target)
clf.score(train_data, target)
## 0.8787878787878788

# 로지스틱 회귀
clf = LogisticRegression()
clf.fit(train_data, target)
clf.score(train_data, target)
## 0.8047138047138047

# 나이브 베이즈 분류
clf = GaussianNB()
clf.fit(train_data, target)
clf.score(train_data, target)
## 0.8002244668911336

# 서포트 벡터 머신
clf = SVC()
clf.fit(train_data, target)
clf.score(train_data, target)
## 0.8361391694725028

## 의사결정트리와 랜덤 포레스트가 가장 높은 점수 
## test 데이터에서도 train과 마찬가지로 passengerId는 삭제하고, 
## test 데이터를 모델에 적용해서 예측한 결과를 predict에 저장

clf = DecisionTreeClassifier()
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1)
predict = clf.predict(test_data)

submission = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : predict})

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv("submission.csv")
submission.head()

~~~

# 결과 

![result](https://i.imgur.com/CifLGW2.png)


## 참고 

https://predictors.tistory.com/6
https://www.ahmedbesbes.com/blog/kaggle-titanic-competition
https://towardsdatascience.com/how-i-got-a-score-of-82-3-and-ended-up-being-in-top-4-of-kaggles-titanic-dataset-bb2875cee6b5
https://romanticq.github.io/%EC%BA%90%EA%B8%80/kaggle-titanic/
