---
title: 생선 분류 문제
category: DL
tag: [DL, Data]
---

* 혼자 공부하는 머신러닝+딥러닝 책을 공부한 내용을 정리하는 포스트입니다.

* 데이터셋은 캐글에 공개된 생선 데이터를 사용한다. 

[Fish Market Dataset](https://www.kaggle.com/aungpyaeap/fish-market)



# [생선 분류 문제]

데이터 셋에 있는 생선은 다음과 같다. 

- 도미
- 곤들매기
- 농어
- 강꼬치고기
- 로치
- 빙어
- 송어

## 도미 데이터 준비하기

도미의 특성(feature)을 산점도로 확인

~~~
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.title("Domi")
plt.xlabel("length")
plt.ylabel("weight")

plt.show
~~~

![domi_scatter](https://i.imgur.com/h2Db28x.png)


## 빙어 데이터 준비하기

~~~
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)

plt.xlabel("length")
plt.ylabel("weight")

plt.show
~~~

![도미, 빙어 산점도](https://i.imgur.com/vM7nHNt.png)


## K-Nearest Neighbors (K-최근접) Algorithm을 활용하여 데이터 구분하기 

~~~
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l,w] for l, w in zip(length, weight)] # length, weight list를 2차원 list로 변환
~~~

### 정답 데이터 작성

도미 1
빙어 0 으로 표기한다. 

~~~
fish_target = [1] * 35 + [0] * 14
print(fish_target)
~~~

~~~
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

kn.score(fish_data, fish_target)
~~~

|K-최근접 이웃 알고리즘
어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고 다수를 차지하는 것을 정답으로 사용

ex) 
~~~
kn.predict([[30,600]])
-> array([1])
~~~

근접한 n개의 데이터를 참고하여 판단을 하는데, 이 때 이 n의 수를 지정할 수 있다. 

ex)
~~~
kn49 = KNeighborsClassifier(n_neighbors=49)  ## 49개의 근접한 데이터를 참고
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
~~~
