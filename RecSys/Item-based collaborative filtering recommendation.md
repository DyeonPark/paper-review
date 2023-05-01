# Item-based collaborative filtering recommendation algorithms

Keywords: CF, RecSys
담당자: 박동연, 주혜인
발표일: 2023/03/29

**🔥 색상 별 의미 🔥**

- **파란색** → 알면 좋을 내용 및 요약
- 노란색 배경 ****→ 중요한 내용
- **초록색** → 작성자의 의견 (이자 뇌피셜)

# 😎 몸에 좋은 한 줄 요약

<aside>
📌 **Item-based CF 가 user-based CF 보다 좋다!**

</aside>

# 2. CF(Collaborative Filtering, 협업 필터링)이란?

![Untitled](Item-based%20collaborative%20filtering%20recommendation/Untitled.png)

- 추천 시스템에서 사용되는 기술 중 하나
- **사용자들의 과거 행동(구매 이력, 선호도 등)을 바탕으로 유사한 취향을 가진 사용자들이 선호하는 아이템을 추천**하는 방법

- 간단한 Annotation 정리
    - 사용자 목록 **$U$:** $u_{1…m}$
    - 아이템 목록 $**I$:** $i_{1…n}$
    - 각 사용자들이 아이템에 대한 선호도를 저장한 $**Ratings\,Table**$ — (implicit/explicit)

- Collaborative filtering은 두 가지 역할 수행
    - **Prediction**: 유저가 아이템을 선호할 정도를 **수치로 예측하여 반환**
    - **Recommendation**: 유저가 좋아할 것 같은 아이템을 **추천 리스트로 반환**
     ($Top-N\,\, recommendation\,Algorithm$)

## 2-1. CF의 두 종류

- **User-based CF**
    - 어떤 사용자 A와 유사한 취향을 가진 다른 사용자 B를 찾아내어, 사용자 B가 선호하는 아이템을 추천하는 방법
    - nearest-neighbor 알고리즘으로 알려져 있음 ([k-NN Algorithm](https://rebro.kr/183))
    - 그러나 한계점이 존재
        - `Sparsity (희소성)`:  아이템 데이터셋이 유저의 수에 비해서 너무 크면, 유저간의 유사도 계산 기반 추천이 의미가 적어짐
            - 활성 유저(active user)의 구매 기록이 아이템 데이터셋을 반영하지 못하기 때문에, 이를 기반한 추천도 데이터셋을 충분히 고려 및 반영하여 추천해주지 못할 것
        - `Scalability (확장성)`:  수많은 유저가 존재할 때 연산량 부담
            - 본 논문에서는 유저는 동적(실시간)이고, 아이템은 정적(사전입력)이라고 가정
            - 또한 E-commerce가 지금처럼 활성화되기 전의 논문이라서 아이템의 수에 비해서 유저의 수가 더 많다고 가정한 것으로 추정 (2001s)
            - 따라서, 아이템보다 많은 수의 동적인(실시간) 유저 간의 유사도를 계산하는 것이 연산량에 부담이라고 표현한 것으로 추정
- **Item-based CF**
    - 어떤 아이템 A와 비슷한 아이템 B를 찾아내어, 아이템 A를 선호하는 사용자들에게 아이템 B를 추천하는 방법
    - 모델의 머신러닝 알고리즘에 따라서 다르게 동작함 (Bayesian Network / Clustering / Rule-based)

# 3. Item Similarity Computation

![Untitled](Item-based%20collaborative%20filtering%20recommendation/Untitled%201.png)

<aside>
💡 **알고리즘 한 줄 요약: 사용자-아이템 평가 행렬을 이용하여 사용자가 좋아하는 아이템과 유사한 아이템을 추천**

</aside>

**알고리즘 동작 단계**

1. 사용자 — 아이템 평가 행렬 구성
2. 특정 아이템 $i$와 임의의 아이템 $j$간의 유사도 계산
3. 특정 아이템 $i$와 가장 유사한 아이템 $k$개 및 유사도 값 리스트 도출 
(이 때, 유저 $u$가 구매한 아이템은 포함되면 안됨)
4. 위에서 도출한 값을 활용하여, 유저 $u$가 아이템 $k$ 대해서 평가할 확률 $P$를 계산
5. $P$값을 토대로 가장 높은 값을 보이는 아이템 $N$개를 추천

## 3.1 Item Similarity Computation

### 3.1.1 Cosine-based Similarity

![Untitled](Item-based%20collaborative%20filtering%20recommendation/Untitled%202.png)

- cos 함수를 이용하여 유사도를 측정
- 아이템 $i$와 아이템 $j$에 해당하는 열을 각 벡터로 생각하여, 유사도 측정

### 3.1.2 Correlation-based Similarity

![Untitled](Item-based%20collaborative%20filtering%20recommendation/Untitled%203.png)

- 피어슨 상관계수를 이용한 방법
- $U$ : 두 아이템 모두 구매한 유저들의 집합 — isolation 선행 필요 ($i$와 $j$를 둘다 구매한 유저를 골라내는 작업)
- $R_{u, i}$ : 유저 $u$가 아이템 $i$에 대해서 평가한 점수
- $R_i$: 아이템 $i$의 평균 점수

### 3.1.3. **Adjusted** Cosine Similarity

![Untitled](Item-based%20collaborative%20filtering%20recommendation/Untitled%204.png)

- 유저의 평가 성향을 고려한 방법임! — 유저들의 평균 점수를 고려
(짜게 주는 사람이 주는 10점과 후하게 주는 사람의 10점은 다르다!)
- $\overline{R_u}$  은 임의의 유저 u의 평균점수

**위 과정을 거침으로써 특정 아이템 $i$에 대해 아래와 같은 집합을 도출**

- 가장 유사한 아이템 $k$개의 집합: {$i_1, i_2, . . . , i_k$}
- 각 아이템의 아이템 $i$와의 유사도: { $s_{i1}, s_{i2}, . . . , s_{ik}$}
- 이 때, 임의의 유저 $u$가 구매한 아이템은 포함되면 안됨

## 3.2 Prediction Computation

CF system에서 가장 중요한 단계는 prediction의 관점에서 output interface를 만들어내는 것이다.

유사도를 바탕으로 유사하다고 할 수 있는 item들의 set을 골라낸 후에 이를 활용해서 prediction을 구하는 두 가지 방법을 소개한다.

### 3.2.1 Weighted Sum

- user u의 item i에 대한 예측은 i와 유사한 **item들에 대한 rating의 가중합**으로 구한다.
- item $i$와  $j$의 similarity 인 $s_{i,j}$가 가중치로 활용된다.

![https://blog.kakaocdn.net/dn/L1rFf/btr5OQMA0VP/Qp1UxbMlhGILooAMumZlk0/img.png](https://blog.kakaocdn.net/dn/L1rFf/btr5OQMA0VP/Qp1UxbMlhGILooAMumZlk0/img.png)

→  이러한 접근으로 통해 어떻게 active user가 비슷한 상품들을 평가하는지를 포착하려 했다.

### 3.2.2 Regression

regression을 통한 prediction은 위의 weighted sum method와 비슷하지만 직접적으로 비슷한 item들에 대한 rating을 사용하는 것이 아니라 **regression model에 기반한 rating의 근사치**를 사용한다.

그렇다면, 근사치를 사용하는 이유는 무엇일까?

앞서 계산한 유사도(cosine/correlation)은 두 벡터가 사실은 유사한데 거리는 멀리 떨어져 있어 결과를 오도할 수 있기 때문이다.

직접적인 $R_{u,N}$ 대신 $R'_{u,N}$을 구하는 방식은 아래와 같다.

![https://blog.kakaocdn.net/dn/bJ0H2F/btr53HNurQj/bgCoh5VKo6zwg2MxurbRi0/img.png](https://blog.kakaocdn.net/dn/bJ0H2F/btr53HNurQj/bgCoh5VKo6zwg2MxurbRi0/img.png)

여기서는 user에 따라 다른 rating을 활용하는게 아니라 user들의 item i 에 대한 rating의 평균값을 $\bar{R_i}$로, 모든 유저들에 대해 N개의 비슷한 item들에 대한 rating의 평균 값을 $\bar{R'_N}$으로 둔 회귀식입니다. 이렇게 구한 $\bar{R'_N}$을 위의 weighted sum과정에서 $R_{u,N}$ 대신 활용하는 것이라고 저는 이해했습니다만 논문에는 위에 적은 이외의 자세한 내용은 언급되지 않아서 혼란이 좀 있네요!

![https://blog.kakaocdn.net/dn/byDvI0/btr5QAhVaOS/AW7CQVhnHzq7sTPH0XRCa0/img.png](https://blog.kakaocdn.net/dn/byDvI0/btr5QAhVaOS/AW7CQVhnHzq7sTPH0XRCa0/img.png)

N = 5일때의 Item-based Collaborative Filtering 알고리즘을 도식화 한 것이다.

## 3.3 Performance Implications

- 이 논문에서는 item-item similarity를 미리 계산하기 위해 model-based approach를 사용한다.
    - **model-based system** : neighborhood generation과 prediction generation을 분리시킴으로서 scalability를 높일 수 있다.
        - similarity가 correlation-based로 계산되는 것은 동일하지만 계산이 item space에서 이루어진다.
    - neighborhood-based CF system : user-user simliarity를 계산하는 과정에서 performance bottleneck이 발생해 실시간 추천이 불가능할 수도 있다.

- 일반적인 E-Commerce상황에서 item은 고정되어 있고 user는 계속 변한다.
    
    이러한 item의 static한 특징으로 인해 아이템 간의 유사도를 **미리 계산**해둔다. 
    
    미리 계산하는 방법 : all-to-all similarity 를 계산해두고 필요한 유사도 값을 훑어서 검색한다.
    
    →  이 방법은 시간을 절약하는 동시에 n개의 item에 대해 $O(n^2)$의 공간복잡도를 가진다.
    
- **model size k** : item j에 대하여 가장 비슷한 item k개 (k << n)
    
    즉 상품을 추천할 때 한번에 얼마나 많은 유사한 상품들을 고려할건데? → 당연히 많은 상품의 정보를 참고할 수록 정확해지겠지만 늘어날수록 효율성은 떨어지겠죠? 그래서 아래에서 trade-off에 대해 소개합니다.
    

<aside>
🔎 **[prediction generation algorithm]**

1. user 의 item에 대한 prediction에 대해 미리 계산된 $k$개의 가장 유사한 아이템을 찾는다.
2. user $u$가 $k$개의 아이템 중 몇개의 아이템을 구매했는지 살핀다.
3. 그리고 item-based algirothm을 이용해 prediction을 계산한다.
</aside>

- Performance trade-off
    - 좋은 성능을 위해 model size를 키우는 것은 성능에 문제를 발생시킬 수 있다. (high space complexity)
    - 그러나 작은 모델 사이즈에도 좋은 성능을 낼 것이라는 가설을 세우고 이를 실험을 통해 증명한다.

# 4. Experimental Evaluation

## 4.1 Data Set

Movie Data - MovieLens (사용자들이 영화를 평가하고 추천받는 사이트)

- 100,000개의 rating을 얻기 위해 user를 랜덤하게 선발하고 80%를 train data로 활용했다.
- 이 데이터는 943*1682인 user-item matrix A로 가공되었다.
- sparsity level : $1-\frac{nonzero\ items}{total\  entries} = 1-\frac{100,000}{943 *1682} = 0.9369$

## 4.2 Evaluation Metrics

- Statistical accuracy metrics
    - user rating과 recommentation score를 비교한다.
    - 아래의 MAE외에도 RMSE, Correlation을 사용하기도 한다.

![https://blog.kakaocdn.net/dn/v7KYh/btr5Ul5CJsX/gNOVCVfeCTKPAW2qXN5zq1/img.png](https://blog.kakaocdn.net/dn/v7KYh/btr5Ul5CJsX/gNOVCVfeCTKPAW2qXN5zq1/img.png)

- Decision support accuracy metrics
    - user가 얼마나 높은 퀄리티의 아이템을 효과적으로 고르는데 prediction engine이 기여하는 정도를 평가
    - prediction process를 good/bad의 binary operation으로 본다.
    - reveral rate, weighted errors, ROC sensitivity
- 본 논문에서는 MAE를 활용한다.

### 4.2.1 Experimental Procedure

[Experimental steps]

1. training / test 분리
2. sensitivity of the parameter - training data만 활용해서 plot을 통해 결정

[Benchmark user-based system] 

성능 비교를 위해 user-user 알고리즘의 일종인 Pearson nearest neighbor algorithm 사용

## 4.3 Experimental Results

### 4.3.1 Effect of Similarity Algorithms

- 3개의 유사도 알고리즘을 사용 : cosine, **adjusted cosine**, correlation

![https://blog.kakaocdn.net/dn/k0lRE/btr5RSv1F7K/FBBVVrmgKTs6IOsYs7qBK0/img.png](https://blog.kakaocdn.net/dn/k0lRE/btr5RSv1F7K/FBBVVrmgKTs6IOsYs7qBK0/img.png)

위 결과에 따라 adjusted cosine similarity(offsetting the user-average)를 나머지 실험에서 활용했다.

### 4.3.2 & 4.3.3 아래의 결과에 따라 train  ratio : 0.8, neighborhood size : 30

![https://blog.kakaocdn.net/dn/evAYHn/btr5PSjfRtL/mpXJzsr1kqt9NT18W9FrE1/img.png](https://blog.kakaocdn.net/dn/evAYHn/btr5PSjfRtL/mpXJzsr1kqt9NT18W9FrE1/img.png)

### **4.3.4 Quality Experiments**

![https://blog.kakaocdn.net/dn/k8861/btr6bugli7B/TK7m3pCYPfGXo09zGy2km1/img.png](https://blog.kakaocdn.net/dn/k8861/btr6bugli7B/TK7m3pCYPfGXo09zGy2km1/img.png)

1. 모든 sparsity level에서 item-item이 좋은 성능
2. regression method가 neighbor가 낮을땐 괜찮은 성능을 보이지만 늘어날수록 낮아짐 -> regression model의 overfitting 문제 때문인 것 같다.

## 4.4 Sensitivity of the Model Size

![https://blog.kakaocdn.net/dn/6JGYs/btr6nXC5jHY/3T1Q6JpwH3LLSFaR9IHvW1/img.png](https://blog.kakaocdn.net/dn/6JGYs/btr6nXC5jHY/3T1Q6JpwH3LLSFaR9IHvW1/img.png)

→ 아이템의 일부만을 활용해도 높은 accuracy를 달성할 수 있다. 

1.9%의 아이템만 활용해 96%의 효과를, 3%의 아이템만을 활용해 98.3%의 효과를 낼 수 있다.

→ 아이템의 일부만 활용해 유사도를 미리 계산하는 것이 효과적인 동시에 좋은 예측 성능을 얻을 수 있다.

![Untitled](Item-based%20collaborative%20filtering%20recommendation/Untitled%205.png)

→ model size가 작을수록 recommendation에 걸리는 시간은 덜 들고 throughput(처리율)은 높아진다.

# 5. Conclusion

추천시스템은 사용자들이 효과적으로 원하는 물건을 구매할 수 있게 돕는 동시에 산업에도 매출을 늘릴 수 있게 도움을 준다. E-commerse에 중요한 도구가 되고 있다. 방대한 양의 사용자 데이터에 대한 접근이 가능해지면서 scalability를 극적으로 향상시킬 수 있는 새로운 기술들이 요구되고 있다.

이 논문은 실험을 통해 **CF-based algorithm으로 좋은 퀄리티의 추천을 하는 동시에 거대한 데이터 셋을 계산**할 수 있었다.

# 🧐 논의해볼 것

- 본 논문에서는 item을 static하다고 가정하지만, 실시간으로 item이 추가되거나 사라지는 현시대의 E-commerce에는 맞지 않는 얘기가 아닐까?
    - 민석: 서비스의 종류나 컨텐츠 유형에 따라 다르지 않을까?
- 그렇다면 실시간으로 아이템이 추가되거나 사라지는 요즘 e-commerce에서 주로 사용하는 추천 시스템 및 모델은 무엇일까?
    - Model-based Collaborative Filtering
- model size vs neighborhood size
    
    Neighborhood size와 Model size k는 추천 시스템에서 다른 개념입니다.
    
    Neighborhood size는 Collaborative Filtering 알고리즘에서 유사한 아이템 또는 유저의 개수를 의미합니다. 즉, 해당 아이템과 유사한 다른 아이템들의 개수를 의미하는 반면, 해당 유저와 유사한 다른 유저들의 개수를 의미하기도 합니다. 이 개수가 작을수록 계산 시간이 짧아질 수 있지만, 추천 결과의 다양성이 감소할 수 있습니다.
    
    반면, Model size k는 Matrix Factorization 알고리즘에서 latent factor의 개수를 의미합니다. 이 latent factor는 사용자와 아이템 간의 상호작용을 나타내는 특성입니다. 이 latent factor의 개수를 늘릴수록 정확도를 높일 수 있지만, 연산량이 많아질 수 있습니다.
    
    따라서, Neighborhood size와 Model size k는 서로 다른 개념이며, 추천 시스템에서 각각 다른 방법으로 사용됩니다.
    
    1. **Neighborhood Size**: In item-based collaborative filtering, the neighborhood size refers to the **number of similar items** that are considered when making recommendations for a given user. The algorithm identifies a set of items that are similar to the ones the user has interacted with in the past, and recommends items from this set. The size of this set is known as the neighborhood size.
        
        A larger neighborhood size typically results in more diverse recommendations, but may also result in recommendations that are less personalized to the user's preferences. On the other hand, a smaller neighborhood size may result in more personalized recommendations, but may also be more limited in terms of the variety of recommendations.
        
    2. **Model Size**: The model size refers to the number of **items and users in the dataset**, as well as the number of features used to represent each item and user. In item-based collaborative filtering, the model size affects the accuracy and scalability of the algorithm. A larger model size (i.e., more items, users, and features) generally results in better accuracy, but may also be more computationally intensive and require more resources.