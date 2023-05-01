# Collaborative Filtering for Implicit Feedback Datasets

Keywords: CF, Implicit Feedback, RecSys
담당자: 준영 이, 서민석
발표일: 2023/03/29

[정리](https://www.notion.so/bfe190c430f74be2ab756a2fd37de2ed)

# 😎 몸에 좋은 한 줄 요약

<aside>
📌 암시적 피드백(Implicit Feedback)을 이용한 추천 시스템 분야에서 latent factor model을 사용해서 정확성을 향상시킵니다.

</aside>

# 1. Introduction

<aside>
💬 본 논문에선 TV 프로그램에 대해 Implicit Feedback Datasets(암시적 피드백)을 대상으로 CF를 적용하려고 합니다.

아래는 Implicit Datasets에 대해 알아야 하는 특징들입니다.

</aside>

## 1.1 Implicit Datasets의 특징

- **No negative feedback** - 부정적인 피드백이 없다.
    - 좋아하는 아이템에는 오랫동안 머무를 수 있고, 자주 소비할 수 있다.
    - 근데, 소비하지 않았다고 싫어한다고 볼 수 없다.
- **Inherently noisy** - 노이즈가 많다.
    - 소비는 선호를 보장하지 않음; 소비를 했다고 해서 사용자가 선호한다는 의미는 아님
    - 예를 들어 물건을 구매했다고 해도 자신이 쓴 것인지 선물용인지 알 수 없다.
    - 또, 영상을 오래 틀어 놓는다고 해도 진짜 보고 있는건지 알수 없다.
- **No preference but confidence** - 선호도가 아닌 신뢰도로 생각해야 한다.
    - 노이즈가 많아서 소비는 선호를 보장하지 않지만, 신뢰도로 생각해볼 수 있다.
    - 어떤 영상을 1번 보면 좋아 하겠지? 지만, 100번 보면 좋아 하는구나~할 수 있다.
- **Hard to evaluate** - 평가가 힘들다
    - 평가 기준을 잡기가 애매모호하다.
    - 예를 들면, 평가 기준에는 동시간대에 방영되는 TV 프로그램들 간의 경쟁도 있고 반복적으로 시청하는 데이터도 있다.

# 2. Preliminaries

<aside>
💬 Explicit feedback dataset과 implicit feedback dataset의 input data: $r_{ui}$ 를 비교합니다. 

Implicit feedback dataset에서 $r_{ui}$ 는 ‘사용자 $u$ 가 아이템 $i$ 를 얼마나 소비했는지 관측된 값’을 의미합니다. 

Explicit feedback dataset에서 일반적으로 대부분의 사용자-아이템 쌍에 대한 $r_{ui}$ 값은 unknown으로 결측값 처리되는 반면, implicit feedback dataset을 사용하는 경우, 모든 사용자-아이템 쌍에 대해 $r_{ui}$ 값을 할당하는 것이 가능해집니다.

</aside>

- implicit feedback dataset: $r_{ui}$ → observation
    
    ex 1) $r_{ui}=0.7$ → watched 70% of the show
    
    ex 2) $r_{ui}=2.0$ → watched the show twice
    
    ex 3) $r_{ui}=0.0$ → zero watching time 
    

- explicit feedback dataset: $r_{ui}$ → preference
    
    ex 1) $r_{ui}=1$ → week preference
    
    ex 2) $r_{ui}=5$ → strong preference
    

# 3. Previous work

<aside>
💬 앞서 기존에 있는 모델을 소개합니다.

</aside>

## 3.1 Neighborhood models

<aside>
💬 아이템간의 유사도를 계산하고, 이를 이용해 다른 아이템의 observation을 계산하는 모델입니다.

</aside>

- 아이템 $i$와 가장 유사한 $k$개의 아이템 $j$으로 추론합니다.

$$
\hat r_{ui} = \frac
{\sum_{j \in S^k} s_{ij} r_{uj}}
{\sum_{j \in S^k} s_{ij}}
$$

## 3.2 Latent factor models

<aside>
💬 Latent factor model은 **matrix factorization**에 기반한 모델로,

user-based observation $r_{ui}$을 latent factor들의 내적으로 표현한 모델입니다. 

- matrix factorization
    
    matrix r를 아래와 같이 2 x 3 matrix라고 하면,
    
    $$
    r = \left[
    \begin{matrix}
        r_{11} & r_{12} & r_{13} \\
        r_{21} & r_{22} & r_{23} \\
    \end{matrix}
    \right]
    $$
    
    matrix r은 $(2 \times f)$와 $(f \times 3)$의 행렬곱으로 표현할 수 있습니다.
    
    $$
    X = \left[
    \begin{matrix}
        -x_{1}- \\ 
        -x_{2}- \\
    \end{matrix}
    \right]
    Y = \left[
    \begin{matrix}
        -y_{1}- \\ 
        -y_{2}- \\
        -y_{3}- \\
    \end{matrix}
    \right]
    \\
    r = X^TY
    $$
    
</aside>

### Loss

- 목적식은 아래와 같습니다. (람다항은 정규화를 위함)

$$
min_{x, y} \sum (r_{ui} - x_u^Ty_i)^2 + \lambda(||x_u||^2 + ||y_i||^2)
$$

# 4. Our model

<aside>
💬 Implicit feedback dataset에 적합한 Matrix Factorization 기반의 모델을 설계하였습니다.

</aside>

## 4.1 Transfer raw observation into preference and confidence level

<aside>
💬 사용자의 아이템에 대한 선호가 다양한 신뢰수준을 가지는 형태로 나타난다고 주장하고, 이를 모델에 반영하기위해 preference $p_{ui}$와 confidence $c_{ui}$라는 변수를 도입합니다.

그리고 실험을 통해 이것이 implicit feedback dataset의 특성을 더 잘 반영하고 예측 정확도를 개선하는데 필수적임을 증명합니다.

</aside>

### P**reference : 선호 여부**

$p_{ui}$ → preference of user $u$ to item $i$ 

$$
p_{ui}=\begin{cases}1 & r_{ui}>0\\0 & r_{ui}=0\end{cases}
$$

- if user $u$ consumed item $i$ → user $u$ likes $i$
- if user $u$ never consumed item $i$ → no preference

### C**onfidence : 선호의 신뢰도**

$c_{ui}$ → measure our confidence in observing $p_{ui}$

$$
c_{ui}=1+\alpha r_{ui}
$$

- we have minimal confidence in $p_{ui}$ for every user-item pair
- we observe more evidence for positive preference, our confidence in $p_{ui}=1$ increase accordingly

## 4.2 Loss

<aside>
💬 제안 모델은 Matrix Factorization의 기본 컨셉과 동일합니다.

binary 변수인 preference를 예측하는 점과 
confidence를 도입한 목적함수를 정의하여 사용하는 점에서 차이를 보입니다.

</aside>

### L**oss**

preferences are assumed to be the inner product: $p_{ui}=x_{u}^{T}y_{i}$

$$
\min_{x_{*}, y_{*}}\sum_{u,i}^{}c_{ui}(p_{ui}-x_{u}^{T}y_{i})^{2}+\lambda(\sum_{u}||x_{u}||^2+\sum_{i}||y_{i}||^2)
$$

- L2 regularization → overfitting 방지
    - $\lambda(\sum_{u}||x_{u}||^2+\sum_{i}||y_{i}||^2)$
- account for the varying confidence levels

## 4.3 Optimization: Alternative Least Square(ALS)

<aside>
💬 implicit feedback dataset에서는 최적화에 $m \cdot n$ 개의 모든 사용자-아이템 쌍을 고려해야 합니다. 

따라서 ****exlpicit feedback datasets에서 널리 사용되는 SGD 같은 최적화를 사용하는 것은 어려움이 있다(prevent)고 주장하며 ALS라는 효율적인 최적화 프로세스를 제안하고 행렬의 수학적 성질을 이용하여 이를 확장 가능한 형태로 구현합니다.

</aside>

- explicit feedback datasets → sparse objective function
- implicit feedback datasets → dense cost function

optimization : ALS(Alternating least square)

- $x_{u}$ , $y_{i}$ 중 하나를 상수로 고정하고 다른 하나를 업데이트 하는 것을 번갈아 반복
- 하나를 상수로 고정하면 목적함수가 quadratic form이 되어 global mininum을 한번에 계산할 수 있음
    - $x_{u}=(Y^{T}C^{u}Y+\lambda I)^{-1}Y^TC^{u}p(u)$
    - $y_{i}=(X^{T}C^{i}X+\lambda I)^{-1}X^TC^{i}p(i)$
    - 알고리즘 ([링크](https://youtu.be/5im_ZSOZdxI))
        
        `1: Initialize $X$, $Y$`
        
        `2: repeat`
        
        `3:     for $u$ = 1 to $m$ do`
        
        `4:         $x_{u}=(Y^{T}C^{u}Y+\lambda I)^{-1}Y^TC^{u}p(u)$`
        
        `5:     end for`
        
        `6:     for $i$ = 1 to $n$ do`
        
        `7:         $y_{i}=(X^{T}C^{i}X+\lambda I)^{-1}X^TC^{i}p(i)$`
        
        `8:    end for`
        
        `9: until convergence`
        

# 5. Explaining recommendations

<aside>
💬 일반적으로 latent factor model에서는 추천에 대한 설명을 하는 것이 어렵다고 알려져 있습니다.

본 논문에서는 제안모델을 사용하는 경우, 추천된 아이템과 유사한 아이템 목록을 함께 제시하는 방법을 통해 추천에 대한 설명이 가능하다고 주장합니다. 

이는 item-oriented neighborhood models에서의 방법과 매우 유사하지만, 가중치 행렬을 도입함으로써 특정 사용자의 개인의 관점에서 아이템-아이템 유사도를 계산한다는 점에서 차이를 보입니다.

</aside>

- similarity between items $i$ and $j$
    - $s_{ij}=y_{i}^{T}y_{j}$
- weighted similarity between items $i$ and $j$ from $u$’s viewpoint
    - $s_{ij}^{u}=y_{i}^{T}W^{u}y_{j}, W^{u}=(Y^{T}C^{u}+\lambda I)^{-1}$
    - $W^{u}$: $f \times f$ weighting matrix associated with user $u$

![Untitled](Collaborative%20Filtering%20for%20Implicit%20Feedback%20Data/Untitled.png)

# 6. Experimental study

## 6.1 Data description

<aside>
💬 실험에 사용한 television service 데이터를 수집한 방법을 설명합니다.

더 정확한 검증을 위해 데이터에 진행한 여러가지 전처리 과정에 대해 설명합니다.

</aside>

### Digital televison service data

- user: 300,000 set top boxes
- item: 17,000 unique programs
- 시스템은 4주 기간의 데이터 학습, 그 다음 1주 예측
    - 4주 미만은 낮은 예측성능, 그 이상은 적은 성능 증가폭 → tv 방송 스케줄링의 계절성을 고려해봤을 때 긴 학습기간은 불필요

### Remove “easy” prediction

- $r_{ui}$→ how many times user $u$ watched program $i$
- 매주 반복에서 같은 프로그램을 시청하는 경향이 존재 → 최근에 시청하지 않은 프로그램들만 추천될 수 있도록 training 기간에 사용자가 시청한 적이 있는 프로그램은 test set에서 제거

### Toggle to zero

- 더 정확한 검증을 위해 test set에서 사용자가 50% 이하로 시청한($r_{ui}<0.5$) 프로그램에 대한 $r_{ui}$는 0 값으로 치환

### Employ log scaling

- 같은 프로그램을 반복적으로 시청하는 경향은 $r_{ui}$값의 범위를 너무 넓게 만듦→ log scaling을 통해 보정
    
    ex 1) channel flipping → $r_{ui}=0$
    
    ex 2) watching film → $r_{ui}=2$ or $r_{ui}=3$
    
    ex 3) DVR recording the same program  → $r_{ui}=$ hundreds
    

$$
c_{ui}=1+\alpha\log(1+r_{ui}/\epsilon)
$$

### Down-weight the subsequent shows after a channel tuning event

- 하나의 채널에 오랜 시간 머무는 경향이 존재(momentum effect) → 가중치를 통해 채널변경 없이, 하나의 채널에서 이어지는 프로그램들을 계속 시청하는 경우 순차적으로 낮은 $r_{ui}$ 값을 갖도록 조정 → 2번째 프로그램은 절반, 5번째 프로그램은 99% 감소

## 6.2 평가 방법

<aside>
💬 실험에 사용할 평가기준으로 $rank_{ui}$의 weight-sum값인 $\overline{rank}$를 제시합니다.

이후 실험에서 기존 모델인 Neighborhood, 인기순(Popularity)과 제안모델의 $\overline{rank}$값을 통해, 제안모델이 기존모델보다 좋은 성능을 이끌어낸다고 주장합니다.

</aside>

- $rank_{ui}$ - 프로그램 $i$의 추천 순위
    - 사용자 $u$에게 추천해줄 프로그램을 추천순으로 정리했을때, 프로그램 $i$가 얼마나 우선순위 인지를 $rank_{ui}$로 나타냅니다.
    - $rank_{ui} = 0\%$는 최우선 순위,
    - $rank_{ui} = 100\%$이면 가장 후순위라는 뜻입니다.
- $\overline{rank}$ - 평가기준으로 사용할 $rank_{ui}$의 weight-sum 값
    - 낮을 수록 추천이 잘 된 것이라고 할 수 있습니다.
    
    $$
    \overline{rank} = \frac{\sum_{u,i}r^t_{ui} rank_{ui}}{\sum_{u,i}r^t_{ui}}
    $$
    

## 6.3 실험 1

<aside>
💡 $\overline{rank}$값을 통해 제안모델과 인기도(popularity)와 neighborhood model의 성능을 비교하고자 합니다.

</aside>

### 실험 결과

아래 그래프를 보면 제안 모델의 평가지표가 가장 우수한 것을 알 수 있습니다.

- Popularity
    
    → $\overline{rank} = 16.46\%$
    
- Neighborhood
    
    → $\overline{rank} = 10.74\%$
    
- 제안모델
    
    → $\overline{rank} = 8.35\%$ (200 factors)
    
    → factor의 개수가 증가하면 성능이 더욱 향상되는 경향을 보입니다.
    

![Untitled](Collaborative%20Filtering%20for%20Implicit%20Feedback%20Data/Untitled%201.png)

## 6.4 실험 2

<aside>
💡 테스트 셋을 이용해 모델의 정확도를 측정하고 비교하고자 합니다.

정확도는 추천 프로그램이 테스트 셋에서 상위 x%에 해당하는 비율로 합니다.
누적 분포 함수를 사용해 모델의 정확도를 나타내며, 제안모델은 100개의 factor로 구성되었습니다.

</aside>

### 실험 결과

- 제안모델(Factor)
    - 제안모델의 추천 프로그램이 상위 1%에 해당할 확률은 대략 27%로 나타났습니다.
    - Neighborhood보다 약간 더 나은 성능을 보입니다.
    - Popularity보다 꽤 나은 성능을 보이고 있습니다.
- w/ prev. watched
    - 이전에 시청한 프로그램을 testset에서 제거하지 않았을 때, 제안모델 정확도를 보여줍니다.
    - 새로운 프로그램을 추천하는 것보다 이전에 시청했던 프로그램을 추천하는 것이 더 좋은 선택임을 알 수 있습니다.

![Untitled](Collaborative%20Filtering%20for%20Implicit%20Feedback%20Data/Untitled%202.png)

## 6.5 실험 3

<aside>
💡 제안모델의 성능을 인기도(popularity), 시청시간(watching time)으로 분석하고자 합니다.

데이터 셋을 비슷한 수치의 15개의 동일한 크기의 묶음(equal bin)으로 나누어 성능을 분석합니다.
1번째 묶음이 가장 작은 수치, 15번째의 묶음이 가장 큰 수치를 나타냅니다.

예컨데, 아래 bin 1#의 파란색 점은 가장 인기없는 프로그램 데이터 묶음이고,
아래 bin 15#의 파란색 점은 가장 인기있는 프로그램의 묶음입니다.

</aside>

### 실험결과

- Popularity
    - **인기가 많은 데이터일수록 예측이 쉬워지는 경향**을 보입니다.
    - 반면에 **인기가 적은 데이터일수록 예측이 급격하게 어려워집니다.**
    - 저자는 이러한 경향을 인기가 많은 데이터일수록 많고, 분석하기 쉽기 때문이라고 설명했습니다.
- Watching time
    - 시청 기록이 거의 없는 첫번째 묶음을 제외하면, **모든 묶음의 모델 성능이 성능이 비슷한 경향**을 보입니다.
    - 사용자에 대한 정보를 많이 수집할수록 품질이 향상될 것 같지만, 실제로는 전혀 그러한 경향을 보이지 않았습니다.
    - 저자는 이러한 경향을 많은 사람들이 같은 프로그램을 시청했기 때문이라고 설명했습니다.

![Untitled](Collaborative%20Filtering%20for%20Implicit%20Feedback%20Data/Untitled%203.png)

---

# 회고

- 민석
    - ChatGPT는 도움이 되면서 안 된다,,
        - 결국은 정독한 후에야 논문 내용이 이해가 조~금 됨..
        - 처음엔 답변을 해주는 것 자체가 신기했는데 쓰면 쓸수록 먼가 애매한 답변도 많고 오히려 답변을 듣고 헷갈리게 되는 부분도 있었다. → 내가 질문을 헷갈리게 했나?
        - 다른 능력자분들이 논문 스터디에 ChatGPT를 사용한 예시를 찾아봐야겠다..
    - 같이 하니까 좋았다.
        - 서로 정리한 내용을 상호보완하는식으로 완성하는 과정이 재밌었다.
        - 혼자서 하면 너무 힘들 것 같긴하다..
- 준영
    - 민석님 짱
        - 민석님께서 정확한 정보를 전달해야 한다고 고심하는 것을 보고 마음을 고쳐먹었다..
            
            → 정확한 정보를 전달해야 한다는 것은 당연한 일인데, 그냥 논문 읽기에만 신경쓴듯. 
            
            → 혼자 하다보면 아무생각없이 하는 것 같다…
            
        - 이해가 안되는 부분에 대해 설명해 주셔서 아주 좋았당
            
            → 부분 부분 이해가 안되는 부분을 여쭤봤는데, 친절하게 설명해주셨다.. 갓민석 🙂!
            
        - 구성에 대한 고민을 같이 해보며 어떻게 하면 짜임새 있을지, 어떤 것을 강조해야 할지 생각할 수 있었다
    - 논문 스터디… 어떻게 하는거야..
        - 해보면서 방향감을 찾아가고 싶다.
            
            → 처음해보는 논문 스터디라서 방향감을 아예 못잡고 걱정을 했었다.
            
            → 그렇지만, 일단 해보는게 가장 좋을 것 같음