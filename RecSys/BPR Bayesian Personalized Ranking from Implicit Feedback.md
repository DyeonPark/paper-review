# BPR: Bayesian Personalized Ranking from Implicit Feedback

Keywords: Implicit Feedback, RecSys
담당자: 주혜인, 준영 이
발표일: 2023/04/05

# 1. Introduction

<aside>
💬 이 논문에서는 personalized ranking에 대한 모델을 학습하는 포괄적인 방법을 제시한다.

</aside>

**[Why]** - ‘ranking’자체를 최적화하는 방법은 존재하지 않았다.

**[Item Recommendation]**

- 일련의 상품에 대해 개인화된 랭킹을 만들어내는 것
- 사용자의 item에 대한 선호도는 과거의 interaction으로부터 알 수 있다 - 구매내역/최근 본 상품/등등

**[학습방법]**

1. **BPR-Opt** : optimal personlized ranking
    
    AUC를 최대화하는 BPR-Opt라는 optimization기준을 제시한다.
    
    (derived from maximum posterior estimator)
    
2. **LEARNBPR**
    
    bootstraping sample을 이용한 sgd방법이다. BPR-Opt의 최적화와 관련하여 일반적인 경사하강법에 비해 우수한 성능을 보였다.
    
3. **applying**
    
    SOTA recommedation model에 Learn BPR을 적용하는 방법을 소개한다.
    
4. **Experiment**
    
    BPR을 활용해 모델을 학습하는 것이 personlized ranking 관점에서는 다른 모델보다 우수한 성능을 보였다.
    

# 2. Related Work

<aside>
💬 기존의 모델들은 ranking에 대한 직접적인 최적화는 없었다. 따라서, 본 논문에서 제시하는 personalized ranking을 위한 최적화 기준을 소개한다. 이후 이를 기존의 추천 모델에 적용하였을때 잘 적용되었다는 것을 실험을 통해 증명한다.

</aside>

**문제 - 기존 연구에 관한 정리**

---

- kNN CF : 최근에는 유사도를 파라미터로 간주해 학습을 통해 얻어내기도 한다.
- MF : SVD를 통해 학습되는 경우 overfitting에 취약하다는게 밝혀졌다. 따라서 정규화된 학습 방법이 제안되고 있다.
    - WR-MF : case weight를 활용한 regularized least square 방법
- probabilistic latent semantic model
- multi-class problem으로 바꾼 뒤 set of binary classifier로 해결하는 방법

 ⇒ 위 모델들은 personlized ranking dataset에 대해 평가되었음에도 불구하고 **ranking에 대해 모델 파라미터**를 직접적으로 최적화하지 않았다.

대신, 사용자에 의해 상품이 사용되었는지 아닌지를 예측하기 위한 최적화만 했다.

**본 논문에서 전개될 내용**

---

- 본 논문에서는 item의 쌍에 기반해 personlized ranking을 위한 최적화 기준을 소개한다.
- 기존의 추천 모델 - MF/adaptive kNN 들을 더 나은 ranking quality를 제공할 수 있는 방법에 의해 최적화한다.
- section 5 - maximum margin MF뿐만 아니라 WR-MF와 본 논문의 접근의 관계
- AUC를 기준으로 할 예정이다.
- offline 모델링에서 파라미터를 학습한 후 Online으로 확장시킨다. (기존의 MF가 온라인으로 확장시켰던 전략이 BPR을 사용해도 적용될 수 있다.)
- collaborative vs non-collaborative
    - non-collaborative model과 관련된 점도 있다. 이 접근방식은 개인화된 랭킹은 불가능하고, 하나의 랭킹만 만들어낼 수 있다.
    - collaborative - personlized ranking이 가능하다.
- 평가 단계에서 실험적으로 이론적인  non-personalized ranking의 upper bound보다도 훨씬 더 우수했다는 것을 보일 것이다.

# 3. Personalized Ranking

<aside>
💬 personalized ranking을 이 논문에서 표현하는 방식에 관한 섹션입니다. 
데이터  구성에서 기존의 방식과 조금 차이가 있습니다. **feedback이 관찰되지 않은 경우** 단순히 negative feedback으로 치부하는 것이 아니라 우리가 예측해야 할 대상 즉, **test data로 본다**는 점입니다.

</aside>

- Personlized Ranking?
    - user에게 순위가 매겨진 item list를 제공하는 것이 목적이다.
    - item recommendation이라고도 불린다.
- 본 논문에서는 user의 implicit behavior를 통해 ranking을 추측한다.
    - implicit behavior에는 positive feedback만 존재한다.
- non-observed feedback
    - NA & Negative Feedback

## 3.1 Formalization

본 논문에서 사용될 몇가지 notation과 $>_u$의 세 가지 조건에 대해 설명한다.

- $U$ : users
- $I$  : items
- $S \subseteq U\times I$  : scenario (user & implicit feedback)
    
    즉, 사용자와 아이템의 상호작용이 있었던 데이터 집합이다. 
    
- $>_u \subset I^2$ : personalized ranking of all items
- $>_u$의 세 가지 조건

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled.png)

- $I^+_u := \{i \in I :(u,i) \in S \}$
- $U^+_i := \{ u\in U:(u,i)\in S \}$

## 3.2 Analysis of the problem setting

앞서 계속 말했듯이, implicit feedback은 positive feedback만 존재한다. 그밖의 데이터는 negative feedback과 missing value다.

**[기존의 일반적인 item rec]**

- 목표 : $\hat{x}_{ui}$ (item에 대한 user의 선호도)를 예측한 뒤 이 score를 sorting한다.

**Problem - [일반적인 ML방법]** 

<aside>
⚙️ $(u,i) \in S$ : positive feedback ,그 외의 모든 다른 가능한 조합인 $(U\times I ) \not \ \ \ S$ : negative feedback으로 training data를 구성한다.

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%201.png)

⇒ 이후 모델은 1과 1이 아닌 나머지를 0으로 예측한다.

</aside>

→ 모델이 예측해야할 부분까지 negative feedback로 예측한다.

→ 정확하게 예측해낼 수 있는 (expressive) 모델도 다 0으로 예측하도록 시키니 순위를 제대로 매기지 못한다.

**Solution - 본 논문에서의 data 구성 방식**

$$
D_s := \{(u,i,j)\ |\ i \in I_u^+  \wedge j \in I \not\ \ \ I_u^+  \}
$$

- 단순히 NA를 neg로 대체하는게 아니라 traing data로 item pair($(u,i)$)를 사용하고 정확한 ranking을 구하기 위한 최적화를 실시한다.
- $>_u$ 의 한 부분이 되도록 user를 재구성한다.
    - i를 user가 보았다 : 보지 않은 다른 모든 상품들보다는 선호하는 상품이다.
- 이미 본(안본) 두 상품간의 ranking은 알 수 없다.

![Figure2. observed data S (left) + : user prefers item i over j](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%202.png)

Figure2. observed data S (left) + : user prefers item i over j

**[이 방식의 장점]**

1. training data는 긍정, 부정 피드백 그리고 NA로 구성되어있다. 
    
    관찰되지 않은 두 item간의 NA는 예측되어야 할 대상으로 본다.
    
    따라서, $D_s$와 test data가 disjoint하다.
    
2. training data는 실제 ranking에 대해 만들어졌다.

# 4. Bayesian Personalized Ranking

<aside>
💬 personalized ranking을 최적화하는 기준이 BPR-OPT이고, 이 최적화 기준을 통해 모델이 학습하도록 하는 알고리즘이 LEARN BPR이다. 

이 두 가지를 활용하면 이미 많이 알려진 추천 모델에 적용했을떄 우수한 성능을 보였다.

</aside>

**[이번 섹션에서 소개할 내용]**

1. 최적화 기준인 BPR-OPT을 소개한다.
    - $P(i >_aj|\theta)$ : Likelihood function
    - $P(\theta)$ : prior
    - AUC를 활용한다.
2. BPR-OPT에 관해 모델을 학습시키는 알고리즘인 LEARN BPR을 소개한다.
3. 기존의 우수한 RecSys model에 위 두 가지를 적용시켰을때 잘 작동한다는 것을 보여준다.

## 4.1 BPR Optimization Criterion

<aside>
💬 결국 posterior를 최대화하기 위해선, prior와 likelihood를 정확하게 알아내야 한다.
이 섹션에서는 앞서 정의한 $>_u$에 대한 가정을 바탕으로 식을 구체화하는 과정을 소개한다.

</aside>

모든  $i\in I$ 에 대한 정확한 personalized ranking을 찾는 것은 posterior probability를 최대화하는 $\Theta$를 찾는 것과 같다.

$$
P(\Theta | >_u) \propto P(>_u|\Theta)P(\Theta)
$$

여기서 $>_u$는 잠재적인 user u의 선호라고 생각하면 된다. likelihood function과 prior확률을 정리하는 과정이다.  아래는 각각의 likelihood function, prior probability에 대해 자세하게 설명한다. 

- 1. likelihood function
    - 가정
        1. 모든 **user는 독립적**으로 행동한다.
        2. 특정 사용자의 $(i,j)$쌍의 순서는 **모든 다른 쌍의 순서를 매기는 것과 독립적**으로 작용한다.
        
        ⇒ 이 가정을 바탕으로 $P(>_u|\theta)$는 아래와 같이 single density 의 곱으로 나타내고 모든 user $u\in U$에 대해 결합될 수 있다.
        
        **독립을 가정했기 때문에 모든 user에 대한 결합확률은 아래와 같이 곱셈을 통해 정리할 수 있다.**
        
        ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%203.png)
        
        ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%204.png)
        
        ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%205.png)
        
        - user가 item j 보다 item i를 선호활 확률인 $p(i>_u j)$를 아래와 같이 정의한다.
        - 이는 확률이기에 sigmoid함수를 활용해 정의한다.
        
        ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%206.png)
        
        - 여기서 $\hat{x_{uij}}(\Theta)$는 u, i, j의 관계를 담고 있는 임의의 real-valued function으로 알아내야할 모수에 대한 함수다.
        - MF, adaptive kNN과 같은 방법으로 추정한다.
        
    
- 2. prior probability
    
    ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%207.png)
    
    - 사전 확률을 통해 파라미터의 분포를 가정하는데, 여기서는 평균이 0인 normal을 사용했다.
    - variance-covariance matrix $\Sigma_\Theta$는 모수의 수를 줄이기 위해 $\lambda_\Theta I$로 설정한다.

위에서 도출한 likelihood function과 prior probabilty를 활용해 아래와 같은 personalized ranking을 위한 optimization criterion인 BPR-OPT를 구할 수 있다.

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%208.png)

### 4.1.1 Analogies to AUC optimization

- 참고 - AUC
    
    [https://bskyvision.com/entry/이진-분류기-성능-평가방법-AUCarea-under-the-ROC-curve의-이해](https://bskyvision.com/entry/%EC%9D%B4%EC%A7%84-%EB%B6%84%EB%A5%98%EA%B8%B0-%EC%84%B1%EB%8A%A5-%ED%8F%89%EA%B0%80%EB%B0%A9%EB%B2%95-AUCarea-under-the-ROC-curve%EC%9D%98-%EC%9D%B4%ED%95%B4)
    
    여러개의 분류 모델의 성능을 비교하기 위해 ROC curve를 하나의 scalar 값으로 나타내기 위해 고안된 것이 AUC이다. 즉, AUC는 분류 성능을 비교하기 위한 일반적인 지표다. 
    
    ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%209.png)
    
    위의 그래프를 ROC Curve라고 하며, AUC는 이 ROC Curve 아래의 면적을 나타낸다. 
    
- user $u$에 대한 AUC는 아래와 같다
    
    ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2010.png)
    
- 따라서, average AUC는 아래와 같다.
    
    ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2011.png)
    
- 이를 $D_s$를 사용하면 아래와 같이 표현할 수 있다.
    
    ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2012.png)
    
    - $z_u$는 normalizing constant이다.
    
    ![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2013.png)
    

# 4.2 BPR Learning Algorithm

<aside>
💬 BRP에 **최적화된 학습 방법**을 제시합니다.

먼저 **Full-GD**방식이나 **item-wise&user-wise SGD**방식을 제안모델에 적용했을 때의 문제점을 발견하고
**LEARN-BPR**이라는 새로운 방법을 제안합니다.

</aside>

## 4.2.1 기존 학습 방법의 문제점

### full gradient descent

- 시간복잡도 $O(|S| * |I|)$ → **매우 오래 걸림**
- training pairs $(i,j)$의 아이템 $i$, $j$가 **비대칭** → **학습이 오래 걸림**
    - 예를 들어, 많은 유저에게 **긍정적인** 피드백을 받은 아이템 $i$가 있다고 합시다.
    - 아이템 $i$는 많은 아이템 $j$와 비교되기 때문에 많은 $\hat{x}_{uij}$가 손실 함수에서 사용됩니다.
    - 결국 $i$에 의존하는 gradient가 지배적일 것이므로 **학습률을 매우 작게** 설정해야 합니다.
    - gradient값이 크게 상이하기 때문에 **정규화도 어렵게** 됩니다.

### item-wise, user-wise SGD

- 순차적으로 학습할 경우 → **학습이 오래 걸림**
    - 예를 들어, user-item 쌍 $(u, i)$에 대해서는 많은 아이템 $j$가 $(u, i, j)$쌍으로 존재
    - 고로 순서대로 하면 같은 user-item 쌍에 대해 **연속적으로 많은 학습**이 이루어짐.

## 4.2.2 LearnBPR

- **SGD 기반**
- **uniformly** distributed bootstrap sampling
    - 순서대로 뽑지 않고 **무작위로** 뽑음
    
     →  아무 단계에서 학습을 종료할 수 있다.
    
- 의사코드
    - LearnBPR은 아래와 같이 수행한다. (전형적인 SGD의 알고리즘)
    
    ```jsx
    procedure LearnBPR(DS,Θ)
    	initialize Θ
    	repeat
    		draw (u,i,j) from Ds // uniformly random pick
    		get gradient // 위의 미분값
    		Θ <- Θ + lr * gradient
    	until convergence
    	return Θ
    end procedure
    ```
    
    $$
    \begin{align*}
    \frac{\partial BRP-OPT}{\partial \theta} &= \sum_{(u,i,j)\in D_s} \frac{\partial}{\partial \theta} ln\, \sigma(\hat{x}_{uij}) - \lambda\frac{\partial}{\partial \theta}||\theta|
    |^2
    \\
    &∝ \sum_{(u,i,j)\in D_s} \frac{-e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \frac{\partial}{\partial \theta} \hat{x}_{uij} - \lambda\theta &&
    \end{align*}
    $$
    

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2014.png)

# 4.3 Learning models with BPR

<aside>
💬 본 섹션에선 **Matrix Factorization**과 **Adaptive-kNN** 모델에서 BPR를 적용하는 방식에 대해 설명합니다.

</aside>

- 이하 모델에서 training triples을 적용할 수 없으니 아래와 같이 수정한다.

$$
\hat{x}_{uij} := \hat{x}_{ui} - \hat{x}_{uj}
$$

## 4.3.1 MF(Matrix Factorization)

- latent factor W, H에 대한 user-itme matrix $\hat{X}$를 아래와 같이 추론한다.
- $**\theta = (W, H)**$

$$
\hat{X} := WH^t\\
\hat{x}_{ui} = ⟨w_{u}, h_{i}⟩ = \sum w_{uf} \cdot h_{if}
$$

- 각 파라미터에 대한 gradient는 아래와 같이 계산한다.
- 각 항에 overfitting을 방지하기 위한 $\lambda_W, \lambda_{H+}, \lambda_{H-}$가 추가된다.

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2015.png)

## 4.3.2 ANN(Adaptive-kNN)

- 아이템 $i$와 근접한 k개의 아이템 $l$에 대한 **코사인 유사도** $c_{il}$에 대한 $\hat{x}_{ui}$를 아래와 같이 추론한다.
- $\theta = C$

$$
\hat{x}_{ui} = \sum_{l\in I^+_u ∧ l \ne i} c_{il}
$$

- 각 파라미터에 대한 gradient는 아래와 같이 계산한다.
- 각 항에 overfitting을 방지하기 위한 $\lambda_{I+}, \lambda_{I-}$가 추가된다.

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2016.png)

# 5. Relations to other methods

<aside>
💬 **제안 학습방법과 다른 두 학습방법을 간단히 비교**합니다.
독자에 따라 아래 학습방법에 대한 사전지식이 필요할 수 있습니다.

</aside>

## 5.1 Weighted Regularized Matrix Factorization (WR-MF)

- $c_{ui}$는 학습 파라미터가 아닌 (u, i) 튜플마다 미리 주어진 가중치임.
    
    🆚 LEARN-BPR: WR-MF가 더 빠르다. (미리 준비한 가중치 $c_{ui}$ 때문임)
    
- Squared Error
    
    🆚 LEARN-BPR: MLE
    

$$
\sum_{u \in U}\sum_{i \in I} c_{ui} (⟨w_u, h_i⟩-1)^2 + \lambda||W||^2_f + \lambda||H||^2_f
$$

## 5.2 Maximum Margin Matrix Factorization (MMMF)

- MF에만 한정되어 있음
    
    🆚 LEARN-BPR: 더 일반적이고 여러모델에 사용가능
    
- Explicit Dataset에 적합
    
    🆚 LEARN-BPR: Implicit Dataset에 적합
    

$$
\sum_{(u,i,j) \in D_s} max(0, 1-⟨w_u, h_i-h_j⟩) + \lambda||W||^2_f + \lambda||H||^2_f
$$

# 6. Evaluation

<aside>
💬 앞서 설명한 두개의 모델 **MF, kNN**을 다양한 학습방법으로 **학습**해보고 **성능을 비교**합니다.

모델 **MF**는 **SVD-MF, WR-MF, BPR-MF**로 학습했으며
**kNN**은 **Cosine-kNN, BPR-kNN**으로 학습했습니다.
추가로, **인기도**에 따른 결과도 같이 비교합니다.

성능지표는 AUC로 합니다.

</aside>

## 6.1 Datasets

- online shop **Rossmann** dataset
    - 10,000 users
    - 4000 items
    - 총 구매기록 426,612
    - 유저가 살 아이템 목록을 예측합니다.
- DVD rental dataset of **Netflix**
    - 10,000 users
    - 5000 items
    - 총 평가기록 565,738 (1 to 5 stars)
    - 전처리
        - implicit feedback으로 구성하기 위해서 rating score를 제거하고
        - **score가 높은 순서대로 아이템을 정렬**했습니다.
- Sub Sampling
    - 모든 item에 대해 적어도 10명의 user
    - 모든 user에 대해 적어도 10개의 item

## 6.2 Evaluation Methodology

- **leave one out evaluation**을 적용함.
    - 각 유저에 대해 하나의 user-item 쌍을 제거하는 방법
- 개인화된 ranking은 test set $S_{test}$의 **평균 AUC값**으로 평가합니다.
    - AUC는 아래와 같습니다.
    - **높은 값**은 **높은 성능**을 의미합니다.
    
    $$
    AUC = \frac{1}{|U|}\sum_u \frac{1}{|E(u)|} \sum_{(i,j)\in E(u)} δ(\hat{x}_{ui} > \hat{x}_{uj})
    $$
    
- 다양한 train/test set에서 10번 학습했습니다.
- 하이퍼파라미터는 grid search를 통해 찾아냈습니다.

## 6.3 Result and Discussion

<aside>
💬 MF, kNN 모델에 적용한 BPR의 **성능이 가장 높게 ****나왔다.

모든 MF 모델에 서로다른 학습 방법을 적용했지만 성능이 크게 다르다는 것을 알수 있었다.

</aside>

- *np_max: 개인화 되지 않은 순위 모델의 이론적 상한*

![Untitled](BPR%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2017.png)

# 7. Conclusion

- **개인화 ranking**에 대한 최적화 기준 **BPR**과 학습 알고리즘 **LEARN-BPR**을 제안했습니다.
- 두 모델 MF와 kNN에서 모두 **다른 최적화 기준보다 더 좋은 결과!**
- **예측 성능**은 모델 뿐만 아니라 **최적화 기준에도 관련이 있다.**

## 🧐 논의해볼 것

- evaluation dataset으로 [**Collaborative Filtering for Implicit Feedback Datasets**](https://www.notion.so/Collaborative-Filtering-for-Implicit-Feedback-Datasets-f7f0e9f2f6ac46ad8239208e4c856967) 에서 사용한 TV-Show 데이터 셋을 사용하면 결과가 어떻게 나올까?
- implicit feedback에는 positive한 정보만 있다고 하는데, **반품이나 구매취소** 같은 데이터는 negative feedback으로 보고 이를 반영하는 모델은 없을까?