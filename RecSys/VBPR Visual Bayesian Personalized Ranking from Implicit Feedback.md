# VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback

Keywords: Implicit Feedback, RecSys, Visual Feature
담당자: 박동연, 강잔미
발표일: 2023/04/05

**🔥 색상 별 의미 🔥**

- **파란색** → 알면 좋을 내용 및 요약
- **노란색** → 중요한 내용
- **초록색** → 작성자의 의견 (이자 뇌피셜)

# 😎 몸에 좋은 한 줄 요약

<aside>
📌 Implicit Dataset을 이용한 Personalized Ranking Task에서 Visual Feature는 유용하게 사용될 수 있고, VBPR는 이를 기반으로 cold start 문제를 어느정도 해결했고, 성능도 높였다!

</aside>

# 1. Introduction

## 1-1. Background

- 추천 시스템은 historcial feedback을 토대로 **개인화된 맞춤 제안**을 해줌
- 기존 추천 시스템
    - 그 중 대표적인 MF(Matrix Factorization) 방법론은 잠재 요소(latent factor)를 사용하는 것
    - 하지만 sparse한 데이터에서 **cold start issue problem**이 있음!
    - 또한 기존 RecSys들은 중요한 정보인 **Visual Data의 feature**를 잘 사용하지 않았음

## 1-2. Goal of this paper

1. **Cold Start Issue** 해결하기
2. **시각적 정보**를 추천 시스템 관점에서 활용하기
3. 유저에게 맞춤화된 **아이템 랭킹 추천**하기

# 2. VBPR

<aside>
💬 VBPR 모델의 구조와 BPR 기반의 학습 방법을 설명합니다!

</aside>

## 2.1 간단하게 살펴보는 모델 구조

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled.png)

1. 사전학습 된 CNN을 통해서 Visual Feature을 뽑아낸다 — cold start 문제 해결!
    - 어떤 CNN을 쓴건가요? — AlexNet
        - AlexNet을 사용
        - 마지막의 Fully Connected Layer의 반환 데이터를 Concat하게 이어서, 4096($F$)차원의 $f_i$ visual feature vector로 반환
        
        ![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%201.png)
        
2. 4096차원의 고차원 Visual Feature을 저차원 D차원의 Item Visual Factor로 임베딩한다
3. Non-Visual F차원의 Item Latent Factor과 Item Visual Factor을 Concat하여 Item Factor로 생성
4. User Factor와 Item Factor, Biases을 사용하여 Prediction Score 도출

## 2.2 수식으로 보는 모델

### 2.1.1. Notation

![깔쌈하게 Notation 정리해주는 논문.. 최고의 논문..](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%202.png)

깔쌈하게 Notation 정리해주는 논문.. 최고의 논문..

### 2.1.2 수식의 변천사

1. **우리가 알고 있는 MF 기반의 기본 수식**
    - 유저의 latent vector와 아이템의 latent vector을 내적
    - 그리고 이에 각 유저들의 편향을 반영하기 위한 bias와 global offset을 더함
    - 하지만 이는 **cold start issue**가 있음 — 데이터가 적을 때 관계를 유추하기 어려워짐

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%203.png)

1. **Cold start issue를 해결하기 위한 명시적 데이터 추가**
    - 명시적(Explicit) 데이터에 속하는 유저와 아이템의 **Visual Interaction 정보($\theta^T_u \theta_i$)를 추가**하여 cold start 이슈를 해결하였음!
        - CNN이 이미지의 전반적인 특징을 학습하기 때문에 visual feature 안에 좀 더 광범위한 정보가 담겨있을 것
        - 이러한 visual feature을 CNN으로 추출해서 활용하면 처음보는 데이터에 대해서도 잘 동작할 것이라고 기대!
    - $\theta^T_u$$\theta_i$: u와 i 사이의 시각적 상호작용 — 각 아이템에 얼마나 끌리는지 (내적)
    - $\theta^T_u$: 유저 u의 visual factor (D×1)
    - $\theta_i$: 아이템 i의 visual factor (D×1 = (D×F) (F×1)) —  CNN으로 뽑아낸 아이템 시각적 특징

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%204.png)

1. **아이템 i의 visual factor의 차원을 임베딩**을 통해 낮추자!
    - 이 때 아이템 i의 visual factor인  $\theta_i$ 는 $Ef_i$ (Dx1)로 임베딩하여 차원을 낮출 수 있다!
        - E는 임베딩 행렬 (D×F)
        - $f_i$는 DeepCNN을 거쳐서 생성된 visual factor (F×1)
    - 여기에 아이템의 visual appearance에 대한 전반적인 유저의 의견 반영을 위해 편향을 곱해서 더해준다고 함!

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%205.png)

## 2.3 모델 학습은 어떻게 하나요? — BPR을 사용합니다!

### 2.3.1 BPR이란 무엇인가요?

**BPR(Bayesian Personalized Ranking)**: SGA로 학습하는 pairwise ranking optimization framework

### 2.3.2 BPR은 어떻게 학습하나요?

- $u$: (user) 유저
- $i$: (positive feedback) 유저가 긍정적 피드백을 준 아이템
- $j$: (non-observed) 유저가 본 적 없는 아이템

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%206.png)

- BPR 모델에서의 최적화 수식
    - $\hat x_{u,i,j} = \hat x_{u,i} - \hat x_{u,j}$
    - $ln\sigma$ : Logistic Sigmoid 함수
    - $\lambda_\theta$: Regularization Term
    - $\theta$ : $(u, i, j)$ 간의 상호작용을 파라미터화

![BPR 모델의 최적화 수식과 가중치 업데이트 방법](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%207.png)

BPR 모델의 최적화 수식과 가중치 업데이트 방법

### 2.3.3 BPR에 기반한 VBPR은 어떻게 학습하나요?

- BPR의 최적화 수식을 기반으로 VBPR에서는 아래와 같이 파라미터들을 업데이트 합니다!
- 자세한 풀이 과정은 아래와 같습니다!
    
    ![자세한 미분과정의 리뷰는 셀프.. 😉 ](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%208.png)
    
    자세한 미분과정의 리뷰는 셀프.. 😉 
    

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%209.png)

## 2.4 Scalability 문제는 없나요?

- $F$: Deep CNN을 통해서 나온 visual feture dimension (4096)
- $K$: latent factor의 차원
- $D$: visual factor의 차원
- $O(K) + O(D×F)
= O(K) + O(D)$
$**= O(K+D)$**
(F는 4096 고정 차원이라서 상수 취급해서 O(DxF) → O(D)로 표기하는 듯 함)
- 그래도 **Linear한 정도**라서, 데이터가 추가되더라도 원래의 BPR에 비해서 시간과 메모리가 비약적으로 소모되지는 않음! → **문제 없다**는 뜻 🤪

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2010.png)

# 3. Experiments

<aside>
💬 앞서 소개한 모델(VBPR)을 현실 데이터에 적용해본 후 , 다른 모델과의 성능을 비교합니다.

</aside>

## 3.1 **Datasets**

### 3.1.1 **Data that we used**

1. Amazon.com
    1. Women’s and Men’s Clothing : **visual features가 의미있다**고 여겨진다.
    2. Cell Phones and Accessories : **visual features가 큰 영향을 미치지는 않으나**, 어느정도 중요한 역할을 할 것으로 예상된다.
2. Tradesy.com
    1. 구매이력, 좋아요(’thumbs-up’)를 제공하므로 이를 Positive feedback으로 사용한다.
    2. 중고마켓의 특성인 일회성 거래로 인해 $Cold\ Start$ ****문제가 존재한다.

### 3.1.2 **About Dataset**

- 앞서 언급한 Implicit feedback과 Visual features  $fi$ 로 부터 추출된 Dataset
- $|I^{+}_{u}| < 5$ 인 user는 제외한다.

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2011.png)

## 3.2 **Visual Features**

- 각 아이템  i에서 하나의 상품 이미지를 모으고 CNN 구조를 구현한 **caffe reference model**을 사용해 visual features $fi$ 를 추출한다.
- **사용한 Caffe reference model**
    - 5 convolutional layers followed by 3 fully-connected layers.
    - pre-trained on 1.2 million ImageNet(ILSVRC2010) images.
- 해당 실험에서는 second fully-connected layer의 output을 사용하며, 그로부터 F = 4096 dimensional visual feature vector $fi$를 얻는다.

## 3.3 **Evaluation Methodology**

- 각 $user$로부터 랜덤하게 $item$를 선택하여 ${\nu}_{u}$(validation)와  $\tau_{u}$(testing)를 정하고 남은 모든 데이터는 $p_{u}$ (training)으로 사용한다.
- 평가 지표로 **AUC**를 ****사용한다.
    
    ![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2012.png)
    
    - $\delta(b)$ : 지시함수, $i$: observed, $j$: not-observed
    - $E(u) = \{(i,j)|(u,i)\in \tau_{u}\ \wedge(u,j)\notin(p_{u}\cup\nu_{u}\cup\tau_{u}) \}$

## 3.4 **Baselines**

- 비교 가능한 visual-aware MF 방법이 없기 때문에 state-of-the-art **MF model**과 주로 비교할 것이며, 추가로 최근에 제안된 **content-based method**와도 비교하도록 한다. 사용한 baseline은 다음과 같다.
    - **Random (RAND)**
        
        모든 user에 대해 무작위로 item 순위를 정한다.
        
    - **MostPopular(MP)**
        
        item의 인기도에 기반해 순위를 정하며 개인화된 것은 아니다.
        
    - **MM-MF**
        
        Pairwise MF model로 $x_{uij}$에서의 hinge ranking loss 최적화하고, BPR-MF에서와 같이 SGA를 사용해 학습한다.
        
    - **BPR-MF**
        
        Pairwise method로 implicit feedback datasets에 대한 최신의 personalized ranking 방식이다.
        
    - **Image-based Recommendation (IBR)**
        - content-based’ baseline으로 문제 설정과 데이터 측면에서 차이가 있지만 visual data를 사용하는 방법이므로 함께 비교하도록 한다.
        - 해당 모델은 feedback을 사용하지 않으며, items과 input 사이의 관계를 인코딩한 그래프를 사용한다.
        - 해당 모델은 visual space에 대해 학습하여 query image와 stylistically 유사한 item을 찾는다.
    
- 공정한 비교를 위해 모든 MF based method에서 동일한 차원수를 사용한다.
- visual and non-visual dimensions: fixed to a **fifty-fifty split** for simplicity
- 모든 **hyperparameter는 validation set에서 가장 좋은 성능을 보인 것을 사용**했으며, 다음과 같다.
    - On Amazon dataset [ BPR-MF, MM-MF, VBPR ] : $**\lambda_{\theta}$ = 10**
    - On Tradesy.com [ BPR-MF, VBPR ] : $**\lambda_{\theta}$ = 0.1** ,  MM-MF : $**\lambda_{\theta}$ = 1**
    - VBPR → $**\lambda_{E}$ = 0**
    - IBR → rank of Mahalanobis transform = 100 (very well on Amazon data)

## 3.5 **Performance**

### **3.5.1 Result**

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2013.png)

- $All\ Items$ : Full test set $\tau$
- $Cold\ start$ : subset of $\tau$ which only consists of **items that had fewer than five positive feedback** instances in the training set.
    1. around **60%** of the test set for the two Amazon datasets.
    2. **80%** for Tradesy dataset
    
    sparse real-world datasets에서 모델이 괜찮은(acceptable) 성능을 갖기 위해 본질적인 cold start를 해결하고 항목을 정확하게 추천해야 한다는 것을 뜻한다.
    
    → VBPR이 다른 baseline과 비교해 cold start를 어느정도 해소해주긴 하지만 딱 수치만 놓고 봤을 때 좋은 성능을 내지 못한다. 즉, 완전한 해결책이 될 수 없음을 이야기하고 싶어서 일부터 cold start case를 만들고 성능 비교를 한 것 같다.
    

🔥 **핵심 결과** 🔥

⇒ MF와 content-based methods의 강점을 결합한 **VBPR은 대부분의 case에서 가장 좋은 성능**을 보인다.

1. BPR-MF과 비교해 평균적으로 VBPR이 $All\ Items$에서 **약 12%** 그리고 $Cold\ start$ 에서 ******약 28%****** 향상된 성능을 보이며, 이것은 **ranking task에 CNN features 포함하는 것의 상당한 이점**을 보여준다. 
2. $Cold\ start$ 에서는 IBR이 MF methods(BPR-MF & MM-MF)보다 성능이 좋은 반면, $Warm\ start$ 에서는 그 반대이다.
    1. pure MF method은 cold start case에서 meaningful factors 학습이 어렵다.
    2. IBR은 historical user feedback에 대해 학습하지 않는다.
3. 특히, 일회성 거래로 인한 **$Cold\ start$ 문제**가 있는 Tradesy dataset에서 **VBPR이 타 모델과 차이**를 보인다.
4. Visual features는 Cellphone datasets보다 **Clothing datasets에서 더 큰 benefit**을 갖는다.
    1. 상대적으로 Cellphone을 선택할 때, Visual features의 역할이 작기 때문으로 추측할 수 있다. 
5. cold items는 본질적으로 ‘unpopular’이기 때문에 Popularity-based methods은 성능이 좋지 않다.

(+) Pair-wise method(VBPR)가 Point-wise method(WRMF)보다 성능이 좋게 나옴 

(VBPR beats WRMF by **14.3%** for all items and **20.3%** for cold start items)

<aside>
💡 **point-wise vs pair-wise** (in Releated Work of this paper)

**point-wise는 Loss function에서 한번에 하나의 아이템만 고려 (WRMF)**
: 하나의 Query(User) — 1개의 Item
: non-observed Item을 Negative로 취급

**pair-wise는 Loss function에서 한번에 2개의 아이템을 고려 (VBPR)**
: 1개의 Postivie Item, 1개의 Negative Item
: 관측된 아이템(Positive)은 관찰되지 않은 아이템(Non-observed)보다 더 선호된다는 것들 가정

</aside>

### **3.5.2 Sensitivity**

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2014.png)

⇒ [ MM-MF, BPR-MF, VBPR ] : **factor의 수가 증가할수록 성능이 좋아지는** 것을 확인할 수 있고 이는 pair-wise method에서 overfitting을 피하는 능력을 보여준다.

### **3.5.3 Training Efficiency**

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2015.png)

⇒ VBPR이 [ MM-MF, BPR-MF ]과 비교해 최적의 training iterations 값으로 수렴하는데 더 오래 걸리지만 가장 큰 dataset에서 약 3.5시간정도 걸리므로 여전히 **효율적**이라고 볼 수 있다.

### **3.5.4 Visualizing Visual Space**

![Untitled](VBPR%20Visual%20Bayesian%20Personalized%20Ranking%20from%20Implicit%20Feedback/Untitled%2016.png)

1. 다른 dataset에서 pre-train된 CNN모델에서 추출한 visual feature이지만, 임베딩을 사용하여 추출된 features의 표현력을 확인할 수 있는 다양한 subcategories에 대한 ‘visual’ transition(loosely)을 학습할 수 있다.
2. VBPR은 hidden taxonomy학습을 돕고, 가장 관련성이 높은 underlying visual dimensions을 찾아 item과 user를 uncovered space으로 맵핑한다.

# 4. **Conclusion**

- **implicit feedback datasets의 personalized ranking task에서 시각적 특성이 갖는 유용함**에 대해 분석.
- 사람의 행동에 가장 많은 영향을 미치는 ‘visual dimensions’을 찾기 위해 **상품 이미지에서 추출된 visual features를 Matrix Factorization에 결합한 확장 가능한 모델 VPBR**을 제안 (모델은 SGA를 사용한 BPR로 학습)
- VBPR 모델은 cold start issue를 **어느정도 해결**했다 (덜어낸 정도..)

# 🧐 논의해볼 것(from Future Work)

- 시간역학으로 모델을 확장하여 시간에 따른 패션의 취향의 흐름을 설명할 수 있을까?
- e************xplicit feedback************에 VBPR을 적용해볼 수 있을까?
