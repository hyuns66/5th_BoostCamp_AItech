# Preview

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/c7714c77-fb08-45fc-823e-021675954e53" width=50%></center>
조건부 확률 $P(A|B)$ : B가 일어났을 때 A 가 일어날 확률

조건부 확률을 다룰 때는 **모든 상황을 가정하에 두고** 다루게 된다. 위 확률은 B가 일어났을 때 A가 일어날 확률이지만 B가 일어난 상황 또한 가정이므로 B가 일어날 확률을 곱해주어야 교집합 확률을 구할 수 있다.

❓ **베이즈 정리가 뭔디??**  
우리가 구하고 싶은거는 어떤 사건이 일어났을 때, 즉 데이터가 주어진 상태에서의 확률을 구하고 싶은것.
*조건부 확률 $P(A|B)$ : B가 일어났을 때 A가 일어날 확률* 을 알고 있다면
**이지만 A가 실제로 일어났을 때 그게 진짜로 B에 의해 일어났는지를 검증하고 싶다는 것이다.

# 정의

$$
P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}
$$

$P(X|Y)$ (posterior) : 구하고싶은 사후확률 (Y가 일어났는데 X에 의해 일어났을 확률)

$P(Y|X)$ (likelyhood) : X의 경우 Y가 일어날 조건부 확률 (우도)

$P(X)$ (prior) : X 조건이 차지하는 사전확률

$Y$ (evidence) : 지금 일어난 사건Y가 일어날 확률 

**evidence 를 구하는 공식**

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/237f2d65-d3c8-4dfc-aff8-9a6f1fd73443" width=35%></center>

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/066a8127-c8e0-4100-a307-27a593684a85" width=90%></center>

evidence를 구하려면 일어나는 사건에 대한 두 가지 likelyhood(긍정, 부정)가 모두 필요하다.
<br><br>

## **최소오류 베이지안 분류기**

베이스정리에 의해 사후확률을 구하고, 높은쪽으로 의사결정


❓ $X$ 이벤트가 발생한 시점에서  
$P(w_1|X)>P(w_2|X)$이면 $X$를 $w_1$로 분류하고  
$P(w_1|X)<P(w_2|X)$이면 $X$를 $w_2$로 분류한다


사전확률은 데이터 표본 수에 따른 추정값을 사용함

우도값은 정확한 조건부확률을 구해야 하므로 중요한 문제가 됨

<br>

## **최소 위험 베이지안 분류기**

손실 행렬을 기반으로, 잘못분류했을 때의 손실이 가장 적은 쪽으로 의사결정


❓ $q_1 = c_{11}p(X|w_1)P(w_1) + c_{21}p(X|w_2)P(w_2)$  
$q_2 = c_{12}p(X|w_1)P(w_1) + c_{22}p(X|w_2)P(w_2)$  
$q_2>q_1$ 이면 $X$를 $w_1$로 분류하고   
$q_2<q_1$이면 $X$를 $w_2$로 분류한다  

<br>  

- 손실행렬 (Zero-One Loss)
    
    $C = \begin{bmatrix} c_{11}&c_{12}\\c_{21}&c_{22}\\ \end{bmatrix} = \begin{bmatrix} 0&1\\1&0\\ \end{bmatrix}$
    
    **Note** : 보통 손실행렬은 특정 도메인에 대한 지식을 가지고 전문가가 작성해야 하지만, 지식이 없다면 위와같은 Zero-One Loss 행렬을 사용한다.
    

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/d0e3963b-930b-4858-81f9-e01b6e86bfbd" width = 70%></center>

<br>  
Zero-One Loss의 경우 결과론적으로 최소오류 베이지안 추론과 동일한 의사결정을 하게된다.

## 일반화

베이지안 분류문제를 M차원으로 확장하고 식별함수로 일반화 하면 아래와 같이 나타내기 가능

$$
X\,를\;k=arg\,max\;g_i(X) 일때\, w_k로\, 분류한다.\\
g_i(X)=\left\{\begin{array}{c} p(X|w_i)P(w_i)\quad\quad(최소오류) \\ \frac{1}{\sum\limits_{j=1}^Mc_{ji}p(X|w_j)P(w_j)}\quad\quad(최소위험) \end{array} \right.
$$
<br> <br>

# 베이즈 정리를 활용한 정보의 갱신

검증을 여러번 거친다고 할 시 이전에 계산된 사후확률을 다음 차례 검증의 사전확률로 사용하여 갱신된 사후확률을 구할 수 있다.

여러번 갱신을 거칠 수록 높은 정밀도가 나오게 된다.

ex) 정밀도가 낮은 검사라도 여러번의 검사를 통해 같은 결과가 나온다면 신뢰할 만 한 데이터가 됨