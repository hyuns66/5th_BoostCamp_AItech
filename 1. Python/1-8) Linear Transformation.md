# 선형변환

> 어떠한 변환 $T$가 linear 라고 하면 $T(c\mathbf{u}+d\mathbf{v})=cT(\mathbf{u})+dT(\mathbf{v})$ 가 모든 $T$ 도메인의 벡터 $\mathbf{u}, \mathbf{v}$와 모든 스칼라값 $c, d$ 에 대하여 성립한다.
> 

→ 좀 더 풀어써보면

> 두개의 input 벡터를 선형결합 하고 $T$ 변환을 한 결과와, 두 개의 input 벡터를 각각 $T$ 변환 하고 선형결합 한 결과가 같으면 $T$ 를 선형 변환이라고 한다.
> 

# 선형 변환은 행렬이다?

일단 변환 $T$ 가 선형 변환이라고 가정을 하고 시작해 보자.

standard basis vector (표준 기저 벡터) 들에 대한 변환 $T$ 를 알 수 있으므로

$$
T(\left[\space
\begin{matrix}
    1\\0
\end{matrix}\space\right]) =
\left[\space
\begin{matrix}
    2\\-1\\1
\end{matrix}\space\right],
\quad\quad
T(\left[\space
\begin{matrix}
    0\\1
\end{matrix}\space\right]) =
\left[\space
\begin{matrix}
    0\\1\\2
\end{matrix}\space\right]
$$

정의역의 벡터 $\mathbf{x}$ 에 대한 선형변환 $T(\mathbf{x})$ 를 위 standard basis vector들의 선형변환 결과들로 나타낼 수 있다.

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/e3015a87-ff9b-45a8-ad52-1cb4784bd021" width = 70%></center>

마지막 방정식이 이해가 안되면 [여기](https://www.notion.so/Linear-System-128cdbf803794c7da16f2066710e3624)를 보고 오면 된다.

벡터 방정식을 행렬 방정식으로 바꿔버리니까 입력 벡터 $\mathbf{x}$에 대한 선형변환 $T(\mathbf{x})$ 가  $\mathbf{x}$에 대한 행렬식이 되어버렸다.

즉 **선형 변환은 행렬로 나타낼 수 있다**는 말이 증명이 된 것이다.

***일반화를 해보면***

> $R^n$ →$R^m$ 으로의 변환 $T$ 가 선형 변환일 때 이 선형 변환 $T$ 는 행렬과 벡터의 곱으로 나타낼 수 있고,
> 
> 
> $$
> T(\mathbf{x})=A\mathbf{x} \quad for \space all\space \mathbf{x}\space in\space R^n
> $$
> 
> 이 때 행렬 $A$ 는 $R^n$ 영역 안에 있는 standard basis vector 들의 선형변환 $T$ 로 이루어진 벡터들의 집합으로 구성되며 **standard matrix** 라고 부른다.
>