# 선형 방정식

$$
a_1x_1 +a_2x_2+a_3x_x+...+a_nx_n=b
$$

형태의 방정식을 선형방정식 (Linear Equation)이라 부른다.

이 때 각 계수들과 x항을 하나의 벡터로 표현해 보면

$$
\mathbf{a} = \left[\space
\begin{matrix}
    a_1 \\
    a_2 \\
    a_3 \\
    ... \\
    a_n \\
\end{matrix}
\space\right] ,\quad
\mathbf{x} = \left[\space
\begin{matrix}
    x_1 \\
    x_2 \\
    x_3 \\
    ... \\
    x_n \\
\end{matrix}
\space\right]
$$

이므로 이를 행렬곱 연산으로 표현식을 바꿔보면 아래와 같이 나타낼 수 있다.

$$
\mathbf{a}^T\mathbf{x}=b
$$

# Identity Matrix (항등행렬, 단위행렬)

$$
\mathbf{I} = \left[\space
\begin{matrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1 \\
\end{matrix}\space\right]
$$

이렇게 생긴걸 항등행렬 또는 단위행렬이라 부른다.

**why?)** 무슨 벡터를 곱하던 간에 결과는 같은 벡터가 나오므로 **항등행렬** 이라 부름

$$
n차원\space벡터\space \mathbf{x}가\space 있을 \space때 \\
I_n\mathbf{x}=\mathbf{x}
$$

# Inverse Matrix (역행렬)

일반적으로 $n\times n$ 행렬인 square matrix 에서만 이야기를 한다.

**정의**

> $n\times n$ 행렬 $A$가 있을 때 곱해서 항등행렬이 나오도록 만드는 행렬 $A^{-1}$을 $A$의 역행렬 이라 함
> 

$$
A^{-1}A=AA^{-1}=I_n
$$

**rectangular matrix에서의 역행렬**

$n\times m$ 행렬의 경우 $m\times n$ 사이즈의 행렬을 곱해서 항등행렬이 나오도록 하는 역행렬을 구할 수 있지만, 정사각 행렬처럼 $A^{-1}A=AA^{-1}=I_n$ 이 성립되지는 않고 좌변에 곱하냐, 우변에 곱하냐 둘 중 하나의 등식만 성립하게 된다.

$B_{nm}B_{mn}$ 은 $n\times n$  사이즈의 행렬이 나오고 $B_{mn}B_{nm}$은 $m\times m$ 사이즈의 행렬이 나오기 때문

## 역행렬이 왜 중요한데??

어떠한 상황에서 [선형방정식](https://www.notion.so/Linear-System-1-b47dafada41e4e389419b406f2eb7e55)의 형태로 표현된 연립방정식들을 행렬곱으로 나타내면 아래와 같다.

$$
A\mathbf{x}=\mathbf{b}
$$

- A : 주어진 변수들이 담긴 행렬
- x : 구해야 하는 벡터
- b : 결과값이 담긴 벡터 (사실은 역행렬을 취해준 형태)

여하튼 저런 행렬방정식이 있을 때 역행렬 $A^{-1}$을 알 수 있으면 아래처럼 바로 x를 구할 수 있다.

$$
A\mathbf{x}=\mathbf{b}\\
A^{-1}A\mathbf{x}=A^{-1}\mathbf{b}\\
I_n\mathbf{x}=A^{-1}\mathbf{b}\\
\mathbf{x}=A^{-1}\mathbf{b}
$$

# 행렬방정식(Matrix Equation)의 해석

<img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/482f1e44-715e-4ced-b242-1e8507d1cf64">  
<br><br>

위와 같은 데이터가 있다고 해보자. 우리의 상황은 저 데이터들이 선형시스템에 존재하는 데이터라고 가정했을 시 weight, height, is_smoking 변수들에 일정한 가중치를 곱해 life_span을 예측하고 싶은것이다. 그 일정한 가중치를 찾는 과정이 딥러닝인데, 일단 간단한 예제로 직접 풀어보자

<br>

## 벡터 방정식 (Vector Equation)

일단 위 데이터를 행렬식으로 나타내면 아래와 같다.

$$
\left[\space
\begin{matrix}
    60 & 5.5 & 1\\
    65 & 5.0 & 0 \\
    55 & 6.0 & 1 \\
\end{matrix}\space\right]
\left[\space
\begin{matrix}
    x_1\\
    x_2 \\
    x_3 \\
\end{matrix}\space\right]=
\left[\space
\begin{matrix}
    66\\
    74 \\
    78 \\
\end{matrix}\space\right]\\
A\mathbf{x}=\mathbf{b}

$$

이걸 풀어서 x1, x2, x3 항으로 묶어보면 아래와 같은 식으로 다시 나타낼 수 있다.

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/5e8ea218-49bd-4a82-91f6-538b8ec75d7d" width=80%></center>  
<br>

이 때 $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ 들을 3차원 공간상의 벡터라고 볼 수 있기 때문에 이를 ***벡터 방정식***이라고 부른다.

**즉, 선형 시스템에서의 가중치를 구하기 위한 행렬 방정식은 벡터 방정식으로 나타낼 수 있다.**

## 일반적인 행렬방정식에서의 변환

[위 식](https://www.notion.so/Linear-System-1-b47dafada41e4e389419b406f2eb7e55)은 일반적인 행렬 방정식이라고 보기 어렵다. 행렬 x 행렬이 아닌 행렬 x 벡터의 형태이기 때문이다.

$$
\left[\space
\begin{matrix}
    1 & 1 & 0\\
    1 & 0 & 1 \\
    1 & -1 & 1 \\
\end{matrix}\space\right]
\left[\space
\begin{matrix}
    1 & -1\\
    2 & 0 \\
    3 & 1 \\
\end{matrix}\space\right]=
\left[\space
\begin{matrix}
    x_1 & y_1\\
    x_2 & y_2 \\
    x_3 & y_3 \\
\end{matrix}\space\right]=
\left[\space
\begin{matrix}
    \mathbf{x}\space\mathbf{y}
\end{matrix}\space\right]
$$

를 벡터방정식으로 표현하면

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/a6995242-a0fa-44f9-909d-525d1cc43a42" width=80%></center>

처럼 $\mathbf{x}, \mathbf{y}$를 따로 구할 수 있다.

즉 왼쪽 행렬(재료 벡터들의 집합)과 오른쪽 행렬(계수들의 집합) 들의 **선형결합**으로 최종 행렬  $\mathbf{x}, \mathbf{y}$를 표현한다고 말한다.

위 식을 $A\mathbf{x}=\mathbf{b}$ 라고 했을 때 전체를 전치시키면 Row combination 형식으로도 표현이 가능하다.

$(A\mathbf{x})^T=\mathbf{x}^TA^T$이므로 행렬식으로 표현하면 아래처럼 되고

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/a028bad9-1e04-454e-a6cd-e7b34def33e0" width=80%></center>


이걸 collumn 기준이 아닌 row를 기준으로 잡아 **선형결합** 형식으로 표현하면 아래처럼 변환이 가능하다.

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/cd5d777b-3344-4857-adcd-57ae85e634da" width=80%></center>

Collumn combination 형식과는 반대로 오른쪽 행렬(재료 row벡터들의 집합)과 왼쪽 행렬(계수들의 집합)들의 선형결합이다.

# 선형 독립과 선형 종속

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/64c2b143-72b6-4953-b9a5-b1550d5ebb87" width=80%></center>

위에서 예시로 들었던 행렬식을 가지고 선형독립과 선형종속에 대하여 설명해보면 아래와 같다.

> $\mathbf{b}$ 가 $Span\{\mathbf{a_1}, \mathbf{a_2}, \mathbf{a_3}\}$ 에 포함될 때 에만 solution이 존재하며
$\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ 이 선형독립 (Linearly Independent)일 때 위의 식 $A\mathbf{x}=\mathbf{b}$ 은 유일한 하나의 해를 갖는다.
$\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ 이 선형종속 (Linearly Dependent)이라면 위의 식 $A\mathbf{x}=\mathbf{b}$ 은 무한한 여러개의 해를 갖는다.
> 

**선형 독립인 경우**

- 주어진 벡터들 중 하나가 다른 벡터로 이루어진 Span 안에 포함되는 경우
- b를 0으로 채워진 벡터로 놓고 해를 구해보았을 때 자명해만 존재하는 경우
    - 다른 해도 존재한다면 하나의 벡터를 다른 벡터들의 선형결합으로 표현가능하다는 뜻이고 이건 위의 케이스에 해당되기 대문
- 하나의 벡터가 다른 벡터의 상수배로 표현되는 경우가 없어야 함

예를들어 주어지는 벡터들이 3차원 벡터인데 4개가 주어지면 이 벡터들은 선형종속이다.

→ 이미 3개의 벡터만으로 모든 3차원 공간을 span 할 수 있는데 그 공간 안에 포함되는 벡터가 추가되었으므로.

그렇다면 주어지는 벡터들이 3차원 벡터인데, 2개만 주어졌다면?

→ $\mathbf{b}$ 가 $Span\{\mathbf{a_1}, \mathbf{a_2}, \mathbf{a_3}\}$ 에 포함된다고 했을 때 이 벡터들은 선형독립이 될 수 있다. 그러나 두 벡터들이 서로의 상수배로 이루어지면 선형종속이다.

## ****부분공간의 기저와 차원****

n차원 공간 $R^n$ 에서 벡터를 몇개 뽑아내어 span한 결과 생긴 공간을 collumn space 라고 부르며 이것은 부분공간의 정의에 속한다.

> 부분공간의 정의
공집합이 아닌 $R^n$ 상의 벡터들의 집합이 스칼라곱과 덧셈에 의해 닫혀있다면 (선형 결합되어 있다면) 이를 $R^n$ 상의 부분공간이라 부른다.
> 

+) collumn space를 이루는 벡터들의 집합은 unique 하지 않다. (공간상의 다양한 벡터들의 조합으로 가능하다.)

**rank**

> 행렬이 가지는 independent한 collumn의 수
= collumn space의 dimension (row space의 dimension)
> 

m*n 행렬에서의 rank의 최댓값은 min(m, n) 이다.

- rank-deficient : 최대 rank보다 작은 경우
- full-row-rank : min(m, n) = n이고 rank = n인 경우
- full-collumn-rank : min(m, n) = m이고 rank = m인 경우

[[선대] 2-9강. rank (행렬의 계수)](https://www.youtube.com/watch?v=HMST0Yc7EXE&list=PL_iJu012NOxdZDxoGsYidMf2_bERIQaP0&index=11)

- rank가 collumn(row) space의 dimension인 이유
    
    <img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/9a6893ea-c77b-40dd-9dcb-542d41c174b2">
    
    좌측의 collumn space A 를 보면 벡터 [2, 1, 1]은 벡터 [1, 1, 0] + [1, 0, 1] (선형결합)으로 표현되므로 선형독립 조건에 부합하지 않다. 따라서 [2, 1, 1]을 제거하고 기저벡터 [1, 1, 0], [1, 0, 1] 로만 표현한 collumn space가 같은 A 를 나타내게 되고 최종적으로 basis 벡터들도 모두 independent 해졌으므로 이 행렬의 dimension이 rank 라고 볼 수 있다. 
    → **Col A는 2개의 벡터로 이루어진 span영역 이므로 dimension = 2가 되고 rank 역시 2가 된다.**