# 벡터와 행렬, 그리고 경사하강법

# 벡터의 노름(norm)

> 일반적인 2차원 평면상에서의 거리는 그냥 절댓값 구해버리면 되지만, 벡터는 d차원의 공간 상에서의 한 점을 의미하기에 단순히 절댓값으로 생각하면 안된다.
norm은 곧 원점으로부터 벡터가 나타내는 점의 거리를 고차원 공간까지 확장하여 일반화한 수식이다.
> 
- 원점에서부터의 거리를 말한다.
- L1 노름, L2 노름 두 종류가 있음

## L-1 norm

> 각 성분의 변화량의 절댓값을 모두 더한 거리
> 


$$
||\mathbf{x}||_1 = \sum_{i = 1}^{d}{|x_i|}
$$
<center><img src="https://github.com/hyuns66/hyuns66/assets/72616557/da12615f-73c2-4b44-86d0-606ca3b3faf7" width="50%"/></center>


## L-2 norm

> 피타고라스 정리를 이용한 유클리드 거리
> 

$$
||\mathbf{x}||_2 = \sqrt{\sum_{i = 1}^{d}{|x_i|^2}}
$$

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/3b8504d9-6e36-496e-82d4-f86aa24d0605" width="50%"></center>


위에서는 2차원 평면공간을 예시로 들었지만, 실제로는 모든 d차원의 공간에서 계산이 가능하다.

## norm을 구하는 python 코드

```python
import numpy as np

"""
아래 함수들에 들어가는 인자 x는 배열로 나타내어진 벡터이므로 numpy 라이브러리를 사용한
행렬 연산이 이루어진다.
"""

def l1_norm(x):
	x_norm = np.abs(x)
	x_norm = np.sum(x_norm)
	return x_norm

def l2_norm(x):
	x_norm = x*x
	x_norm = np.sum(x_norm)
	x_norm = np.sqrt(x_norm)
	return x_norm
```

# 행렬

- 벡터를 원소로 가지는 2차원 배열
- 벡터는 d차원 공간에서의 점을 나타내므로 행렬은 d차원 공간상의 점들의 집합이다.
- 각각의 행벡터는 데이터를 의미한다.

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/3cf1e433-e8f1-464e-9028-df52db29d07b" width=70%></center>

<br>
<br>

## 연산자로써의 행렬

- 행렬은 벡터공간에서 사용되는 연산자로써 이해할 수 있다.

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/7eb1aa37-226f-4eff-9739-18b0197b21db" width=70%></center>

n차원 공간의 벡터 $\mathbf{z}$와 m차원 공간의 벡터 $\mathbf{x}$가 있을 때 n*m 행렬을 곱해주게 되면 $\mathbf{x}$ 벡터를 $\mathbf{z}$ 벡터의 차원으로 보낼 수 있다.

이는 **패턴을 추출**하거나 **데이터를 압축**하는데 쓰일 수 있다는 의미이기도 하다.
<br>
<br>
<br>
# 경사 하강법

**핵심 아이디어**

> **함수의 현재 위치에서 미분값을 더하면 함수값이 증가하고, 미분값을 빼면 함수값이 감소한다.**
→ 만약 감소중인 함수여서 현재 위치의 미분값이 음수가 나왔다면, 현재위치에서 음수를 더해 뒤로가면 함수값이 증가할 것이고, 증가중인 함수여서 현재위치의 미분값이 양수가 나왔다면 현재 위치로부터 양의 방향으로 이동했을 때 함수값이 증가하게 된다.
함수값이 감소하는 경우도 같은 방법으로 설명 가능하다.
> 

- 경사 상승법 : 미분값을 더해가면서 함수값이 증가하는 방향으로 이동하는 방법, 극대값의 위치를 구할 때 사용
- 경사 하강법 : 미분값을 빼가면서 함수값이 감소하는 방향으로 이동하는 방법,  극소값의 위치를 구할 때 사용

→ 경사 상승이던 하강이던 극값에 도달하여 미분값이 0이 되면 더이상 업데이트가 일어나지 않는다.
<br>
<br>

## 경사 하강법 : 알고리즘

```python
"""
gradient : 미분을 계산하는 함수
init : 시작점
lr: 학습률, 미분을 통해 업데이트되는 속도를 조절한다. (중요한 변수)
eps : 알고리즘 종료조건, 컴퓨터로 미분을 계산할 때에는 정확히 0이 나오지 않기 때문에 
			종료조건 eps가 필요하다. (특정값 보다 작아지면 종료)
"""

var = init
grad = gradient(var)
while (abs(grad) > eps):
	var = var - lr * grad
	grad = gradient(var)
```

이런식으로 현재점 기준 미분값을 구하고, 미분값 크기에 비례해서 현재위치를 옮기는 방식으로 극대, 극소값을 찾을 수 있다.

2차원 공간에서의 2차함수나 3차함수를 예시로 들면 이해하기가 쉬울텐데, **만약 변수로 벡터가 들어온다면??**  

<br>

## 벡터가 입력인 다변수함수에서의 경사 하강법

이 때는 미분 대신 편미분을 사용해야 한다.

> 편미분이란 다변수 함수의 특정 변수를 제외한 나머지 변수를 상수로 간주하여 미분하는 것으로 다차원의 벡터가 변수로 들어오더라도, 각 벡터들을 편미분 하여 하나의 그레디언트 벡터로 만들 수 있다.
> 
<br>  

**편미분 예시**

<center><img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/183f49f6-a513-498d-8b03-59fc16ee06dd" width=80%></center>  
<br>

**그레디언트 벡터**

$$
\nabla{f}=(\delta_{x_1}f, \delta_{x_2}f,\delta_{x_3}f, ....\space, \delta_{x_d}f)
$$

d차원의 입력변수 벡터들을 각각 편미분한 성분벡터들의 합

**그림으로 이해**

<img src="https://github.com/hyuns66/5th_BoostCamp_AItech/assets/72616557/bb927da9-7ebc-45c3-9f1b-f0aefb30d493">

(x, y, z) 공간에서 f(x, y) 표면을 따라 $-\nabla{f}$ 벡터를 그리면 오른쪽의 그림처럼 극소값을 향하는 벡터가 그려진다.