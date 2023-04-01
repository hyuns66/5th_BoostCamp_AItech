# 행렬의 표현

$$
2x_1 + 2x_2+x_3 = 9 \\
2x_1 -x_2+2x_3=6\\
x_1 - x_2 + 2x_3=5

$$

$$
\left[
\begin{array}{cc}
   2 & 2 & 1 & 9 \\
   2 & -1 & 2 & 6 \\
   1 & -1 & 2 & 5 \\
\end{array}
\right]
$$

동치인 위 두 행렬은 아래처럼 배열로 표현 가능

```python
coefficient_matrix = [[2, 2, 1], [2, -1, 2], [1, -1, 2]]
constant_vector = [9, 6, 5]
```

그러나 파이썬 자체가 인터프리터 언어라 속도가 빠르지 않고, 배열자체가 포인터를 활용하여 메모리에 간접적으로 접근하는 방식이라 메모리효율면에서 좋지 않기 때문에

- 다양한 Matrix 계산
- 굉장히 큰 Matrix의 표현
- 처리속도의 향상

등의 이유로 적절한 패키지를 활용하는 것이 좋다.

***그것이 바로 그 유명한 Numpy(Numerical Python)***

# Numpy란?

- 파이썬의 고성능 과학계산용 패키지
- Matrix와 Vector와 같은 Array연산의 사실상 표준
- 일반 List 연산에 비해 빠르고 메모리 효율적
- 반복문 없이 데이터배열에 대한 처리 지원
- 선형대수와 관련된 다양한 기능 지원
- C, C++, 포트란 등의 언어와 통합 가능

# Numpy 모듈 예제

## ndarray

- NumPy의 N차원 배열 객체이다.
- Python의 list와 가장 큰 차이점은, 하나의 데이터 타입만 넣을 수 있다는 점이다.(Python List는 Dynamic typing을 지원하지만, ndarray는 지원하지 않는다)
- C로 프로그래밍을 한 사람이라면, 고 이해하면 가장 쉬운데, C의 Array를 사용하여 배열을 생성하기 때문

```python
import numpy as np

# 일반 파이썬 배열은 dynamic typing이 가능하다.
a = [1, 2.3, False, "아랄랄ㄹ라라", -32.11984337]

# 그러나 ndarray는 생성 시 하나의 데이터 타입만을 지정해주어야 한다.
a_ndarray = np.array([1, 2, 3, 4, 5], int)
```

- ***재밌는 사실 (파이썬에서의 리스트 메모리 구조)***
    
    ![image](https://user-images.githubusercontent.com/72616557/229288112-a55bd00a-75be-4898-8122-ed03aa223c38.png)
    
    일단 ndarray는 C언어의 array처럼 메모리에 순서대로 값들이 차곡차곡 쌓이므로 메모리 효율과 연산속도 측면에서 매우 좋은 퍼포먼스를 낼 수 있다. (C언어 배열과 완전 동일)
    
    **파이썬 List는 어떻게 생겼는데?**
    
    기본적으로 메모리 어딘가에 (알 수 없는 곳 어딘가로 초기화됨) static 한 int형 값들이 하나씩 저장되어 있다.
    
    파이썬에서 list를 선언하면 실제로는 값이 아닌 주소값이 리스트에 할당되고 각 주소들은 static한 int형 값들을 가리키게 된다.
    
    바로 코드를 보자
    
    ```python
    """ 
    일반 리스트의 경우 a[0]과 b[4]는 메모리 어딘가에 저장되어 있는
    static 한 int형 값 1 을 가리키고 있기 때문에 주소가 같아서 True가 뜬다.
    즉 a와 b는 실제로는 주소값을 담고 있는 배열로 선언되는데 파이썬 자체에서
    주소값을 한번 따고 들어가서 static한 값을 반환하도록 구현되어 있는 것.
    """
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1]
    
    a[0] is b[4]    # is 연산자는 주소가 같은지 여부를 반환하는 메서드입니다.
    # >>> True      # 분명 다른 배열인데 숫자가 같으니까 주소값이 같다고 출력됨
    
    """ 
    Numpy에서 다루는 배열 ndarray의 경우 c언어의 array로 구현되어서 
    메모리에 차례대로 값들을 때려박아버리기 때문에
     a[0]과 b[4]의 값은 같지만 실제 메모리주소는 달라서 False가 뜬다. 
    """
    a = np.array(a)
    b = np.arra(b)
    
    a[0] is b[4]    # C언어 배열에서는 이게 정상
    # >>> False
    ```
    

## array shape

| Rank | Name | Example |
| --- | --- | --- |
| 0 | scalar | 7 |
| 1 | vector | [10, 10] |
| 2 | matrix | [[10, 10], [15, 15]] |
| 3 | 3-tensor | [[[1 ,5, 9], [2, 6, 10]], [[3, 7, 11],  [4, 8, 12]]] |
| n | n-tensor | ….. |

```python
vector = [1, 2, 3, 4]
matrix = [[[1 ,5, 9], [2, 6, 10]], [[3, 7, 11],  [4, 8, 12]]]

""" shape -> ndarray의 형태를 tuple 형으로 반환 """
np.array(vector, int).shape
# >>> (4,)
np.array(matrix, int).shape
# >>> (2,2,3)

""" size -> ndarray의 총 데이터 수를 int형으로 반환 """
np.array(vector, int).size
# >>> 4
np.array(matrix, int).size
# >>> 12  (2*2*3)

""" ndim -> ndarray의 차원을 int형으로 반환 """
np.array(vector, int).ndim
# >>> 1
np.array(matrix, int).ndim
# >>> 3

""" nbytes -> ndarray가 차지하는 메모리 용량을 int형으로 반환 (byte 단위)
계산식 : <자료형 크기> * <np.array().size>  """
np.array(vector, dtype = np.int8).nbytes
# >>> 4
np.array(matrix, dtype = np.float64).nbytes
# >>> 96 (64 * 12 / 8)
```

## Handling shape

**reshape**

array의 shape을 변경함, element의 개수는 동일

```python
"""
np.array().reshape() 함수로 shape형태를 변경할 수 있으며, 인자로 -1이 들어갔을 땐
size를 기반으로 row 개수를 자동으로 선정해서 정해준다.
"""
np.array(test_matrix, int).reshape(2,4).shape
# >>> (2, 4)
np.array(test_matrix, int).reshape(-1, 2).shape
# >>> (4, 2)
```

**flatten**

reshape 함수의 기능을 하나 따온 것인데, 다차원 matrix를 1차원 배열로 변경시켜줌

```python
matrix = [[[1 ,5, 9], [2, 6, 10]], [[3, 7, 11],  [4, 8, 12]]]
np.array(matrix, int).flatten()
# >>> [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
```

## Indexing & Slicing

**Indexing**

```python
"""
ndarray 에서는 파이썬 list와는 좀 다르게 2차원 배열에서 [n, n] 형식으로 indexing 가능함
"""
matrix = [[1, 3], [4, 2], [5, 8]]
nd_matrix = np.array(matrix, int)

matrix[1][0]
# >>> 4
matrix[1, 0]
# >>> error
nd_matrix[1, 0]
# >>> 4
```

**Slicing**

위의 Indexing 기법을 바탕으로 list와 달리 행과 열 부분을 나눠서 slicing 가능

matrix의 부분집합을 추출할 때 유용하다.

```python
""" 사용법
nd_matrix[<row_space>, <column_space>] 형식으로 부분집합을 추출한다.
"""

matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
nd_matrix = np.array(matrix, int)

nd_matrix[:, 2:]    # 전체 row의 2열 이상
# >>> array([[3, 4, 5], [8, 9, 10]])
nd_matrix[1, 1:3]    # 1 row의 1열~2열
# >>> array([7, 8])
nd_matrix[1:3]    # 1row ~ 2row 전체 (쉼표가 없으면 row만 취급)
# >>> array([[1,2,3,4,5],[6,7,8,9,10]])
```

+)

slicing 할 때 [start_poing : end_point : step] 을 활용하면 아래 그림과 같이 일정한 간격을 뛰면서도 부분집합 추출이 가능하다.

![image](https://user-images.githubusercontent.com/72616557/229288155-98799ec8-daa7-4f8c-9f44-16c39d5695d7.png)

## Creation function

**arange**

파이썬의 range 함수와 비슷하게 사용해서 array 값의 범위를 지정하여 list를 생성하도록 하는 함수

```python
np.arange(5)
# >>> array([0, 1, 2, 3 ,4])

""" (start_poing, end_point, step) 의 형태로도 가능하다"""
np.arange(3, 5, 0.5)
# >>> array([3., 3.5, 4., 4.5])

""" reshape 응용하면 matrix 한번에 생성 가능"""
np.arange(10).reshape(2, 5)
# >>> array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

"""
zeros : 0으로 초기화
ones : 1로 초기화
empty : 빈공간으로 shape만 잡아줌. 즉 이전에 썼던 쓰레기값이 들어있음
"""
np.zeros(shape=(10,), dtype=np.int8)
# >>> array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

np.ones((2, 5))
# >>> array([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]])
```

**identity**

단위행렬을 생성함

```python
np.identity(n = 3, dtype=np.int8)    # n : number of rows
# >>> array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

np.identity(5)
# >>> array([[1., 0., 0., 0., 0.],
#						 [0., 1., 0., 0., 0.],
#						 [0., 0., 1., 0., 0.],
#						 [0., 0., 0., 1., 0.],
#						 [0., 0., 0., 0., 1.])
```

그니까 아래쪽에 저거 행렬 만들어주는 함수다.

![image](https://user-images.githubusercontent.com/72616557/229288225-bebcb71f-8920-406c-bdc6-be49b06cfed0.png)


**eye**

좀 더 자유로운 형태의 단위행렬 비슷한 행렬? 을 생성해주는 함수

그니까 행렬 범위 자유롭게 지정해서 원하는 위치에 대각선으로 1이 들어가는 행렬

```python
import numpy as np

""" 사용법
N * M matrix를 생성하기 위해서 함수에 인자로 지정해줄 수 있다.
k값을 지정해주면 대각 1 데이터의 시작위치를 지정해줄 수 있다.
만약 k값이 음수라면 -k 번째 row에서 시작한다.
"""

# (example) 2 by 2 행렬, 시작위치 0
>>> np.eye(N=2) 
array([[1., 0.],
       [0., 1.]])

# (example) 2 by 3 행렬, 시작위치 0
>>> np.eye(N=2,M=3) 
array([[1., 0., 0.],
       [0., 1., 0.]])
     
# (example) 7 by 7 행렬, 시작위치 +2
>>> np.eye(7,k=2) 
array([[0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.]])

 # (example) 7 by 7 행렬, 시작위치 -2
>>> np.eye(7,k=-2)
array([[0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.]])
```

- **단위에 대해**
    
    [NumPy의 데이터 타입(자료형), 관련된 함수 - NumPy(2)](https://kongdols-room.tistory.com/53)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6180257d-f08a-4052-81db-f41d70c833a8/Untitled.png)
    
    단위행렬에 int형 말고 다양한 단위로 변경 가능하다. 
    보통 1과 0으로 이루어지는게 단위행렬이니까 무조건 int 형이라고 생각할 수 있는데 boolean 값을 넣으면 True로 셋팅이 된다
    
    그리고 int보다 np.int8이 메모리를 훨씬 적게 잡아먹으므로 적절한 사용처에 따라 메모리를 고려하여 단위를 설정해주면 좋을것 같다.
    
    **단위행렬을 가공하지 않고 정말 단위행렬의 기능으로써만 쓰고자 한다면 boolean으로 셋팅해주는게 메모리를 가장 적게 잡아먹지 않을까?**
    
    ![image](https://user-images.githubusercontent.com/72616557/229288328-cbba7d4b-85d8-420f-a715-fb0afb87ed85.png)
    
    ![image](https://user-images.githubusercontent.com/72616557/229288337-da4d18c2-5f4c-459a-9a84-24aca147f73a.png)

    ![image](https://user-images.githubusercontent.com/72616557/229288342-e017e7e4-f005-4bfa-8879-2f9be81208ae.png)

    boolean은 딱 참거짓만 나타내니까 1비트로 충분하지 않을까 하고 예상했는데
    
    > bool형은 첫 글자가 대문자인 Python 예약어로써 **1byte**
    의 크기를 가지고 있다.
    > 
    
    라네요…. 
    

**diag**

대각행렬의 값을 추출함

![image](https://user-images.githubusercontent.com/72616557/229288643-3454f74d-5f81-4ae4-a55a-3aade5922df0.png)

이런거에요~~ 위에서 기술한 eye 행렬 가지고 마스킹해서 값 추출해서 반환하는 함수

**concatenate**

![image](https://user-images.githubusercontent.com/72616557/229288674-6c0e5578-08c5-4619-a3fa-71c8eb9d0210.png)

## array operation

**element-wise operation**

행렬간에 사칙연산이 지원된다.

+, -, *, / 연산자를 사용하면 shape가 같은 행렬들에 한 해서 같은 위치에 있는 값들끼리 사칙연산을 적용하여 같은 shape의 행렬을 반환한다.

*(곱하기) 연산의 경우 행렬곱(내적)이 아닌 element-wise operation으로 적용된다.

![image](https://user-images.githubusercontent.com/72616557/229288886-32414d76-7f49-4351-8398-f6675adcba3d.png)

이런식으로 덧셈 뺄셈 곱셈 나눗셈 다 가능해요~

**dot product**

행렬 두개 넣어서 행렬곱 연산 가능

```python
a = np.arange(1, 7).reshape(2, 3)
b = np.arange(7, 13).reshape(3, 2)

a.dot(b)
```

**transpose**

row와 collumn을 뒤집는 전치행렬 생성

```python
a = np.arange(1, 7).reshape(2, 3)
# >>> array([1, 2, 3],
#           [4, 5, 6])

# 아래 두 표현 모두 동치이다.
a.transpose()
a.T
# >>> array([1, 4],
#           [2, 5],
#           [3, 6])
```

**broadcasting**

shape이 같은 행렬끼리의 사칙연산은 위에서 기술했던 element-wise operation에 의해 사칙연산이 수행되지만

 shape이 다른 행렬끼리 연산을 하게되면 빈 공간에 복사해서 채워넣은 다음 shape 모양을 맞춰서 사칙연산이 수행된다. 

이를 broad casting 연산이라 부름

![image](https://user-images.githubusercontent.com/72616557/229288917-5a3d7917-f282-4af2-8f5b-63d378a72fa6.png)

## Comparisons

**All & Any**

```python
"""
Array의 데이터 전부 또는 일부가 조건에 만족하는지 여부를 Boolean형으로 반환
이 연산에도 broadcasting 이나 element-wise 연산이 적용된다.
"""

a = np.arange(10)
# >>> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a < 5
# >>> array([True, True, True, True, True, False, False, Flase, False, False])

np.any(a < 5), np.any(a < 0)    # 하나라도 조건에 만족한다면 True
# >>> (True, False)

np.all(a > 5), np.all(a < 10)    # 모두가 조건에 만족한다면 True
# >>> (False, True)

""" 
a < 5 와 같은 연산으로 나온 Boolean 배열 [True, False, False, False] 같은 배열들을
logical_and, logical_or 같은 연산을 통해서 묶어줄 수 있다.
"""
# 예시
a = np.array([1, 3, 0], float)

np.logical_and(a > 0, a < 3)
# >>> array([True, False, False])
# [True, True, False]와 [True, False, True]의 logical_and 연산결과
# 유사하게 logical_or, logical_not 연산들도 있음
```