# 모듈

- 대사의 어떤 부분 혹은 조각
- 작은 프로그램 조각
- 같은경로 내에 존재하는 .py 파일을 모듈로 사용가능
- 다른 프로그램에서 사용하기 쉽도록 각각의 프로그램을 모듈화 하는것이 중요

파이썬에서 자체적으로 Built-in-module을 지원해주기도 함

```python
# Built-in-module 예시
import random    # 난수생성 모듈
import collections    # 여러가지 유용한 자료구조 지원
import heapq    # 힙 모듈
import time    # 시간을 잴 때 사용하는 모듈
import urllib.request   # 웹
# >>> 추가정보들은 구글링
```

# 모듈의 import

모듈 파일을 만들고 import 한 뒤 (모듈이름).(함수) 로 사용한다.

```python
# module_A.py
def add(a, b):
	return a + b

def sub(a, b):
		return a - b

# main file
import module_A
module_A.add(1,2)    # 3
module_A.sub(5,4)    # 1
```

<aside>
‼️ 모듈을 바로 import 하게되면 모듈안에 있는 모든 코드가 로딩되므로 필요한 부분만 import 하거나 모듈에 전처리를 해주어야 한다.

</aside>

모듈을 아래처럼 짜고 import 하면 바로 hi가 출력되버린다.

```python
# module_A.py

def add(a, b):
	return a + b

def sub(a, b):
		return a - b

print("hi")

# main file
import module_A
# >>> hi
```

그래서 모듈안에는 함수로만 구현을 하거나, 메인함수 부분이 모듈 내에 구현이 되어있다면 메인 프로그램으로 실행될 때에만 적용되도록 if문 처리를 해주어야 한다.

```python
# module_A.py

def add(a, b):
	return a + b

def sub(a, b):
		return a - b

if __name__ == "__main__":
	print("hi")
```

혹은 원하는 함수만 따로 import 할 수도 있다.

```python
# module_A 안에 있는 add 함수만 import 하여 사용
from module_A import add

add(1, 3)   # 함수 자체를 import 했기 때문에 모듈명을 안붙여도 된다.
sub(4, 3)   # error

# 모듈 안에 있는 모든 함수를 import 
from module_A import *

add(1, 3)   # 역시 함수 자체를 import 했기 때문에 모듈명을 안붙여도 된다.
sub(4, 3)   # 1
# >>> 근데 이렇게 하면 코드 내에서 어떤 모듈 함수인지 좀 불분명해져서
# >>> 아래 Alias 설정 에서 설명하듯이 별칭을 붙여서 import 하는편이 좋음
```

# Alias 설정

모듈을 import 하면 호출할 때 마다 모듈명을 붙여야 하기 때문에 번거로움을 줄이기 위해 모듈에 별명을 붙여줄 수 있다.

```python
import module_A as a

a.add(3, 5)   # 8
a.sub(4, 3)   # 1

# 관용적으로 사용되는 별칭들 예시
import tensorflow as tf
import pandas as pd
import numpy as np
```

# 패키지

- 대형 프로젝트를 만들기 위한 코드묶음
- 모듈들의 합, 디렉토리로 연결됨
- 각 디렉토리 안에는 `__init__.py` 가 꼭 있어야 했는데 Python 3.3 부터 없어도 되는거로 바뀜

<aside>
‼️ 하나의 프로젝트 디렉토리 안에 트리구조로 디렉토리들을 생성하여 그 안에 모듈들로 채워넣어서 패키지를 구성한다.
루트 디렉토리 안에는 `__main__.py` 가 있어서 실행파일로 실행될 수 있도록 구현한 것이 프로젝트가 된다.

</aside>

다양한 프로젝트에는 다양한 모듈이 import되어 사용되는데 이 때 각 환경에 맞는 버전이 다를 수 있다.

이때 프로젝트 가상환경을 구축하여 버전관리를 할 수 있게 해주는 패키지 관리도구를 사용한다.

![image](https://user-images.githubusercontent.com/72616557/229287327-6f3b2222-84b4-4414-897c-ab70617e41b5.png)


**아나콘다**

- 머신러닝이나 데이터 분석 등에 사용하는 여러가지 패키지가 기본적으로 포함되어있는 파이썬 배포판.
- 때문에 해당 분야를 파이썬으로 접근하고자 할 때 세팅이 매우 간단해짐
- 파이썬 가상 환경을 구축하는데도 유용하게 사용가능
- 내부적으로 conda라는 환경/패키지 관리자가 존재하며 이 conda를 통해 패키지를 설치하거나 가상 환경을 관리할 수 있음.