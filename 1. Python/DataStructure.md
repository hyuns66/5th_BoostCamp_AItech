# 튜플

- 소괄호 ‘( )’ 를 사용하여 표현
- 리스트와 거의 동일하게 동작하지만 **값의 변경이 불가능한 리스트**
- 프로그램이 작동하는 동안 변경되지 않아야 하는 값들에 사용
- 원본 데이터set같이 사용자의 실수에 의해 변경되면 안되는 값에 사용
- 값이 하나밖에 없으면 마지막에 , 를 붙여주어야 함

```python
a = [1, 2, 3]   # list
a = (1, 2, 3)   # tuple

# 하나짜리 튜플
a = (1)    # 정수 1로 인식됨. 틀린표현
a = (1,)   # 튜플 ( 1 )로 인식됨. 맞는표현

b = a[1]   # 리스트처럼 인덱스로 접근 가능
# a[1] = 3   # 값 변경 불가능 (에러)
```

# 집합

- 값의 중복이 없는 리스트
- set함수 안에 리스트를 넣으면 집합으로 묶어줌
- 중괄호 ‘{ }’ 로 선언가능
- 순서가 없음 → 리스트처럼 인덱스로 접근 불가능
    - for문으로 set 안의 원소들 접근가능

```python
# 초기화
s = set([1,1,2,3,3,3])    # s = {1,2,3}
또는
s = {1,2,3} 

# 값 하나 추가
s.add(4)    # {1,2,3,4}

# 값 여러개 추가
s.update([5,6,7])   # {1,2,3,5,6,7}

# 값 하나 제거
s.remove(3)   # {1,2} 
s.discard(3)   # 똑같이 동작하지만 존재하지 않는 아이템 삭제하려 할 시 에러 발생

# 접근
for num in s:
	print(num)
# >>> 1 2 3

# 집합 객체 제거
del s

# 집합 연산
s1 = {1,2,3}
s2 = {3,4,5}
# 합집합
s1.union(s2)    # {1,2,3,4,5}
# 교집합
s1 & s2         # {3}

```

# 스택

대괄호 ‘[ ]’를 사용하여 표현 (리스트)

```python
stack = list()
또는
stack = [1,2,3,4,5]

stack.append(1)    # push
stack.pop()        # pop

stack[-1]          # pop처럼 삭제하지 않고 마지막 요소 조회
```

# 큐

- 사실상 덱(deque)으로 모든 자료구조 구현 가능
- 동작속도는 왠만하면 deque가 list보다 빠르기 때문에 단순 스택이나 리스트가 아닌 이상 deque를 사용하는편이 좋다.

```python
# 꼭 collections 라이브러리를 통해 import해주어야 함
from collections import deque

deq = deque()      # [1,2,3,4,5] 를 기준으로 각 함수 적용

# deq[0]으로 put
deq.appendleft(10)     # [10,1,2,3,4,5]

# deq[-1]으로 push
deq.append(0)          # [1,2,3,4,5,0]

# deq[0]을 get
deq.popleft()          # [2,3,4,5]

# deq[-1]을 pop
deq.pop()              # [1,2,3,4]

# item을 찾아서 deq에서 삭제
deq.remove(3)          # [1,2,4,5]

# deq 회전
deq.rotate(1)          # [5,1,2,3,4]
deq.rotate(-1)         # [2,3,4,5,1]
deq.rotate(3)          # [3,4,5,1,2]
```

# 힙 (우선순위 큐)

- **우선순위 큐** : 들어오는 순서 관계없이, 우선순위가 높은 데이터가 먼저 나가는 자료구조
- **힙** : 우선순위 큐를 구현하기 위해 만들어진 자료구조
- 여러개의 값을 받아놓고, 최대값, 최소값을 찾는 연산이 빠르다.

**힙의 특징**

- **완전이진트리** 형태로 이루어져 있다.
- 부모노드와 서브트리간 대소 관계가 성립된다. (반정렬 상태)
- 이진탐색트리(BST)와 달리 중복된 값이 허용된다.
- 힙의 종류
    - **최대 힙 (Max Heap)**
        
        부모 노드의 키 값이 자식 노드보다 크거나 같은 완전이진트리이다.
        
        *❝ key(부모노드) ≥ key(자식노드) ❞*
        
    
    ![https://blog.kakaocdn.net/dn/cT2Dxb/btqSATggBLA/CIBeKSLq0s6MDTNVM345Jk/img.png](https://blog.kakaocdn.net/dn/cT2Dxb/btqSATggBLA/CIBeKSLq0s6MDTNVM345Jk/img.png)
    
    - **최소 힙 (Min Heap)**
        
        부모 노드의 키 값이 자식 노드보다 작거나 같은 완전이진트리이다.
        
        *❝ key(부모노드) ≥ key(자식노드) ❞*
        
    
    ![https://blog.kakaocdn.net/dn/bwtTZl/btqSASIpEE1/zJxtetzfI1OGHucT99Mcuk/img.png](https://blog.kakaocdn.net/dn/bwtTZl/btqSASIpEE1/zJxtetzfI1OGHucT99Mcuk/img.png)
    

**힙 연산**

> upheap (enque)
> 
- 힙에 데이터를 삽입할때 사용되는 연산.
- 힙의 가장 마지막 자리에 데이터 삽입
- 부모노드와 비교하면서 우선순위에 따라 교환 여부 체크
- 교환이 불필요할 때 까지 반복
- enque 연산에서 아래와 같이 적용된다
    
    ![image](https://user-images.githubusercontent.com/72616557/228262338-e3519ae3-b759-4882-af69-d877aed748a7.png)

    

> heapify(downheap) (deque)
> 
- 힙에서 최대(최소) 값을 pop 한 뒤, 재정렬 할 때 사용
- root 노드의 값을 pop
- 가장 마지막 leaf 노드를 root 노드로 올림
- 좌/우 자식노드 중 우선순위가 높은 자식노드와 비교하여 교환여부 체크
- 더이상 교환이 불필요할 때 까지 반복
- downheap
    
   ![image](https://user-images.githubusercontent.com/72616557/228312269-f6e96957-6d71-427b-aad0-a604ec7d2d53.png)

    
- deque
    
    ![image](https://user-images.githubusercontent.com/72616557/228312415-53012ec9-f590-4191-9be7-0f473139f407.png)
    

사실상 C언어에서 linked_list 나 배열 사용해서 구현가능한데 파이썬에서는 heapq 모듈을 지원한다.

heap 자료구조 자체를 지원하는게 아니라 함수를 지원하기 때문에 리스트를 통해 함수를 호출해야 한다.

기본적으로 heapq는 최소힙으로 구현되어있음

```python
import heapq

heap = list()

# --- enque ---
heapq.heappush(heap, 4)
heapq.heappush(heap, 1)
heapq.heappush(heap, 7)
heapq.heappush(heap, 3)
# >>> heap = [1, 3, 7, 4]

# --- deque ---
heapq.heappop(heap)     # pop heap[0] (== 1)
# >>> heap = [3, 7, 4]
# root 노드는 heap[0]으로 조회가능

# --- 리스트를 힙으로 변환 ---
list = [4, 1, 7, 3]
print(list)    # >>> [4, 1, 7, 3]
heapq.heapify(list)
print(list)    # >>> [1, 3, 7, 4]

# --- 최대힙 구현 ---
# 기본적으로 최소힙이기 때문에 heappush를 할 때 값을 음수로 뒤집어서
# 역정렬이 되게 한 다음 꺼낼 때 다시 양수로 바꿔주는 방법 사용
max_heap = list()
nums = [4, 1, 7, 3, 8, 5]

for num in nums:
  heappush(max_heap, (-num, num))  # (우선 순위, 값)

while heap:
  print(heappop(max_heap )[1])  # index 1

# --- heap sort ---
# 리스트를 받아서 힙에 넣어주고 하나씩 꺼내기만 하면 정렬이 되는 알고리즘
def heap_sort(nums):
  heap = []
  for num in nums:
    heappush(heap, num)

  sorted_nums = []
  while heap:
    sorted_nums.append(heappop(heap))
  return sorted_nums
```