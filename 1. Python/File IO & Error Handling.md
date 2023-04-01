# Exception Handling

```python
try:
	# 예외 발생 우려되는 코드
except <Exception Type>:
	# 해당 예외 발생 시 대응코드
else:
	# 예외 발생하지 않았을 때 작동할 코드
finally:
	# 예외가 발생하던 발생하지 않던 무조건 작동할 코드

# 예시
a = [1, 2, 3, 4, 5]
for i in range(N):
	try:
		print(i, 10//i)
		print(a[i])
	except ZeroDivisionException:
		print("Can't divided by 0")
	except IndexError as e:    #IndexError 에 e라는 별명 붙임
		print(e)
	except Exception as e:    # 무슨 예외가 나올지 모르는 경우 Exception으로 퉁침
		print(e)                # 무슨 Exception인지 알아서 출력됨
```

# Built-in-Exception

파이썬에서 기본적으로 제공하는 예외들

- IndexError : List의 Index범위를 넘어갈 때
- NameError : 존재하지 않는 변수를 호출하 때
- ZeroDivisionError : 0으로 숫자를 나눌 때
- ValueError : 변환할 수 없는문자/숫자를 변환하려 할 때
- FileNotFoundError : 존재하지 않는 파일을 호출할 때
- ETC……

# 강제로 Exception 유도하기

raise 또는 assert 구문을 사용하여 Exception을 일으킬 수 있다.

```python
raise <Exception Type>(예외정보)

# 예제
# 정수값을 입력하지 않을 경우 ValueError 가 일어나서 프로그램이 종료되도록 작성
while True:
	value = input("변환할 정수값 입력")
	for digit in value:
		if digit not in "0123456789":
			raise ValueError("숫자값을 입력하지 않았습니다")
	print("정수값으로 변환된 숫자 -", int(value))

assert 예외조건

# 예제
# assert 뒤에 붙은 예외조건에 의해 True, False가 나오는데, False인 경우
# AssertionError 가 뜨면서 프로그램이 종료된다.
def get_binary_number(decimal_number : int):
	assert isinstance(decimal_number, int)
	return bin(decimal_number)

print(get_binary_number(10.0))
# >>> int가 아닌 float형을 넣었으므로 assert 구문에서 False로 걸려 에러가 난다.
# AssertionError Traceback ~~~~ 블라블라 하고 프로그램 종료
```

# File I/O

**파일 open/close**

```python
f = open("<파일 이름>", "접근 모드", encoding = "utf8")
f.close()
# >>> encoding은 지정해주지 않아도 됨 (선택사항)
```

| 접근 모드 | 설명 |
| --- | --- |
| r | 읽기모드 - 파일을 읽기만 할 때 사용 |
| w | 쓰기모드 - 파일에 내용을 쓸 때 사용 (기존파일을 오픈해서 사용할 시 덮어쓰기됨) |
| a | 추가모드 - 파일의 마지막에 새로운 내용을 이어서 쓸 때 사용 |

**with 구문 사용하기**

with문의 동작과정

- 진입할 때 객체의 `__enter__`함수 호출
- 종료될 때 객체의 `__exit__`함수 호출

따라서 with문에서 빠져나갈 때 자체적으로 file.close() 함수를 실행해주므로 with문을 사용하면 매번 파일을 close 시켜주지 않아도 된다.

```python
with open('test.txt', 'r') as f:
	f.read()

# >>> f.close()를 호출해주지 않아도 with문을 빠져나가면서 자동으로 호출해줌.
# 즉 아래코드와 동치이다.

f = open('test.txt', 'r')
f.read()
f.close()
```

> 굳이 with문을 사용하는 이유는?
→ Exception과 같은 알 수 없는 이슈로 인해 프로그램이 강제종료되거나 해서 정상적으로 코드가 동작하지 않아 file.close()되지 않고 열려있는 상태에서 프로그램이 종료되는것을 막을 수 있다.
> 

# OS 모듈

os 모듈을 사용하여 디렉토리 구조를 핸들링 할 수 있다.

```python
import os
os.mkdir("test")
os.mkdir("./test/log")
# >>> 현재 경로 기준으로 test 폴더 만들고 안에 log 폴더 만들기

# 기존에 존재하는 폴더를 또 만드려고 시도하면 Exception 뜨면서 종료됨
# Exception Handling 응용
try:
	os.mkdir("test")
except FileExistsError as e:
	print("Already Created")

# 또는 조건문 활용
if not os.path.exists("test"):    # True/False 반환
	os.mkdir("test")
```

shutil 모듈을 사용하면 파일을 직접 핸들링 할 수 있다.

pathlib 모듈을 사용하면 파일경로를 String이 아닌 객체로 다루기 때문에 각 운영체제 환경에 대응하기가 수월해진다. 

- Mac OS 와 Window OS는 파일경로를 String으로 표현하는 방식이 다름.
- 즉 shutil 모듈을 사용하면 두 운영체제간 경로명 충돌이 나서 제대로 동작하지 않을 확률이 있지만 pathlib 모듈을 사용하면 String 경로가 아닌 객체로 표현하기 때문에 충돌이 나지 않는다.

```python
import shutil
# 파일을 다른 경로로 복사하는 예제
from_file_path = '../test_folder/folder1/sample_01.txt' # 복사할 파일
to_file_path = '../test_folder/folder2/sample_02.txt' # 복사 위치 및 파일 이름 지정
shutil.copyfile(from_file_path, to_file_path)

import pathlib
# 파일을 다른경로로 옮기는 예제
p_sub_dir_file = p_dir.joinpath('sub_dir', 'file2.txt')

print(p_sub_dir_file)
# temp/dir/sub_dir/file2.txt

print(p_sub_dir_file.is_file())
# True
```

여러가지 기능이 많은데 너무 많기 때문에 필요할 때 따라서 구글링해서 사용하는게 좋을듯

[[python] OS 모듈 (+ shutil 모듈)](https://colinch4.github.io/2020-12-04/OS_module/)

[[python] pathlib 사용법 (패스(경로)를 객체로써 조작, 처리)](https://engineer-mole.tistory.com/191)

# 데이터 영속화(Pickle)

text 형식으로 파일에 작성하는 것 외에도 객체정보 자체를 binary 파일 형식으로 저장할 수 있다.

- 일반적으로 프로그램이 종료되면 그 안에 있던 객체들도 삭제되어 더이상 사용불가능하다.
- 객체들을 파이썬 전용 binary 파일 형태로 저장해줄 수 있는 라이브러리가 pickle이다.
- 객체정보를 binary 파일 형태로 저장 후 다른 프로그램에서 다시 불러올 수 있다.
- 원래는 프로그램과 함께 사라져야 하는 데이터 객체를 저장하기에 데이터 영속화라 부른다.

```python
# --------- program_A.py ---------
import pickle

class Multiply(object):
	def __init__(self, multiplier):
		self.multiplier = multiplier

	def multiply(self, number):
		return self.multiplier*number

mult = Multiply(5)
mult.multiply(10)
# >>> 50

# .pickle 파일에 Multiply 객체정보를 바이너리 형식으로 저장
# pickle 파일형식은 바이너리 파일이므로 wb, rb 형식으로 오픈해주어야 함
f = open("multiply_opject.pickle", "wb")
pickle.dump(mult, f)    # mult 객체를 f 파일에 옮겨적음
f.close()

# --------- program_B.py ---------
import pickle

# multiply_object.pickle파일에 작성되어 있는 객체정보를 불러와서 사용
f = open("multiply_opject.pickle", "rb")
multiply_pickle = pickle.load(f)
multiply_pickle.multiply(5)
# >>> 25
```

# Logging

| Level | 개요 |
| --- | --- |
| debug | 개발 시 처리기록을 남겨야 하는 로그 정보 |
| info | 처리가 진행되는 동안의 정보 |
| warning | 사용자가 잘못 입력한 정보나 처리는 가능하나 원래 개발 시 의도치 않은 정보가 들어옴 |
| error | 잘못된 처리로 인해 에러가 났으나, 프로그램은 동작할 수 있음 |
| critical | 잘못된 처리로 데이터 손실이나 더이상 프로그램이 동작할 수 없음 |

```python
import logging

# 로그 생성
# 파라미터에 이름을 넣어주거나, 넣어주지 않으면 "root" 이름으로 생성됨
logger = logging.getLogger("<logger 이름>")

# 로그의 출력 기준 설정
# 기본적으로 로그는 사용자 레벨의 WARNING부터 찍히게 되어있으므로 
# 그냥 사용하게 되면 DEBUG와 INFO 로그가 찍히지 않는다.
# basicConfig 함수를 통해 기본 출력레벨을 DEBUG로 조정하여 모든 로그가 찍힐 수 있게 함
logging.basicConfig(level = logging.DEBUG)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# asctime : 시간
# name : logger 이름
# levelname : Logging_Level (debug, info, warning, error, critical)
# message : 로그에 입력할 메시지

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('my.log', mode = "a", encoding = "utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

for i in range(10):
	logger.info(f'{i}번째 방문입니다.')

# >>> console 창에 로그가 출력되며 my.log 파일에도 계속해서 로그 기록이 저장된다.
```
<img src="https://user-images.githubusercontent.com/72616557/229287502-8fd65ea8-3d9e-4162-afae-0fe382c7672e.png" width="70%" height="70%"/>

**BUT)**

실제 동작하는 프로그램에서 log를 남기기 위해서는 데이터 파일 위치, 파일저장소, Operation Type 등등의 설정이 필요함. 

**ConfigParser (파일에 설정내용 작성)**

- 프로그램의 실행설정을 .conf파일에 저장
- Section, Key, Value 값의 형태로 설정된 설정파일을 사용
- 설정파일을 Dict Type 으로 호출 후 사용

```python
# ------- logging.conf -------
[loggers]
keys = root

[handlers]
keys = consoleHandler

[formatter]
keys = simpleFormatter

[logger_root]
level = DEBUG
handlers = consoleHandler

[handler_consoleHandler]
class = StreamHandler
level = DEBUG
formatter = simpleFOrmatter
args = (sys.stdout)

[formatter_simpleFormatter]
format = %(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt = %n/%d/%Y %I:%M:%S %p

# ------- program.py -------
import configparser
config = configparser.ConfigParser()

config.read('logging.conf')
# 아래에서 설정파일 내용가지고 key값들로 조회해서 기초설정 진행 

```

**ArgParser (실행시점에서 직접 설정)** 

- console 창에서 프로그램 실행 시 Setting 정보 저장
- 거의 모든 console 기반 Python 프로그램 기본으로 제공
- 프로그램을 실행할 때 console 창에 옵션들을 직접 입력해서 실행하는 방법

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--decimal", dest="decimal", action="store")
parser.add_argument("-f", "--fast", dest="fast", action="store_true")
# action = "store" : -d 옵션으로 값을 입력해서 decimal 변수에 저장
# action = "store_true" : -f 옵션이 입력되었으면 fast 변수에 True 저장, 입력 안하면 False
# action = "store_false" : store_true랑 반대로 옵션 입력되었으면 False, 입력 안하면 True
# 이외에도 다른 action들 많은데 너무 많아서 여기다 다 안쓰고 아래 링크타서 확인 하면 된다.
args = parser.parse_args()

print(args.decimal)
print(args.fast)

# ------ console창 실행 --------
$ ./run.py -d 1 -f
1             # args.decimal
True       # args.fast
```

더 자세한 내용은 아래 공식문서를 보면 알 수 있다.

[argparse - Parser for command-line options, arguments and sub-commands](https://docs.python.org/ko/3/library/argparse.html#action)