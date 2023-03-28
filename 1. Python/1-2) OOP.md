**관용적인 네이밍 규칙**

snake_case : 함수 및 변수명에 사용  
CamelCase : Class명에 사용



# 클래스의 형태

```python
Class SoccerPlayer(object):
	def __init__(self, name, position, back_number):    # 변수 초기화
		self.name = name
		self.position = position
		self.back_number = back_number
# >>> 이 SoccerPlayer 클래스는 name, position, back_number 변수를 가진다.

	def kick(self)
		print(self.name + "이 공을 찹니다!!")
# >>> 클래스 내 메서드는 self를 꼭 인자로 넣어야 클래스 매서드로 인정됨

```

## 맹글링

<aside>
💡 mangle : 짓이기다
파이썬이 변수/함수 이름을 짓이겨서 다른이름으로 바꿔버리는 것을 의미
언더바( _ )를 변수 앞뒤로 붙여서 `__변수__` 와 같이 표현

</aside>

아래와 같은 케이스에 사용한다.

- (앞 언더바 1개)모듈내에서만 변수/함수를 사용할때
- (뒤 언더바 1개)파이썬 변수/함수명 충돌을 피하기 위해
- (앞 언더바 2개)네임 맹글링
- (앞뒤 언더바 2개씩)매직 메소드

자세한 내용은 아래에 정리 잘돼있어서 이거 참조 

[[python] 파이썬 언더바, 언더스코어, 밑줄, 맹글링, 매직메소드](https://losskatsu.github.io/programming/py-underscore/#72-%EC%98%A4%EB%B2%84%EB%9D%BC%EC%9D%B4%EB%94%A9-%EB%B0%A9%EC%A7%80%EC%9A%A9)

이중에 **네임 맹글링**만 다뤄보면, 객체 내 변수를 private화 시키는 것이라 볼 수 있다. 

```python
# 변수 앞에 __ (언더바 두 개)를 붙이면 private화 되어서 외부에서 접근이 불가능하다.
# 즉 man.__hobby = "축구" 와 같이 임의대로 변경 불가능해진다.
class TestClass:
    def __init__(self):
        self.name = "왕춘삼"
        self.age = 47
        self.__hobby = "인형놀이"

man = TestClass()
print(man.name, man.age, man.__hobby)
# AttributeError: 'TestClass' object has no attribute '__hobby'
```

# 매직메서드

**스페셜 메소드(Special method) 또는** **던더 메소드(Double UNDERscore method)** 라고 부르기도 함

- 이미 파이썬 내에 정의되어 있고, 클래스 내부에서 매직 메소드들을 오버라이딩 하여 사용
- 또한 직접 호출해서 사용하지 않고, 정해진 규칙에 따라 알아서 호출된다는 특징이 있다.

💡 `__init__` `__add__` `__doc__` `__str__` 처럼 파이썬에 내장되어있는것을 오버라이딩하여 구현해서 사용함  


아래 보이는 것처럼 파이썬에 존재하는 모든 자료형은 사실 클래스로 구현되어 있기 때문에 우리가 구현한 클래스들도 여타 자료형처럼 사칙연산 및 출력 등이 가능하다.

이를 구현해주는 함수를 매직메서드라고 부른다.

```python
print(int)     <class 'int'>
print(float)   <class 'float'>
print(str)     <class 'str'>
print(list)    <class 'list'>
print(tuple)   <class 'tuple'>
print(dict)    <class 'dict'>
```

즉 1+1 연산을 하게되면 int 형 클래스 내부에 구현되어있는 `__add__` 매직 메서드가 호출되어 미리 정의된 기능을 수행할 뿐이다.

이를 이용하면 아래처럼 클래스 간 사칙연산을 수행하게 만들 수 있다.

```python
class Fruit(object):
    def __init__(self, name, price):
        self._name = name
        self._price = price

    def __add__(self, target):
        return self._price + target._price

    def __sub__(self, target):
        return self._price - target._price

    def __mul__(self, target):
        return self._price * target._price

    def __truediv__(self, target):
        return self._price / target._price

apple = Fruit("사과", 100000)
durian = Fruit("두리안", 50000)

print(apple + durian) # 150000
print(apple - durian) # 50000
print(apple * durian) # 5000000000
print(apple / durian) # 2.0
print(f"{apple}와 {durian}") # 사과와 두리안
```

매직메서드의 종류는 너무 다양하기 때문에 여기서 모든것을 다 기술할 수는 없고 대표적인 예시만 적어놓도록 하겠다. (나머지는 필요에 따라 키워드로 추가 구글링)

- **__add__, __sub__, __mul__, __truediv__**

각각 +, -, *, / 기호에 매핑되어 해당 연산을 할 때 호출됩니다.

- **__len__**

객체의 길이를 반환할 때 사용합니다. len()함수가 내부적으로 객체의 이 메소드를 호출합니다.

- **__bool__**

객체의 boolean 표현을 나타낼 때 사용합니다.

- **__new__**

객체를 생성할 때 가장 먼저 실행되는 메소드입니다. __init__보다 먼저 실행되는게 특징이고 새로 생성된 객체를 반환합니다.  특수한 상황이 아니면 잘 사용하지 않는 메소드 입니다.

**첫 번째 인자로 클래스 자신이 넘어옵니다.**

- **__init__**

우리가 보통 생성자라고 부르는 메소드입니다. __new__ 메소드로 인해 객체가 생성되고 나면 호출됩니다. **데이터를 초기화 하는등의 목적**으로 사용합니다.

- **__del__**

객체가 소멸될 때 호출됩니다.

- **__str__, __repr__**

객체의 문자열 표현을 위해 사용됩니다.

# 상속

부모클래스로부터 속성과 메서드를 물려받은 자식클래스 생성

```python
# 모든클래스는 Object클래스 상속이 기본이기에 Country(Object) 로 선언하는 것이 원칙이지만
# Python3 부터는 Object는 자동으로 상속시켜주므로 작성하지 않아도 무관
# Python3 이전 버전을 사용할 때에는 클래스 선언 시 Country(Object)로 작성해주세요
class Country:
    """Super Class"""

    name = '국가명'
    population = '인구'
    capital = '수도'

    def show(self):
        print('국가 클래스의 메소드입니다.')

class Korea(Country):    # Country 클래스를 상속받은 자식클래스
    """Sub Class"""

    def __init__(self, name):
        self.name = name

    def show_name(self):
        print('국가 이름은 : ', self.name)

# >>> from inheritance import *
# >>> a = Korea('대한민국')
# >>> a.show()
# 국가 클래스의 메소드입니다.
# >>> a.show_name()
# 국가 이름은 :  대한민국
# >>> a.capital
# '수도'
# >>> a.name
# '대한민국'
```

이렇게 하면 자식클래스에서 부모클래스의 함수까지 모두 사용가능하고 부모클래스의 함수를 자식클래스에서 재정의하여 메소드 오버라이딩도 가능하다. 굳이 예제 안씀 알잖아 우리 컴공이면 >.<

```python
class 자식클래스(부모클래스1, 부모클래스2):
        ...내용...

# 이런식으로 다중상속도 가능하다.
```

# Polymorphism (다형성)

- 같은 이름 메소드의 내부로직을 다르게 작성
- Dynamic typing 특성으로 인해 파이썬에서는 같은 부모클래스의 상속에서 주로 발생함
- 중요한 OOP의 개념이지만 일단 간단하게 짚고 넘어가기

뭐 여러가지로 구현될 수 있는데 예시 몇개 들어보자면

**메서드 오버라이딩**

```python
# 같은 Person을 오버라이딩하여 구현된 자식클래스이지만 다르게 동작하는 work 함수
class Person:
    def __init__(self, name):
        self.name = name

    def work(self):
        print (self.name + " works hard")        

class Student(Person):
    def work(self):
        print (self.name + " studies hard")

class Engineer(Person):
    def work(self):
        print (self.name + " develops something")
```

**객체에 따라 다르게 동작하는 같은함수**

```python
# 같은 attack 메서드지만 각자 다르게 동작하는 함수 구현
class Elf:
    def __init__(self, name):
        self.name = name

    def attack(self):
        print ("마법으로 공격합니다.")

class Fighter:
    def __init__(self, name):
        self.name = name

    def attack(self):
        print ("주먹으로 공격합니다.")

elf1 = Elf('Dave')
fighter1 = Fighter('Anthony')
ourteam = [elf1, fighter1]
for attacker in ourteam:
    attacker.attack()

# >>> 마법으로 공격합니다.
# >>> 주먹으로 공격합니다.
```

솔직히 간단하게만 보면 뭐 별거없다. 그냥 같은함수 다르게 동작하게 구현하면 끝
