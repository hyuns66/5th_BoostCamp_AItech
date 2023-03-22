# Dataset과 DataLoader

> **데이터 샘플을 처리하는 코드는 지저분(messy)하고 유지보수가 어려울 수 있습니다; 더 나은 가독성(readability)과 모듈성(modularity)을 위해 데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적입니다.**
> 

특정 도메인에 대한 데이터를 자주 다뤄야 하는 경우 해당 데이터셋에 대한 표준화된 처리방법의 제공이 필요합니다. 이를 Pytorch 라이브러리의 `Dataset` 객체를 이용하여 `CustomDataset`으로 만들 수 있습니다.

`DataLoader` **는** `Dataset` **을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.**

## Dataset은 어떻게 만드는데요?

데이터 형태에 따라 함수를 다르게 정의할 수 있지만 최근에는 [HuggingFace](https://huggingface.co/)와 같은 표준화된 라이브러리를 사용합니다. 

**(중요)** 모든것을 데이터 생성 시점(`__init__()`)에 처리할 필요는 없습니다. 

- CPU에서 데이터 전환을 하고
- GPU에서 학습이 일어납니다.

따라서 병렬적으로 처리가 가능하기에 `getItem()`  메서드에서 변환된 Tensor를 반환해주는 것이 아니라 학습에 필요한 시점에 CPU에서 Tensor로 변환하여 GPU 로 건네주는 방식으로 진행합니다.

**Dataset의 기본 구성요소**

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
		"""
		해당 메서드는 Dataset의 최대 요소 수를 반환하는데 사용됩니다. 
		해당 메서드를 통해서 현재 불러오는 데이터의 인덱스가 적절한 범위 안에 있는지 확인할 수 있습니다.
		"""
    def __init__(self,):
        pass
		"""
		해당 메서드는 Dataset의 최대 요소 수를 반환하는데 사용됩니다. 
		해당 메서드를 통해서 현재 불러오는 데이터의 인덱스가 적절한 범위 안에 있는지 확인할 수 있습니다.
		"""
    def __len__(self):
        pass
		"""
		해당 메서드는 데이터셋의 idx번째 데이터를 반환하는데 사용됩니다. 
		일반적으로 원본 데이터를 가져와서 전처리하고 데이터 증강하는 부분이 모두 여기에서 진행될 겁니다. 
		"""
    def __getitem__(self, idx):
        pass
```

## DataLoader 클래스

Dataset은 하나의 데이터를 어떻게 (Tensor) 변환할 것인가에 대한 내용이라면 DataLoader는 여러개의 Dataset들을 어떻게 묶어서 학습 모델에 전달해 줄 것인가에 대한 내용입니다.

```python
class CustomDataset(Dataset):
    def __init__(self, text, labels):
            self.labels = labels
            self.data = text

    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.data[idx]
            sample = {"Text": text, "Class": label}
            return sample

text = ['Happy', 'Amazing', 'Sad', 'Unhapy', 'Glum']
labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']
MyDataset = CustomDataset(text, labels)
```

```python
MyDataLoader = DataLoader(MyDataset, batch_size=3, shuffle=True)
for dataset in MyDataLoader:
    print(dataset)
```

Dataset에서는 data들을 홀드하고 있고 DataLoader에서 iterator 형식으로 데이터를 순차적으로 넘겨준다.

- batch_size : 몇 개의 데이터를 뽑아서 건네줄건지
- shuffle : 데이터를 랜덤으로 뽑아서 건네줄 건지

이외에도 많은 파라미터가 있는데 (데이터를 어떻게 뽑을건지, 몇개씩 로드할건지 등등) 아래 공식문서를 참고해보면 될 것 같습니다.

[torch.utils.data — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)

# 모델 불러오기

학습 도중에 비정상종료된다면 (colab 같은 애들 오래두면 런타임 끊어지는것 과 같은 문제) 학습데이터가 날라가기 때문에 도중에 저장할 필요가 있습니다.

## torch.save()

`torch.save()` 함수를 통해 딥러닝 모델에 관련된 객체를 저장할 수 있습니다. 파이썬에서 피클 형태로 객체를 저장하는 형식과 비슷합니다.

```python
# 객체 하나만 바로 저장하는 경우
torch.save(model, os.path.join(MODEL_PATH, "model_pickle.pt"))

# Ordered Dictionary로 여러 객체들을 저장하는 경우
torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        }, f"saved/checkpoint_model_{e}_{epoch_loss/len(dataloader)}_{epoch_acc/len(dataloader)}.pt")

# 어떤 경우이던 간에 torch.save 함수 안에 필수 파라미터로 객체, 파일명 들어가는건 동일
```

파라미터로 object와 파일명이 들어가는데 object에는 model 자체만 저장하는것, dictionary형태로 여러개의 객체를 저장하는것도 가능합니다. (Ordered Dictionary 자료형을 저장되는데 이도 객체기 때문입니다. )

## torch.load()

`torch.load()`  함수로 해당 파일을 객체로 가져올 수 있습니다.

```python
# 모델 자체를 저장한 경우 위와같이 바로 모델에 할당가능
model = MyNewModel()
model = torch.load(os.path.join(MODEL_PATH, "model_pickle.pt"))

# Ordered Dictionary 형태로 여러 객체가 저장되어 있는 경우 model.load_state_dict 거쳐야 함
checkpoint = torch.load('saved/checkpoint_epoch0010_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict']])
```

다음의 형식으로 저장되고 가져올 수 있습니다.

```python
new_model.load_state_dict(torch.load(os.path.join(
    MODEL_PATH, "model.pt")))
print(new_model)

# TheModelClass(
#   (layer1): Sequential(
#     (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2))
#     (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (layer2): Sequential(
#     (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
#     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (layer3): Sequential(
#     (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU()
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (drop_out): Dropout(p=0.5, inplace=False)
#   (fc1): Linear(in_features=576, out_features=1000, bias=True)
#   (fc2): Linear(in_features=1000, out_features=1, bias=True)
# )
```

# Monitoring tools

## Tensorboard

> TensorFlow의 프로젝트로 만들어진 시각화 도구
> 
- scalar : metric 등 상수값의 연속(epoch)을 표시
- graph : 모델의 computational graph 표시
- histogram : weight 등 값의 분포를 표현
- image : 예측값과 실제값을 비교표시
- mesh : 3D 형태의 데이터를 표현하는 도구

## Wandb

weight and bias라고 부르며 딥러닝 실험의 추적, 모델의 비교, 하이퍼파라미터 튜닝 등을 위해 사용됩니다. 

다양한 딥러닝 프레임워크를 지원하며, 학습 중인 모델의 메트릭 및 하이퍼파라미터를 실시간으로 추적하여 시각화할 수 있습니다. 

# Multi GPU 학습

- 모델을 나누거나
- 데이터를 나누는 방식으로 진행

모델을 나누는 방식은 옛날 (Alex-net) 부터 사용했지만 병목, 파이프라인 등의 어려움으로 인해 고난도 과제였습니다.

데이터를 나누는 방식은 예를들어 100 batch 짜리 데이터가 있을 때 각각 50 batch씩 나누어서 학습한 다음 나온 미분값을 평균내어 각각 업데이트하여 진행하는 방식입니다.

## DP vs DDP

**DP (Data Parallel)**

replicate → scatter → parallel_apply → gather 순서대로 진행합니다. Gather가 하나의 gpu로 각 모델의 출력을 모아주기 때문에 하나의 gpu의 메모리 사용량이 많을 수 밖에 없습니다.

데이터 병렬 처리를 사용하므로 데이터 간 통신 비용이 증가하고, GPU의 개수가 많아질수록 학습 속도가 느려질 수 있습니다.

**DDP (Distributed Data Parallelism)**

모델 병렬 처리 방식을 사용하여 여러 기기에서 모델의 파라미터를 분할하여 처리하는 기술입니다. 이 방식은 모델의 크기가 큰 경우에 적합하며, 다수의 GPU를 사용하여 효율적으로 모델을 학습할 수 있습니다. DDP는 데이터 병렬 처리보다 효율적이며, 모델이 큰 경우에는 데이터 병렬 처리 방식보다 더 나은 성능을 보입니다.

그러나 DP 방식보다 구현이 조금 어려운 단점이 있습니다.

# Hyper Parameter Tuning

AI 성능에 영향을 미치는 요소는 아래 3 가지가 있습니다.

1. 모델
2. 데이터셋
3. 하이퍼파라미터 튜닝

이중 모델은 각 분야에 가장 효율적이고 최적화되어있는 모델이 정형화되어있는 경우가 많으므로 AI 성능에 가장 큰 영향을 미치는 것은 데이터셋이라고 할 수 있습니다.

→ 데이터를 많이 모으고 전처리를 잘 할 수록 좋은 성능의 모델을 만들 수 있습니다.

## Hyper Parameter Tuning이란?

하이퍼 파라미터 튜닝으로 얻는 성능향상은 사실 드라마틱하게 크지 않습니다.

마지막 효율 한 방울까지 쥐어짜낼 때 쓰는 기법이라고 생각하시면 됩니다.

- learning rate, 모델의 크기, optimizer 와 같이 모델이 스스로 학습하지 않는 값들을 사람이 직접 지정해주어야 합니다.
- 요즘은 조금 덜 하지만 하이퍼파라미터 값에 의해 성능이 매우크게 변동되는 경우도 있습니다.

**그러나..! 보편적으로 투자시간대비 성능향상률이 크지 않은 경우가 보편적이므로 마지막 수단으로 사용하는 것을 권장합니다.**

AI 성능을 올리고싶다면 ***“어떻게 해야 좋은 데이터를 더 많이 모을 것인가***” 에 대한 내용을 고려해보는 것이 우선이겠죠?

**GridSearch & RandomSearch**

batch_size를 32, 64, 128 … 로 늘리고, learning_rate를 0.1, 0.01, 0.001 로 변경시키는 것처럼 일정한 간격을 가지고 search하는 것을 GridSearch라고 합니다.

반면에 랜덤으로 값을 대입하다가 가장 잘 나오는 값을 찾는 것을 RandomSearch라고 합니다.

**OldSchool**에서의 ****하이퍼파라미터 튜닝은 넓은 범위의 RandomSearch를 수행하다가 값이 가장 잘 나오는 구간에서 GridSearch로 세밀하게 값을 조정하는 방식을 사용했습니다.

## Ray

<aside>
⚠️ ML/DL 의 병렬처리를 지원하기 위해 개발된 모듈이며 기본적으로 현재 분산병렬 ML/DL 모듈의 표준입니다.
Hyperparameter Search를 위한 다양한 모듈을 제공하며 그 외에도 파이썬의 일반적인 병렬처리 또한 Ray를 사용하기도 합니다.

</aside>

[Ray Tune을 이용한 하이퍼파라미터 튜닝](https://tutorials.pytorch.kr/beginner/hyperparameter_tuning_tutorial.html)

Ray를 사용한 하이퍼파라미터 튜닝은 네 가지 절차를 통해 수행됩니다.

1. config에 search space 지정
2. 학습 스케줄링 알고리즘 지정
3. 결과출력 양식 지정
4. 병렬처리 양식으로 학습진행

- 학습 스케줄링 알고리즘?
    
    다양한 하이퍼파라미터 셋에 대하여 병렬처리 방식으로 학습을 진행하고 결과를 관찰하기 때문에 모든 경우의 수에 대해 끝까지 학습을 진행하면 리소스가 낭비될 수 있습니다.
    
    따라서 학습스케줄링 알고리즘에 따라 중간중간 퍼포먼스가 잘 나오지 않아 가망이 없다고 판단되는 파라미터셋을 제외시키고 계속해서 학습을 진행시키게 됩니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/648c1c6a-9798-459a-afaa-98e085e462fe/Untitled.png)
    

```python
data_dir = os.path.abspath("./data")
    load_data(data_dir)
		
		# config에 search space 지정
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),  # Grid Search를 하는 예시입니다.
        "batch_size": tune.choice([2, 4, 8, 16])
    }

		# 학습스케줄링 알고리즘 지정
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

		# 결과출력 양식 지정
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
		# 병렬처리 양식에 따라 학습진행
    result = tune.run(
				# 하이퍼파라미터 튜닝 간에 모델을 학습하는 전체 과정을 하나의 함수로 넣어주어야 합니다. (train_cifar)
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial}왜, 어
```

# 공포의 단어 OOM (Out Of Memory)

**why? 😥**

- 왜, 어디서 발생했는지 알기 어려움
- Error backtracking이 이상한데로 감
- 메모리의 이전상황을 파악하기 어려움
    
    → 보통 OOM 은 iteration이 돌던 중간에 발생하기 때문에 어디로 딱 찾아가서 고쳐내기 어렵다.
    

### **일단 1차원적이고도** **보편적인 (그러나 대부분 해결가능한) 해결방법**

Batch Size 줄인다음에 re-launch, re-run

## 그러나 무지성 해결법 말고 좀 제대로 알고 고치고 싶다!

### **GPUtil 사용하기**

> nvidia-smi 처럼 Colab 환경에서 GPU의 상태를 보여주는 모듈입니다. iter 마다 메모리가 늘어나는지 실시간으로 확인 가능합니다.
> 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45c24d50-42e4-4230-9328-96dee066fd37/Untitled.png)

### **torch.cuda.empty_cache() 써보기**

> 보통 back propagation이 수행될 때 메모리를 더 많이 사용하므로 임시 데이터가 저장이 되는데  계속 누적되어 사용하지 않는 메모리가 쌓이는 경우 이 함수를 사용하면 GPU상 cache를 정리하여 가용 메모리를 확보할 수 있습니다.
> 
- del 과의 다른 점
    
    del은 메모리 자체를 날리는게 아니라 관계를 끊어서 free시키는 기능을 하지만 위 함수는 re-launch 하는것과 같이 메모리를 초기화하는 효과가 있습니다.
    
    즉 del을 사용하면 메모리가 남아있는 상태로 둥둥 떠다니다가 Garbage Collector에 의해 불특정한 시점에 삭제된다.
    
    empty_cache()를 하면 즉시 clear 된다.
    

**GPUtil 을 사용하여 empty_cache() 함수 동작 시각화해보기**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41ca8ced-d335-4e1e-af25-b5c80697459b/Untitled.png)

**보통 학습시작 전 무거운 iteration이 돌기 전에 한번 써주면 좋습니다.**

### Training loop에 tensor로 축적되는 변수 확인하기

- tensor로 처리된 변수는 기본적으로 GPU상에 메모리로 올라갑니다. 심지어 back propagation에 사용하기 위해 required_gradient를 선언해주면 메모리 버퍼까지 사용됩니다.
- loop 안에서 tensor 를이용한 반복된 연산을 수행할 경우 computational graph가 생성되며 메모리를 잠식합니다.

**back propagation에서 총 손실량을 구하는 상황을 가정해봅시다**

```python
# 미분 가능하지만 쓸데없이 GPU 메모리를 잡아먹는 코드
total_loss = 0
for i in range(1000000):
		optimizer.zero_grad()
		ouput = model(input)
		loss = criterion(output)    # loss는 tensor로 반환됩니다.
		loss.backward()
		optimizer.step()
		total_loss += loss    # 1000000 번의 연산에 tensor가 사용됨
```

위 loop에서 criterion으로 계산된 loss는 tensor입니다.

이를 total_loss에 더하기 위해 그대로 사용하면 computational graph가 생성되면서 1000000개의 tensor가 GPU 메모리에 누적되기 때문에 OOP가 발생할 확률이 올라갑니다.

**loss 는 1-d tensor이기 때문에 굳이 tensor로 연산에 활용할 필요없이 파이썬의 기본객체로 변환하여 계산에 사용한다면 GPU의 메모리를 사용하지 않고 연산에 활용할 수 있습니다.**

```python
# 맨 밑에 한 줄만 바꾸면 돼요
total_loss = 0
for i in range(1000000):
		optimizer.zero_grad()
		ouput = model(input)
		loss = criterion(output)    # loss는 1-d & scalar tensor로 반환됩니다.
		loss.backward()
		optimizer.step()
		total_loss += loss.item    # 굳이 tensor로 안쓰고 기본객체로 변환합니다
		# total_loss += float(loss)  # 이렇게 해도 됩니다. (기본 float 객체로 변환)
```

### del 명령어 적절히 사용하기

- python의 메모리 배치 특성상 loop이 끝나도 메모리를 차지하기 때문에 반복문 밖에서도 사용가능한 특징이 있습니다.

```python
for x in range(5):

# 반복문 밖에서
i = x
print(i)   # 4 
```

위 코드처럼 반복문 안의 변수가 밖에서도 참조됩니다. 이는 iteration이 끝나도 메모리에 변수가 남아있다는 뜻입니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd308876-c0ec-4684-821e-8bb305050b87/Untitled.png)

그럼 만약에 반복문 안에 큰 사이즈의 모델이 들어있다면 반복문이 종료되고 더이상 사용하지 않는 변수가 큰 메모리를 쓸데없이 차지하고 있을 것입니다.

```python
# 이런 식으로 for 문을 돌려서 최종 result를 도출한 다음 밖에서 그냥 return이 가능합니다.
for i in range(5):
		intermediate = f(input[i])
		result = g(intermediate)
output = h(result)
return output
```

반복문 안에서 큰 모델 intermediate 에 대한 연산을 반복수행하여 최종 결과 result를 반환하는 코드를 위와같이 작성해도 잘 돌아갑니다. 

그러나 intermediate와 result 변수는 함수의 return이 끝나도 계속해서 메모리에 남아있게 됩니다.

del 함수를 적절히 사용해서 garbage collector가 메모리를 정리할 수 있게 하는것도 좋은 메모리관리 방법입니다.

```python
# 적절한 시점에 del 명령어를 통해 필요없어진 메모리를 free 해줍시다.
for i in range(5):
		intermediate = f(input[i])
		result = g(intermediate)
		del intermediate
output = h(result)
del result
return output
```

### 가능 batch size 실험해보기

OOM이 발생한 경우를 캐치해서 batch_size를 1로 돌려보는 코드입니다.

이런 방법을 사용해서 oom이 발생한 경우 batch_size가 어디까지 가능한지 체크해보는 것도 하나의 방법입니다.

```python
oom = False
try:
		run_model(batch_size)
except RuntimeError:  # Out of memory
		oom = True

if oom:
		for _ in range(batch_size):
				run_model(1)
```

### torch.no_grad() 사용하기

- Inference 시점에서는 torch.no_grad() 를 사용합시다.
- backward pass로 인해 쌓이는 메모리에서 자유롭습니다.

아래처럼 no_grad()함수를 사용할 수 있고 이 구문 안에서 일어나는 과정은 backward가 pass 됩니다.

```python
with torch.no_grad():
		for data, target in test_loader:
				output = network(data)
				test_loss = F.nll_loss(output, target, size_average=False).item()
				pred = output.data.max(keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
```

**Inference, 혹은 학습이 끝나고 테스트하는 시점에서는 꼭 torch.no_grad()를 사용합시다**

### 그외 꿀팁

- colab에서는 너무 큰 사이즈의 모델을 사용하지 맙시다. (LSTM 같은 애들은  backward propagation이 굉장히 긴 computational graph를 가집니다.)
- CNN의 대부분의 에러는 크기가 안맞아서 생기는 경우입니다. (torchsummary 등으로 사이즈를 맞춰보세요)
- tensor의 float precision을 16bit로 줄이는 것을 고려해볼 수 있습니다.