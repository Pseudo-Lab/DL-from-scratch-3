# 나만의 딥러닝 프레임워크 만들기 (밑시딥3)

<div align="center">
<a href="https://pseudo-lab.com"><img src="https://img.shields.io/badge/PseudoLab-S10-3776AB" alt="PseudoLab"/></a>
<a href="https://discord.gg/EPurkHVtp2"><img src="https://img.shields.io/badge/Discord-BF40BF" alt="Discord Community"/></a>
<a href="https://github.com/Pseudo-Lab/10th-template/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/Pseudo-Lab/DL-from-scratch-3?color=2b9348"></a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPseudo-Lab%2FDL-from-scratch-3&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>
<br>

<!-- sheilds: https://shields.io/ -->
<!-- hits badge: https://hits.seeyoufarm.com/ -->


## 🌟 프로젝트 비전 (Project Vision)
_"나만의 미니멀 딥러닝 프레임워크, DeZero 만들기"_ 
<br/><br/>

### 🖐️ 밑바닥부터 시작하는 딥러닝 시리즈 소개
딥러닝의 눈부신 발전에는 파이토치, 텐서플로 등 이를 뒷받침하는 딥러닝 프레임워크들이 있었습니다. 딥러닝 프레임워크는 직관적인 인터페이스와 자동미분, 역전파 등 편리하고 강력한 기능들을 탑재하여 다양한 딥러닝 아이디어의 구현을 가능케합니다. 하지만 이는 역설적이게도 개발자들로 하여금 딥러닝의 원리에 대한 이해를 배제하고 딥러닝에 입문하게끔 합니다.

`밑바닥부터 시작하는 딥러닝`시리즈는 딥러닝 프레임워크에 의존하지 않고 딥러닝의 개념을 '밑바닥부터' 구현하며 그 동작원리를 블랙박스가 아닌 화이트박스로써 이해하도록 돕는 훌륭한 교재입니다. 딥러닝의 기초 이론들을 최소한의 라이브러리(numpy 등)만 사용하여 구현하고 경험하는 과정에서 여러분들은 전체적인 동작과정을 머릿속에서 그려낼 수 있는 힘을 길러낼 수 있습니다.
<br/><br/>

### 🤔 DeZero란?
`밑바닥부터 시작하는 딥러닝`시리즈의 3권은 딥러닝에 대한 심화 개념과 동시에 프레임워크 개발에 대해서도 다룹니다. 저자는 `DeZero`라는 오리지널 프레임워크를 설계하고 조립해나가며 `Define-by-Run`과 같은 현대적인 프레임워크들의 특징에 대해 이해할 수 있도록 합니다. (**De**ep Learning from **Zero**)

따라서 `DeZero`를 구현하는 과정에서 여러분은 단순히 딥러닝에 대해서가 아닌, **"딥러닝 프레임워크"**에 대해 이해하실 수 있습니다. 이를 통해 단순히 딥러닝 프레임워크의 동작원리뿐만 아니라, 내가 아닌 제3자를 고려하여 코드를 작성할 때 개발자의 고민사항에 대해 공감하실 수 있을 것입니다. 총 60단계로 이루어진 과정을 차근차근 밟아나가면 여러분들께서 직접 만드신 프레임워크로 다음과 같은 코드도 작성할 수 있습니다.
```python
import dezero
import dezero.functions as F
from dezero import DataLoader
from dezero.models import MLP

...

train_loader = DataLoader(train_set, batch_size)
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = dezero.optimizers.Adam().setup(model)
optimizer.add_hook(dezero.optimizers.WeightDecay(1e-4))  # Weight decay

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
```
<br/><br/>

### 📅 프로젝트 진행 방식
**1. 교재 리딩 및 코드 작성**
   - 해당 주차에 정해진 분량을 공부합니다.
   - 각자의 GitHub에 `DeZero` repository를 만들어 정해진 분량을 코딩해 commit 합니다.
   - 매주 일요일까지 commit 링크를 가짜연구소 GitHub에 공유합니다.

**2. TMI(Too-Much-Insights) 공유**
   - 공부하며 흥미로웠던 내용, 생각난 아이디어, 덧붙이고 싶은 지식 등 자신이 얻은 인사이트를 공유합니다.
   - 새로운 시도나 잘 안 되는 부분에 대해서도 공유할 수 있습니다.
   - 꼭 공부한 내용과 관련된 내용이 아닌 TMI도 좋습니다.
   - 이 또한 매주 일요일까지 위 commit 링크와 함께 가짜연구소 GitHub에 공유합니다.

**3. 월요일 저녁 디스코드 모임**
   - 매주 디스코드 모임을 진행할 Moderator를 선발합니다.
   - Moderator는 사회자를 담당하며, 주로 이번주의 TMI들을 소개합니다.
   - 모임은 정해진 진행방식 없이 그 날의 Moderator 맘대로 진행합니다!
<br/><br/>

### 🎁 프로젝트에서 챙겨가실 것들
- 딥러닝 기반 이론에 대한 이해
- 현대적 프레임워크의 동작 원리 이해
- 효율적인 파이썬 프로그래밍 방법 체득
- 규모 있고 체계적인 소프트웨어 개발 경험
<br/><br/>

## 🧑 역동적인 팀 소개 (Dynamic Team)

| 역할          | 이름 |  관심 분야                                                                 | Social                         |
|---------------|------|-----------------------------------------------------------------------|----------------------------------------|
| **Project Manager** | 고지호 | 머신러닝, 딥러닝, 수학적 최적화 | [LinkedIn](https://www.linkedin.com/in/jiho-ko-58b64a276) |
| **Member** | 박진경 | - | - |
| **Member** | 김도현 | - | - |
| **Member** | 김진영 | - | - |
| **Member** | 우다현 | - | - |
| **Member** | 김덕주 | - | - |
| **Member** | 김솔 | - | - |
| **Member** | 송윤호 | - | - |
| **Member** | 정지윤 | - | - |

<br/>

## 🛠️ 우리의 개발 문화 (Our Development Culture)
**우리의 개발 문화**  
```python
class CollaborationFramework:
    def __init__(self):
        self.tools = {
            'communication': 'Discord' | 'KakaoTalk',
            'project_management': 'GitHub',
            'ETC': 'TBD'
        }
    
    def workflow(self):
        return """주간 사이클:
        1️⃣ ~일요일: 각자 교재 리딩 및 DeZero 업데이트
        2️⃣ 일요일: 업데이트 commit 링크와 TMI를 가짜연구소 GitHub에 공유
        3️⃣ 월요일: 주마다 발표자가 내용 요약과 리딩 후기 공유"""
```
<br/><br/>

## 💻 주차별 활동 (Activity History)
**🕘모임 일정: 매주 월요일 오후 9시**

시간은 OT날 조정될 수 있습니다.


| 날짜 | 내용 | 발표자 | 
| -------- | -------- | ---- |
| 2025/03/03 | OT       | 고지호  |
| 2025/03/10 | 1단계 상자로서의 변수 ~ 5단계 역전파 이론  | 고지호 | 
| 2025/03/17 | 6단계 수동 역전파 ~ 10단계 테스트  | 김덕주 |
| 2025/03/24 | ⭐Magical Week⭐  | - | 
| 2025/03/31 | 11단계 가변 길이 인수 ~ 16단계 복잡한 계산 그래프 | 정지윤 | 
| 2025/04/07 | 17단계 메모리 관리와 순환 참조 ~ 22단계 연산자 오버로드  | 김진영 | 
| 2025/04/14 | 23단계 패키지로 정리 ~ 26단계 계산 그래프 시각화  | 미정 | 
| 2025/04/21 | 27단계 테일러 급수 미분 ~ 32단계 고차 미분  | 미정 | 
| 2025/04/28 | ⭐Magical Week⭐  | - | 
| 2025/05/05 | 어린이날  | - | 
| 2025/05/12 | 33단계 뉴턴 방법으로 푸는 최적화 ~ 36단계 고차 미분 이외의 용도  | 미정 | 
| 2025/05/19 | 37단계 텐서를 다루다 ~ 40단계 브로드캐스트 함수  | 미정 | 
| 2025/05/26 | 41단계 행렬의 곱 ~ 44단계 매개변수를 모아두는 계층  | 미정 | 
| 2025/06/02 | 45단계 계층을 모아두는 계층 ~ 47단계 소프트맥스 함수와 교차 엔트로피 오차  | 미정 | 
| 2025/06/09 | 48단계 다중 클래스 분류 ~ 52단계 GPU 지원  | 미정 | 
| 2025/06/16 | 53단계 모델 저장 및 읽어오기 ~ 56단계 CNN 메커니즘  | 미정 | 
| 2025/06/23 | 57단계 conv2d 함수와 pooling 함수 ~ 60단계 LSTM과 데이터 로더  | 미정 | 
| 2025/06/30 | 마무리 회고  | - | 

<br/>

## 📕 필수 교재 (Textbook)
**[밑바닥부터 시작하는 딥러닝3](https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922)**
<br/><br/>

## 🌱 참여 안내 (How to Engage)
### 수료 기준
- `출석 기준`: Discussion 작성 + 디스코드 모임 참여
- `수료 기준`: OT, Magical Week, 회고를 제외한 주에 10회 이상 출석 (결석 3회까지 인정)

### 누구나 청강을 통해 모임을 참여하실 수 있습니다.
- 특별한 신청 없이 정기 모임 시간에 맞추어 디스코드 #Room-AK 채널로 입장
- Magical Week 중 행사에 참가
- Pseudo Lab 행사에서 만나기
<br/><br/>

## About Pseudo Lab 👋🏼</h2>

[Pseudo-Lab](https://pseudo-lab.com/) is a non-profit organization focused on advancing machine learning and AI technologies. Our core values of Sharing, Motivation, and Collaborative Joy drive us to create impactful open-source projects. With over 5k+ researchers, we are committed to advancing machine learning and AI technologies.

<h2>Contributors 😃</h2>
<a href="https://github.com/Pseudo-Lab/10th-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Pseudo-Lab/DL-from-scratch-3" />
</a>
<br><br>

<h2>License 🗞</h2>

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
