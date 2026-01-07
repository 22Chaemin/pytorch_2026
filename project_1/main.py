import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 1. 모델 아키텍처 정의 (ImprovedCNN)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # 특징 추출부 (Feature Extraction)
        self.features = nn.Sequential(
            # 첫 번째 블록: 3x32x32 -> 64x32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # 배치 정규화: 학습 속도 향상 및 안정화
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 크기 절반 감소: 64x16x16

            # 두 번째 블록: 64x16x16 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 크기 절반 감소: 128x8x8

            # 세 번째 블록: 특징 심화
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 전역 평균 풀링: 입력 이미지 크기에 상관없이 1x1 크기로 압축
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        # 분류기 (Classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # 과적합 방지를 위한 드롭아웃 (50%)
            nn.Linear(256, 10) # 256개 특징을 10개 클래스로 매핑
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # 4D 텐서를 2D로 펼침 (배치 사이즈, 특징수)
        x = self.classifier(x)
        return x

def main():
    # 학습 장치 설정 (GPU 사용 가능 시 CUDA, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 중인 장치: {device}")

    # 하이퍼파라미터 설정
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    EPOCHS = 10

    # 2. 데이터 전처리 및 증강 (Data Augmentation)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 무작위 좌우 반전
        transforms.RandomCrop(32, padding=4), # 무작위 상하좌우 패딩 후 자르기
        transforms.ToTensor(), # Tensor 형태로 변환
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 정규화
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # CIFAR-10 데이터셋 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 모델 생성 및 장치 이동
    model = ImprovedCNN().to(device)

    # (선택 사항) 연산량 및 파라미터 수 확인
    try:
        from thop import profile
        input_dummy = torch.randn(1, 3, 32, 32).to(device)
        flops, params = profile(model, inputs=(input_dummy, ), verbose=False)
        print(f"FLOPs: {flops/1e6:.2f}M, Params: {params/1e6:.2f}M")
    except ImportError:
        pass

    # 3. 손실 함수 및 최적화 도구 설정
    criterion = nn.CrossEntropyLoss() # 다중 분류를 위한 교차 엔트로피
    # SGD(확률적 경사 하강법) + Momentum + Weight Decay(L2 정규화)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # 학습률 스케줄러: 지정된 에폭(5, 8)에서 학습률을 0.1배로 감소
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

    # 4. 학습 루프
    print("학습 시작...")
    for epoch in range(EPOCHS):
        model.train() # 모델을 학습 모드로 설정
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()    # 변화도(Gradient) 초기화
            outputs = model(inputs)  # 순전파 (Forward)
            loss = criterion(outputs, labels) # 손실 계산
            loss.backward()          # 역전파 (Backward)
            optimizer.step()         # 가중치 업데이트
            
            running_loss += loss.item()
        
        scheduler.step() # 에폭 종료 후 학습률 조정
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(trainloader):.4f}")

    # 5. 테스트 루프 (모델 평가)
    model.eval() # 모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 동작 방식 변경)
    correct = 0
    total = 0
    with torch.no_grad(): # 평가 시에는 기울기 계산을 하지 않음 (메모리 절약)
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # 가장 높은 확률을 가진 인덱스 추출
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"최종 정확도: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()
