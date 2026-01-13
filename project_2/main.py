import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 연산 장치 설정 (GPU 사용 가능 여부 확인)
# CUDA(NVIDIA GPU)가 사용 가능하면 'cuda', 아니면 'cpu'를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 하이퍼파라미터 설정
batch_size = 64      # 한 번에 학습할 데이터의 개수
num_epochs = 20      # 전체 데이터를 몇 번 반복해서 학습할지 결정
noise_factor = 0.5   # 원본 이미지에 섞을 노이즈의 강도 (0~1 사이)
learning_rate = 1e-3 # 가중치를 업데이트하는 속도 (학습률)

# 3. 성능 평가 함수 정의
def calculate_score(mse_loss):
    """
    모델의 성능인 MSE(평균 제곱 오차)를 점수(0~100)로 변환하는 함수입니다.
    MSE가 낮을수록(복원이 잘 될수록) 높은 점수가 나옵니다.
    """
    score = 100 - (mse_loss * 1000)
    return max(0.0, score) # 점수는 최소 0점 이상이 되도록 설정

# 4. 데이터 로드 및 전처리
# transforms.ToTensor(): 이미지 데이터를 0~1 사이의 값을 가진 PyTorch 텐서로 변환합니다.
transform = transforms.Compose([transforms.ToTensor()])

# MNIST 학습 데이터셋 로드 (손글씨 숫자 데이터)
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# MNIST 테스트 데이터셋 로드 (학습에 사용되지 않는 평가용 데이터)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader: 데이터를 셔플(섞기)하고 배치 사이즈만큼 나눠서 제공해주는 도구
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. 모델 구조 정의 (Convolutional Autoencoder)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # [Encoder] 입력 이미지의 핵심 특징을 추출하는 과정
        self.encoder = nn.Sequential(
            # 입력 채널: 1 (흑백), 출력 채널: 32, 필터 크기: 3x3
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),           # 비선형 활성화 함수
            nn.BatchNorm2d(32),  # 학습을 안정화하고 속도를 높이는 배치 정규화
            # 출력 채널을 64로 늘려 더 복잡한 특징을 학습
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # [Decoder] 추출된 특징을 바탕으로 원래의 깨끗한 이미지를 복원하는 과정
        self.decoder = nn.Sequential(
            # 채널 수를 다시 줄여나가며 이미지 구조를 복원
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 최종 출력 채널은 입력과 동일한 1 (이미지 복원)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            # Sigmoid: 픽셀 값을 원본과 같은 0~1 사이로 제한
            nn.Sigmoid() 
        )

    def forward(self, x):
        # 입력 데이터 x가 인코더와 디코더를 순차적으로 통과함
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 6. 모델 생성 및 손실함수, 최적화 알고리즘 설정
model = Model().to(device)                 # 모델을 연산 장치(GPU/CPU)로 전송
criterion = nn.MSELoss()                    # 손실함수: 예측값과 실제값의 차이를 제곱하여 평균 (MSE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 가중치 업데이트 알고리즘 (Adam)

# 7. 학습 단계 (Training)
print("학습 시작...")
model.train() # 모델을 학습 모드로 설정 (Batch Normalization 등이 활성화됨)

for epoch in range(num_epochs):
    train_loss = 0.0
    for data in train_loader:
        img, _ = data # MNIST는 (이미지, 라벨)을 반환하나 오토인코더에서는 라벨이 필요 없음
        img = img.to(device)

        # [데이터 변형] 원본 이미지에 가우시안 노이즈 추가
        # torch.randn_like: 원본과 같은 크기의 랜덤 텐서 생성
        noisy_img = img + noise_factor * torch.randn_like(img)
        # 0보다 작거나 1보다 큰 값을 0~1 범위로 조정
        noisy_img = torch.clamp(noisy_img, 0., 1.)

        # 1) 순전파 (Forward): 노이즈가 섞인 이미지를 모델에 입력
        output = model(noisy_img)

        # 2) 오차 계산: 복원된 이미지(output)와 원래 깨끗한 이미지(img) 비교
        loss = criterion(output, img)

        # 3) 역전파 (Backward): 오차를 바탕으로 각 가중치의 기울기(gradient) 계산
        optimizer.zero_grad() # 이전 루프의 기울기 초기화
        loss.backward()       # 역전파 수행
        optimizer.step()      # 가중치 업데이트

        train_loss += loss.item()

    # 에폭마다 평균 손실값 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}")

# 8. 모델 평가 단계 (Evaluation)
model.eval() # 모델을 평가 모드로 설정 (기울기 계산 중단 등 효율적 동작)
total_test_loss = 0.0

# 평가 시에는 기울기를 계산할 필요가 없어 메모리를 절약함
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.to(device)

        # 테스트 데이터에도 동일한 조건의 노이즈 추가
        noisy_img = img + noise_factor * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0., 1.)

        # 모델에 노이즈 이미지 입력 및 결과 생성
        output = model(noisy_img)
        loss = criterion(output, img) # 손실 계산
        total_test_loss += loss.item()

# 최종 평균 MSE 계산 및 점수 환산
avg_test_loss = total_test_loss / len(test_loader)
final_score = calculate_score(avg_test_loss)

print(f"최종 Test MSE Loss: {avg_test_loss:.5f}")
print(f"최종 점수 (Score): {final_score:.2f} / 100점")

# 9. 결과 시각화
model.eval()
with torch.no_grad():
    # 테스트 데이터 중 일부(배치 하나)를 가져옴
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)

    # 노이즈 주입 및 모델을 통한 복원 과정 수행
    noisy_imgs = images + noise_factor * torch.randn_like(images)
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
    outputs = model(noisy_imgs)

    # 시각화를 위해 데이터를 다시 CPU로 옮기고 넘파이 배열로 변환
    images = images.cpu().numpy()
    noisy_imgs = noisy_imgs.cpu().numpy()
    outputs = outputs.cpu().numpy()

    # Matplotlib을 이용한 결과 출력 (상단: 원본, 중간: 노이즈, 하단: 복원 결과)
    plt.figure(figsize=(10, 6))
    for i in range(5):
        # 1. 원본 이미지 출력
        ax = plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title('Original')

        # 2. 노이즈가 추가된 입력 이미지 출력
        ax = plt.subplot(3, 5, i + 1 + 5)
        plt.imshow(noisy_imgs[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title('Noisy Input')

        # 3. 모델이 복원한 결과 이미지 출력
        ax = plt.subplot(3, 5, i + 1 + 10)
        plt.imshow(outputs[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title('Denoised Output')

    plt.tight_layout()
    plt.show()