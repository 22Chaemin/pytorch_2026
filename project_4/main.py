import os
import glob
import copy
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# FLOPs/Params 계산 라이브러리
from thop import profile

# -------------------------------------------------------------------
# [설정] 디바이스 할당
# -------------------------------------------------------------------
# CUDA 사용 가능 시 GPU, 불가 시 CPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 사용 장치: {device}")


# ==================================================================
# 1. Transforms 정의 (전처리)
# ==================================================================
# 모델 출력층(Tanh) 범위인 [-1, 1]에 맞춰 정규화 수행

# [Input] 흑백 이미지
# 1채널 -> (mean=0.5, std=0.5) 정규화
transform_input = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# [Target] 컬러 이미지 (Ground Truth)
# 3채널 -> RGB 각각 (0.5, 0.5) 정규화
transform_target = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ==================================================================
# 2. Dataset 클래스
# ==================================================================
class ColorizationDataset(Dataset):
    """
    흑백/컬러 이미지 쌍 로드 데이터셋
    mode='train': (흑백, 컬러) 반환
    mode='test' : (흑백, 파일명) 반환 (저장용)
    """
    def __init__(self, black_dir, color_dir, mode='train'):
        self.mode = mode
        # 파일 목록 로드 및 정렬
        self.black_files = sorted(glob.glob(os.path.join(black_dir, "*")))
        
        if mode == 'train':
            self.color_files = sorted(glob.glob(os.path.join(color_dir, "*")))
            # 데이터 개수 일치 확인
            assert len(self.black_files) == len(self.color_files), "이미지 개수 불일치"

    def __len__(self):
        return len(self.black_files)

    def __getitem__(self, idx):
        # 1. 흑백 이미지 로드 (Input)
        # 강제 1채널 변환 (.convert("L"))
        img_black = Image.open(self.black_files[idx]).convert("L")
        img_black = transform_input(img_black)
        
        # 2. 모드별 반환값 분기
        if self.mode == 'train':
            # 학습용: 정답 이미지 포함
            img_color = Image.open(self.color_files[idx]).convert("RGB")
            img_color = transform_target(img_color)
            return img_black, img_color
        else:
            # 추론용: 저장 위해 파일명 반환
            filename = os.path.basename(self.black_files[idx])
            return img_black, filename


# -------------------------------------------------------------------
# [경로 설정]
# -------------------------------------------------------------------
# 실행 파일 기준 상대 경로 사용 (이식성 확보)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')

print(f"데이터셋 경로: {data_dir}")

train_dir_b = os.path.join(data_dir, 'train_black')
train_dir_c = os.path.join(data_dir, 'train_color')
test_dir_b  = os.path.join(data_dir, 'test_black')
test_dir_c  = os.path.join(data_dir, 'test_color')

# 결과 및 체크포인트 저장 경로
result_dir = os.path.join(current_dir, 'results', 'eval')
checkpoint_dir = os.path.join(current_dir, 'checkpoints')

# -------------------------------------------------------------------
# [DataLoader 생성]
# -------------------------------------------------------------------
# 학습 데이터 로더
if os.path.exists(train_dir_b) and os.path.exists(train_dir_c):
    train_dataset = ColorizationDataset(train_dir_b, train_dir_c, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(f"학습 데이터: {len(train_dataset)}장")
else:
    print("경로 없음: 학습 로더 생성 건너뜀")
    train_loader = None

# 테스트 데이터 로더
if os.path.exists(test_dir_b):
    test_dataset = ColorizationDataset(test_dir_b, None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"테스트 데이터: {len(test_dataset)}장")
else:
    print("경로 없음: 테스트 로더 생성 건너뜀")
    test_loader = None


# ==================================================================
# 3. Model 정의 (U-Net)
# ==================================================================
class UNet(nn.Module):
    """
    Encoder-Decoder 구조
    특징 추출(Down) -> Bottleneck -> 특징 복원(Up + Skip Connection)
    """
    def __init__(self):
        super(UNet, self).__init__()

        # --- Encoder (Downsampling) ---
        self.enc1 = self.conv_block(1, 64)      # 1채널 입력
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2) # 크기 1/2 축소

        # --- Bottleneck ---
        self.bottleneck = self.conv_block(512, 1024)

        # --- Decoder (Upsampling) ---
        # Upsampling Layers
        self.up4 = self.up_block(1024, 512)
        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up1 = self.up_block(128, 64)
        
        # Conv Layers after Concat
        # __init__ 내 정의 필수 (GPU 할당 문제 방지)
        # 입력 채널 = Up채널 + Skip채널
        self.dec4 = self.conv_block(1024, 512) 
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        
        # Final Output
        # 3채널(RGB) 출력, Tanh로 -1~1 범위 매핑
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_c, out_c):
        """Conv -> BN -> ReLU 반복 블록"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_c, out_c):
        """Transposed Conv (업샘플링)"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # [Encoder]
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # [Bottleneck]
        b = self.bottleneck(p4)
        
        # [Decoder + Skip Connection]
        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1) # Skip Connection 결합
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# 모델 생성 및 GPU 이동
model = UNet().to(device)
print("모델 생성 완료")


# ==================================================================
# 4. Loss & Optimizer
# ==================================================================
# 손실 함수: L1 Loss (L2보다 선명한 결과 생성)
criterion = nn.L1Loss() 

# 최적화: Adam (생성 모델 표준 설정)
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))


# ==================================================================
# 5. Training Loop (학습)
# ==================================================================
def train_model(model, loader, epochs=20):
    print("학습 시작...")
    model.train() # 학습 모드
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for images, targets in loader:
            images = images.to(device)   # Input
            targets = targets.to(device) # Target
            
            # 1. 기울기 초기화
            optimizer.zero_grad()
            
            # 2. 예측
            outputs = model(images)
            
            # 3. 손실 계산
            loss = criterion(outputs, targets)
            
            # 4. 역전파
            loss.backward()
            
            # 5. 가중치 갱신
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
        
        # 체크포인트 저장 (5 epoch 마다)
        if (epoch+1) % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/unet_epoch_{epoch+1}.pth")
            
    print("학습 완료")


# ==================================================================
# 6. Evaluation Utils (평가/시각화)
# ==================================================================

def print_model_complexity(model, input_size=(1, 1, 256, 256)):
    """FLOPs 및 파라미터 수 계산 (Deepcopy 사용)"""
    device = next(model.parameters()).device
    
    # 모델 복사 및 장치 이동 (원본 영향 방지)
    temp_model = copy.deepcopy(model)
    temp_model.to(device)
    temp_model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # thop 프로파일링
    flops, params = profile(temp_model, inputs=(dummy_input, ), verbose=False)

    print(f"[모델 복잡도]")
    print(f" - FLOPs  : {flops/1e9:.3f} G")
    print(f" - Params : {params/1e6:.3f} M")
    print("-" * 30)
    
    del temp_model

def calculate_fid(result_dir, gt_dir):
    """FID Score 측정 (pytorch-fid 호출)"""
    print(f"[FID 측정]")
    device_str = str(device).split(':')[0]
    
    cmd = [
        "python", "-m", "pytorch_fid",
        result_dir, gt_dir,
        "--device", device_str
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"FID 계산 실패: {e}")
    print("-" * 30)

def visualize_result(model, loader, save_dir, num_samples=3):
    """추론 결과 저장 및 샘플 시각화"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 샘플 배치 확보
    images, filenames = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        preds = model(images)
        
        # 전체 데이터 추론 및 저장 (FID용)
        print("전체 테스트 데이터 저장 중...")
        for batch_imgs, batch_files in loader:
            batch_imgs = batch_imgs.to(device)
            batch_preds = model(batch_imgs)
            batch_preds = batch_preds * 0.5 + 0.5 # Denormalize
            
            for i in range(len(batch_preds)):
                from torchvision.utils import save_image
                save_image(batch_preds[i], os.path.join(save_dir, batch_files[i]))
                
    # --- 시각화 (Matplotlib) ---
    images_cpu = images.cpu()
    preds_cpu = preds.cpu()
    
    plt.figure(figsize=(10, num_samples * 3))
    
    for i in range(num_samples):
        # Input 처리 (1채널)
        img_in = images_cpu[i].squeeze()
        img_in = img_in * 0.5 + 0.5
        
        # Output 처리 (3채널)
        img_out = preds_cpu[i].permute(1, 2, 0)
        img_out = img_out * 0.5 + 0.5
        img_out = torch.clamp(img_out, 0, 1)
        
        # Plot
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(img_in, cmap='gray')
        plt.title("Input (Black)")
        plt.axis('off')

        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(img_out)
        plt.title("Output (Colorized)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==================================================================
# 7. Main (실행)
# ==================================================================
if __name__ == "__main__":
    
    # 1. 모델 복잡도 확인
    print_model_complexity(model)
    
    # 2. 학습 (데이터 존재 시)
    if train_loader:
        train_model(model, train_loader, epochs=5)
    
    # 3. 테스트 및 평가 (데이터 존재 시)
    if test_loader:
        # 결과 저장 및 시각화
        visualize_result(model, test_loader, result_dir, num_samples=3)
        
        # FID 측정 (GT 존재 시)
        if os.path.exists(test_dir_c):
            calculate_fid(result_dir, test_dir_c)