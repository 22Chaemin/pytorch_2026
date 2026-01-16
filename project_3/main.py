import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 사용 장치: {device}")

# 이미지: 224x224 리사이징 + 텐서 변환 + ImageNet 정규화
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 마스크: 224x224 리사이징
mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
])

class CustomPetDataset(OxfordIIITPet):
    def __init__(self, root, split, target_types, download, transform=None, target_transform=None):
        super().__init__(root=root, split=split, target_types=target_types, download=download, transform=None, target_transform=None)
        self.custom_transform = transform
        self.custom_target_transform = target_transform

    def __getitem__(self, idx):
        # 1. 부모 클래스를 통해 원본 이미지와 마스크를 가져오세요.
        img, mask = super().__getitem__(idx)

        # -------------------------------------------------------------------
        # [문제 1] 아래 로직을 직접 구현하세요.
        # 목표: 이미지는 전처리를 적용하고, 마스크는 '0(배경)'과 '1(동물)'로 변환해야 합니다.
        #
        # 1. self.custom_transform을 이미지에 적용하세요.
        # 2. self.custom_target_transform을 마스크에 적용하세요.
        # 3. 마스크를 np.array를 통해 텐서(float32)로 변환하세요.
        # 4. 라벨 변환: 원본 마스크의 {1:동물, 2:배경, 3:테두리} 값을 -> {1:동물, 0:나머지}로 변경하세요.
        #    (힌트: torch.where 또는 불리언 인덱싱 사용)
        # 5. 마스크의 차원을 (H, W) -> (1, H, W)로 늘려주세요. (unsqueeze 사용)
        # -------------------------------------------------------------------

        # (여기에 코드를 작성하세요)
        
        # self.custom_transform을 이미지에 적용한다.
        if self.custom_transform:
            img = self.custom_transform(img)

        # self.custom_target_transform을 마스크에 적용한다.
        if self.custom_target_transform:
            mask = self.custom_target_transform(mask)

        # 마스크를 np.array를 통해 텐서로 변환한다.
        mask = np.array(mask)
        mask = torch.from_numpy(mask).float()

        # 원본 마스크의 값을 라벨 변환한다.
        # torch.where를 사용하여 1은 1로, 나머지는 0으로 변경한다.
        mask = torch.where(mask == 1.0, 1.0, 0.0)

        # 마스크의 차원을 늘린다.
        mask = mask.unsqueeze(0)

        return img, mask

# 데이터셋 다운로드 및 로드
print("데이터셋 준비 중...")
train_dataset = CustomPetDataset(root='./data', split='trainval', target_types='segmentation', download=True, transform=img_transform, target_transform=mask_transform)
test_dataset = CustomPetDataset(root='./data', split='test', target_types='segmentation', download=True, transform=img_transform, target_transform=mask_transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"학습 데이터: {len(train_dataset)}장")
print(f"테스트 데이터: {len(test_dataset)}장")

# 샘플 시각화 함수
def show_sample(loader):
    imgs, masks = next(iter(loader))

    plt.figure(figsize=(10, 5))

    img_show = imgs[0].permute(1, 2, 0)
    img_show = img_show * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img_show = torch.clamp(img_show, 0, 1)

    # 이미지 출력
    plt.subplot(1, 2, 1)
    plt.imshow(img_show)
    plt.title("Input Image")
    plt.axis('off')

    # 마스크 출력
    plt.subplot(1, 2, 2)
    plt.imshow(masks[0].squeeze(), cmap='gray')
    plt.title("Target Mask (White: Pet)")
    plt.axis('off')

    plt.show()

show_sample(train_loader)

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        # Pre-trained ResNet18 불러오기
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        # -------------------------------------------------------------------
        # [문제 2-1] ResNet18 기반의 Encoder와 U-Net Decoder를 정의하세요.
        #
        # <힌트: ResNet 레이어 매핑>
        # - layer0: base_layers[:3]
        # - layer1: base_layers[4]
        # - layer2: base_layers[5]
        # - layer3: base_layers[6]
        # - layer4: base_layers[7]
        #
        # <힌트: Decoder 채널 수 (Input -> Output)>
        # - upconv4: 512 -> 256
        # - upconv3: (256 + 256) -> 128  (Skip Connection 고려)
        # - upconv2: (128 + 128) -> 64
        # - upconv1: (64 + 64) -> 64
        # - 최종 출력 레이어 (Conv2d, Kernel size=1)
        # -------------------------------------------------------------------

        # Encoder 부분 (ResNet 계층 활용)

        # layer0 정의 
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        # layer1 정의
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        # layer2 정의
        self.layer2 = self.base_layers[5]
        # layer3 정의
        self.layer3 = self.base_layers[6]
        # layer4 정의
        self.layer4 = self.base_layers[7]

        # Decoder 부분 (Upsampling)

        # Upsampling과 Convolution을 통해 채널 수 조정
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # 최종 출력
        # 224x224 크기로 맞춘 뒤 1채널 마스크 생성
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_class, kernel_size=1)
        )

    def forward(self, x):
        # -------------------------------------------------------------------
        # [문제 2-2] U-Net의 Forward Pass를 완성하세요.
        #
        # 1. Encoder 구간: layer0 -> layer1 -> ... -> layer4 순서로 통과시키세요.
        # 2. Decoder 구간: Skip Connection을 적용해야 합니다.
        #    - up4는 layer4를 입력으로 받음 -> 그 결과와 layer3를 torch.cat으로 결합
        #    - up3는 up4 결과를 입력으로 받음 -> 그 결과와 layer2를 결합
        #    - ... (반복)
        # 3. 마지막 출력을 얻고 Sigmoid를 적용하여 반환하세요.
        # -------------------------------------------------------------------

        # (여기에 코드를 작성하세요)

        # Encoder
        # ResNet의 각 계층을 통과하며 특징을 추출한다.
        l0 = self.layer0(x) # (64, 112, 112)
        l1 = self.layer1(l0) # (64, 56, 56)
        l2 = self.layer2(l1) # (128, 28, 28)
        l3 = self.layer3(l2) # (256, 14, 14)
        l4 = self.layer4(l3) # (512, 7, 7)

        # Decoder
        # l4를 7x7에서 14x14로 업샘플링 하고 l3와 결합
        up4 = self.upconv4(l4) # (256, 14, 14)
        up4 = torch.cat([up4, l3], dim=1) # (256 + 256, 14, 14)

        # up4를 14x14에서 28x28로 업샘플링 하고 l2와 결합
        up3 = self.upconv3(up4) # (128, 28, 28)
        up3 = torch.cat([up3, l2], dim=1) # (128 + 128, 28, 28)

        # up3를 28x28에서 56x56로 업샘플링 하고 l1과 결합
        up2 = self.upconv2(up3) # (64, 56, 56)
        up2 = torch.cat([up2, l1], dim=1) # (64 + 64, 56, 56)

        # up2를 56x56에서 112x112로 업샘플링 하고 l0과 결합
        up1 = self.upconv1(up2) # (64, 112, 112)
        up1 = torch.cat([up1, l0], dim=1) # (64 + 64, 112, 112)

        # 최종 출력
        # l0가 이미 112x112 크기이므로, 최종 결과를 뽑은 뒤 원본 크기로 복원
        output = self.final_conv(up1) # (n_class, 112, 112)

        # 112x112에서 224x224로 Input 크기로 업샘플링
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Sigmoid 적용
        return torch.sigmoid(output)


model = ResNetUNet().to(device)
print("모델 생성 완료")

# Dice Loss 정의
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# Loss 및 Optimizer 설정
criterion_bce = nn.BCELoss()
criterion_dice = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("학습 시작...")
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # -------------------------------------------------------------------
        # [문제 3] 학습 루프의 로직을 작성하세요.
        #
        # 1. Gradient 초기화 (optimizer.zero_grad)
        # 2. 모델 예측 (Forward)
        # 3. 손실 계산 (Loss = BCE + Dice)
        # 4. 역전파 (Backward)
        # 5. 가중치 업데이트 (optimizer.step)
        # -------------------------------------------------------------------

        # (여기에 코드를 작성하세요)

        # Gradient 초기화
        optimizer.zero_grad()

        # 모델 예측
        outputs = model(images)

        # 손실 계산
        # Loss = BCE + Dice (픽셀 단위 정확도 + 전체적인 형태 일치도)
        loss_bce = criterion_bce(outputs, masks)
        loss_dice = criterion_dice(outputs, masks)
        loss = loss_bce + loss_dice

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("학습 완료!")

from thop import profile

def print_model_complexity(model, input_size=(1, 3, 224, 224)):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    # thop를 사용하여 FLOPs와 Params 계산
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)

    flops_g = flops / 1e9
    params_m = params / 1e6

    print(f"[모델 복잡도 평가]")
    print(f"   - FLOPs  : {flops_g:.3f} GFLOPs")
    print(f"   - Params : {params_m:.3f} M")
    print("-" * 30)

# 1. Dice Score 계산 함수
def calculate_dice(pred, target, threshold=0.5):
    # 예측값을 이진화 (0 or 1)
    pred_bin = (pred > threshold).float()

    # 평탄화
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    # 교집합 및 분모 계산
    intersection = (pred_flat * target_flat).sum()
    dice_denominator = pred_flat.sum() + target_flat.sum()

    # 분모가 0이 되는 것을 방지
    epsilon = 1e-6
    dice = (2. * intersection + epsilon) / (dice_denominator + epsilon)

    return dice.item()

# 2. 전체 테스트 셋 평균 Dice Score 확인
def evaluate_model(model, data_loader):
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)

            # 배치 내의 각 이미지별로 점수 계산
            for i in range(len(images)):
                dice = calculate_dice(preds[i], masks[i])
                total_dice += dice
                count += 1

    avg_dice = total_dice / count

    print(f"[전체 테스트 셋 평가 결과]")
    print(f"   - 최종 평균 Dice Score: {avg_dice:.4f} ★")
    print("-" * 30)
    return avg_dice

# 3. 시각화 함수 (Dice Score 포함)
def visualize_result(model, data_loader, num_samples=3):
    model.eval()
    # 첫 번째 배치 가져오기
    images, masks = next(iter(data_loader))
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        preds = model(images)

    images_cpu = images.cpu()
    masks_cpu = masks.cpu()
    preds_cpu = preds.cpu()

    plt.figure(figsize=(12, num_samples * 4))

    for i in range(num_samples):
        # 개별 Dice 점수 계산
        dice = calculate_dice(preds[i], masks[i])

        # 이미지 역정규화
        img = images_cpu[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)

        # 예측 마스크 이진화
        pred_mask = (preds_cpu[i] > 0.5).float()

        # 시각화 배치
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks_cpu[i].squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        # 제목에 Dice Score만 강조해서 표시
        plt.title(f"Prediction\nDice Score: {dice:.4f}", color='red', fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 최종 실행 ---
# 모델 정보 출력
print_model_complexity(model)

# 1. 평균 Dice Score 출력
evaluate_model(model, test_loader)

# 2. 샘플 시각화
visualize_result(model, test_loader, num_samples=3)