# KADID
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def validate(args, model, dataloader, device):
    model.eval()  
    srocc_values, plcc_values = [], []  

    with torch.no_grad():  
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)  
            inputs_B = batch["img_B"].to(device)

            # 입력 데이터 차원 확인 / 조정
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            # 모델 출력값 계산
            proj_A, proj_B = model(inputs_A, inputs_B)

            # 결과 정규화 및 numpy 변환
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # SRCC 및 PLCC 계산
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            
            srocc_values.append(srocc)
            plcc_values.append(plcc)

    # 평균값 반환
    return np.mean(srocc_values), np.mean(plcc_values)


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)  
    best_srocc = -1  

    train_metrics = {'loss': []}  
    val_metrics = {'srcc': [], 'plcc': []} 
    test_metrics = {'srcc': [], 'plcc': []}  

    for epoch in range(args.training.epochs):
        model.train()  
        running_loss = 0.0 
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 입력 데이터 차원 조정
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            # 하드 네거티브 생성
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            print(f"[Debug] hard_negatives shape before processing: {hard_negatives.shape}")

            # 하드 네거티브 차원 조정
            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad() 

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)

                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_negatives = model.backbone(hard_negatives)
                print(f"[Debug] features_negatives shape after backbone: {features_negatives.shape}")

                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])
                elif features_negatives.dim() != 2:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            running_loss += loss.item()

            # 진행 상황 표시 업데이트
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # 검증 실행
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        # 테스트 실행
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad(): 
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 입력 데이터 차원 조정
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = KADID10KDataset(str(dataset_path))

    # 데이터셋 분할
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader 생성
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 모델 초기화
    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # 학습 실행
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )


    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)



# KADID & TID
""" import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset, TID2013Dataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics_per_epoch = []  # 에포크별 테스트 결과 저장

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_negatives = model.backbone(hard_negatives)
                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])  # Apply GAP
                elif features_negatives.dim() == 2:
                    pass  # Already reduced dimensions
                else:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)
                loss = model.compute_loss(proj_A, proj_B, proj_negatives)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        # Test at each epoch
        test_metrics = test(args, model, test_dataloader, device)
        test_metrics_per_epoch.append(test_metrics)  # 각 에포크 결과 저장

        # 평균 SRCC, PLCC 출력
        avg_srcc = np.mean(test_metrics['srcc'])
        avg_plcc = np.mean(test_metrics['plcc'])
        print(f"Epoch {epoch + 1}: Test SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")

    # 모든 에포크의 테스트 결과 출력
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics_per_epoch, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

    return train_metrics, val_metrics, test_metrics_per_epoch

def test(args, model, test_dataloader, device):
    if test_dataloader is None:
        raise ValueError("Test DataLoader가 None입니다. 초기화 문제를 확인하세요.")

    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # KADID10K 경로 설정 및 로드
    kadid_dataset_path = Path(str(args.data_base_path_kadid))
    print(f"[Debug] KADID10K Dataset Path: {kadid_dataset_path}")
    kadid_dataset = KADID10KDataset(str(kadid_dataset_path))

    # TID2013 경로 설정 및 로드
    tid_dataset_path = Path(str(args.data_base_path_tid))
    print(f"[Debug] TID2013 Dataset Path: {tid_dataset_path}")
    tid_dataset = TID2013Dataset(str(tid_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 테스트 데이터 로드
    test_dataloader = DataLoader(
        tid_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
    )

    # 모델 초기화
    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # CSIQ 훈련
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    # 최종 결과 출력
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}") """