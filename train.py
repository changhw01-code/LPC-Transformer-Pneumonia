import random
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataloader import get_dataloaders
from lpc_transformer import SwinTransformer, LMFLoss, SCION
from test import evaluate

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    batch_size = 32
    lr = 1e-4
    seed = 42
    epochs = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SwinTransformer(
        img_size=224, patch_size=4, in_chans=3, num_classes=11,
        embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
        window_size=7, mlp_ratio=4., drop_path_rate=0.1
    ).to(device)

    class_weights = [2.6, 1.5, 1.2, 1.3, 0.8, 0.4, 1.3, 0.3, 0.7, 0.3, 0.6]
    criterion = LMFLoss(class_weights=class_weights, ldam_factor=0.5, focal_gamma=2)
    optimizer = SCION(model.parameters(), lr=lr, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

    best_state = None
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += torch.sum(pred == label).item()
            train_total += label.size(0)
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        val_acc, avg_val_loss = evaluate(model, val_loader, criterion, device, is_test=False)
        print(f"Epoch {epoch+1:03d} | Train Loss:{avg_train_loss:.4f} Train Acc:{train_acc:.2f}% | Val Loss:{avg_val_loss:.4f} Val Acc:{val_acc:.2f}%")

        # 关键：自动保存最优权重到本地 best_model.pth
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, "best_model.pth")
            print(f"✅ 新最优模型已保存，验证精度：{best_acc:.2f}%")
        scheduler.step()

    # 训练结束加载最优权重执行全套测试绘图
    model.load_state_dict(best_state)
    evaluate(model, test_loader, criterion, device, is_test=True)

if __name__ == "__main__":
    main()
