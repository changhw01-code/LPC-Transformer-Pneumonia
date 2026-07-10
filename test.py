import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lpc_transformer import SwinTransformer
from dataloader import get_dataloaders
# 自动创建输出文件夹，防止保存图片报错
os.makedirs("./output", exist_ok=True)

target_names = [
    "CT Normal",
    "CT Omicron and Delta Variant",
    "CT Other Pneumonia",
    "CT Wildtype SARS-CoV-2",
    "X-Ray Bacterial Pneumonia",
    "X-Ray MERS",
    "X-Ray Normal",
    "X-Ray Omicron and Delta Variant",
    "X-Ray Other Viral Pneumonia",
    "X-Ray SARS",
    "X-Ray Wildtype SARS-CoV-2"
]

def evaluate(model, loader, criterion, device, is_test=False):
    model.eval()
    if not is_test:
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(device), label.to(device)
                out = model(data)
                loss = criterion(out, label)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += torch.sum(pred == label).item()
                total += label.size(0)
        avg_loss = total_loss / len(loader)
        acc = 100.0 * correct / total
        return acc, avg_loss
    # 完整测试集指标与可视化
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data, label in tqdm(loader):
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.numpy())
    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)
    rec = recall_score(all_labels, preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    y_bin = label_binarize(all_labels, classes=list(range(len(target_names))))
    auc = roc_auc_score(y_bin, all_probs, average="weighted", multi_class="ovo")
    print(f"\nTest Metrics:\nAcc:{acc:.4f} Recall:{rec:.4f} Precision:{prec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    # 导出分类报告CSV，存入output
    report = classification_report(all_labels, preds, target_names=target_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv("./output/Swin_Report.csv")
    # 混淆矩阵
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(14,12))
    sns.heatmap(pd.DataFrame(cm, index=target_names, columns=target_names), annot=True, fmt="d", cmap="YlGnBu", annot_kws={"size":9})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("./output/Swin_ConfusionMatrix.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    # 提取特征
    def get_feats(model, loader):
        feats, labels = [], []
        with torch.no_grad():
            for data, lab in tqdm(loader):
                data = data.to(device)
                feat = model.forward_features(data)
                feats.append(feat.cpu())
                labels.append(lab)
        return torch.cat(feats).numpy(), torch.cat(labels).numpy()
    feats, feat_labels = get_feats(model, loader)
    # PCA
    pca_2d = PCA(n_components=2).fit_transform(feats)
    plt.figure(figsize=(12,10))
    sc = plt.scatter(pca_2d[:,0], pca_2d[:,1], c=feat_labels, cmap="tab10", alpha=0.7)
    plt.legend(*sc.legend_elements(), labels=target_names, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig("./output/Swin_PCA.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    # TSNE 修复：样本数不足自动降低perplexity消除警告
    sub_size = min(2000, len(feats))
    tsne_input = feats[:sub_size]
    tsne_labs = feat_labels[:sub_size]
    perplexity = min(30, sub_size - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=600, max_iter=1000, random_state=42)
    tsne_2d = tsne.fit_transform(tsne_input)
    plt.figure(figsize=(12,10))
    sc = plt.scatter(tsne_2d[:,0], tsne_2d[:,1], c=tsne_labs, cmap="tab10", alpha=0.7)
    plt.legend(*sc.legend_elements(), labels=target_names, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig("./output/Swin_TSNE.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 测试完成：分类报告、混淆矩阵、PCA、t-SNE文件已生成至 ./output 文件夹")

# 单独运行测试入口
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_dataloaders(batch_size=32)
    model = SwinTransformer(
        img_size=224, patch_size=4, in_chans=3, num_classes=11,
        embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
        window_size=7, mlp_ratio=4., drop_path_rate=0.1
    ).to(device)
    # 修复：设备兼容 + 捕获权重缺失异常
    weight_path = "best_model.pth"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件 {weight_path} 不存在，请先运行train.py训练生成权重！")
    # 强化map_location，多设备兼容
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    evaluate(model, test_loader, None, device, is_test=True)
