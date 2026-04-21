# ============================================================
# Step 5: 医療AI実践 - 乳がん診断データセットで分類
# ============================================================
# scikit-learnに内蔵されている乳がんデータセットを使います
# (MNISTと並んで機械学習の入門によく使われる実際のデータです)
# インストール: pip install scikit-learn torch torchvision matplotlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# ---- 1. データの読み込みと確認 ----
print("=" * 50)
print("乳がん診断データセット")
print("=" * 50)

data = load_breast_cancer()
X = data.data.astype(np.float32)   # 特徴量 (30種類の計測値)
y = data.target.astype(np.int64)   # ラベル (0=悪性, 1=良性)

print(f"サンプル数: {X.shape[0]}")
print(f"特徴量数: {X.shape[1]}")
print(f"クラス: {data.target_names}")
print(f"良性: {(y==1).sum()}件, 悪性: {(y==0).sum()}件")
print(f"\n特徴量の例: {data.feature_names[:5]}")

# ---- 2. データの分割と前処理 ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n訓練データ: {X_train.shape[0]}件")
print(f"テストデータ: {X_test.shape[0]}件")

# PyTorchのテンソルに変換
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# ---- 3. モデルの定義 (表形式データ用の全結合NN) ----
class MedicalClassifier(nn.Module):
    """乳がん診断のための多層パーセプトロン"""

    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedicalClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- 4. 学習 ----
print("\n--- 学習開始 ---")
num_epochs = 50
train_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 検証精度を計算
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device))
        preds = outputs.argmax(dim=1).cpu()
        acc = (preds == y_test_t).float().mean().item()

    train_losses.append(epoch_loss / len(train_loader))
    val_accuracies.append(acc * 100)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {train_losses[-1]:.4f} | "
              f"Test Acc: {acc*100:.1f}%")

# ---- 5. 結果の評価 ----
model.eval()
with torch.no_grad():
    outputs = model(X_test_t.to(device))
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    preds = outputs.argmax(dim=1).cpu().numpy()

print("\n--- 分類レポート ---")
print(classification_report(y_test, preds,
                             target_names=data.target_names))

# ---- 6. 可視化 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 学習曲線
axes[0].plot(train_losses, 'b-', label='訓練損失', lw=1.5)
axes[0].set_xlabel('エポック'); axes[0].set_ylabel('損失')
axes[0].set_title('学習損失曲線'); axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accuracies, 'r-', label='テスト精度', lw=1.5)
axes[1].set_xlabel('エポック'); axes[1].set_ylabel('精度 (%)')
axes[1].set_title('テスト精度曲線')
axes[1].axhline(95, color='green', linestyle='--', alpha=0.5, label='95%ライン')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 混同行列
cm = confusion_matrix(y_test, preds)
im = axes[2].imshow(cm, cmap='Blues')
plt.colorbar(im, ax=axes[2])
axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(['悪性(予測)', '良性(予測)'])
axes[2].set_yticklabels(['悪性(実際)', '良性(実際)'])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha='center', va='center',
                     fontsize=18, color='black')
axes[2].set_title('混同行列')

plt.suptitle('乳がん診断AIの結果', fontsize=14)
plt.tight_layout()
plt.savefig('output_05_results.png', dpi=100)
print("結果を 'output_05_results.png' に保存しました")
plt.show()

# ---- 7. 重要な医療AI指標の解説 ----
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) * 100  # 感度: 陽性を正しく検出できた割合
specificity = tn / (tn + fp) * 100  # 特異度: 陰性を正しく除外できた割合

print(f"\n重要指標:")
print(f"  感度 (Sensitivity/Recall): {sensitivity:.1f}%  ← 見逃し率を最小化")
print(f"  特異度 (Specificity):      {specificity:.1f}%  ← 偽陽性率を最小化")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"\n  !! 医療では FN(見逃し)を特に減らすことが重要 !!")

print("\n--- Step 5 完了! おめでとうございます！---")
print("次のステップ: ResNetなどの転移学習 (Transfer Learning) を調べてみよう")
