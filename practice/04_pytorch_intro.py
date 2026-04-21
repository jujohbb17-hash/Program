# ============================================================
# Step 4: PyTorch入門 - CNNの実装
# ============================================================
# インストール: pip install torch torchvision
# PyTorchは医療AI研究で最も使われているディープラーニングフレームワークです

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorchバージョン: {torch.__version__}")
print(f"GPU利用可能: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}\n")

# ---- 1. テンソル (Tensor) ----
# PyTorchではNumPy配列の代わりに「テンソル」を使う
# テンソルはGPUで計算でき、勾配（微分）も自動で計算される

t = torch.tensor([1.0, 2.0, 3.0])
print("テンソル:", t)
print("型:", t.dtype)
print("形状:", t.shape)

# NumPy配列との変換
arr = np.array([4.0, 5.0, 6.0])
t_from_np = torch.from_numpy(arr)
print("NumPy→テンソル:", t_from_np)

back_to_np = t.numpy()
print("テンソル→NumPy:", back_to_np)

# ---- 2. 画像テンソルの形状 ----
# PyTorchの形状: (バッチサイズ, チャンネル数, 高さ, 幅)
batch = torch.randn(8, 1, 28, 28)  # 8枚の28x28グレースケール画像
print(f"\n画像バッチの形状: {batch.shape}")
print("  → (8枚, 1チャンネル, 28px, 28px)")

# ---- 3. 簡単なCNNモデルの定義 ----
class SimpleCNN(nn.Module):
    """医療画像分類のためのシンプルなCNN"""

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # 特徴抽出部: 畳み込み + プーリングを繰り返す
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 → 14x14

            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 → 7x7
        )

        # 分類部: 全結合層で最終判断
        self.classifier = nn.Sequential(
            nn.Flatten(),                # (32, 7, 7) → 1568
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),             # 過学習を防ぐ
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=2).to(device)
print("\n--- モデル構造 ---")
print(model)

# パラメータ数を確認
total_params = sum(p.numel() for p in model.parameters())
print(f"\n総パラメータ数: {total_params:,}")

# ---- 4. 損失関数とオプティマイザ ----
criterion = nn.CrossEntropyLoss()  # 分類問題の標準的な損失関数
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n損失関数: CrossEntropyLoss (多クラス分類に使用)")
print("オプティマイザ: Adam (lr=0.001)")

# ---- 5. ダミーデータで学習ループを体験 ----
# 実際の研究では本物の医療画像データセットを使う
print("\n--- ダミーデータで学習ループ体験 ---")

n_samples = 200
X = torch.randn(n_samples, 1, 28, 28)  # ダミー画像
y = torch.randint(0, 2, (n_samples,))  # ダミーラベル (0=陰性, 1=陽性)

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

losses = []
model.train()

for epoch in range(10):
    epoch_loss = 0.0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()          # 勾配をリセット
        outputs = model(batch_X)       # 順伝播 (forward pass)
        loss = criterion(outputs, batch_y)  # 損失を計算
        loss.backward()                # 逆伝播 (backpropagation)
        optimizer.step()               # パラメータを更新

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    losses.append(avg_loss)
    print(f"  Epoch {epoch+1:2d}/10 | Loss: {avg_loss:.4f}")

# 学習曲線
plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), losses, 'b-o', markersize=5)
plt.xlabel('エポック')
plt.ylabel('損失')
plt.title('学習曲線 (ダミーデータ)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_04_loss.png', dpi=100)
print("\n学習曲線を 'output_04_loss.png' に保存しました")
plt.show()

# ---- 6. 推論 (予測) ----
model.eval()  # 評価モードに切り替え (DropoutをOFF)
with torch.no_grad():  # 勾配計算不要 → メモリ節約
    test_image = torch.randn(1, 1, 28, 28).to(device)
    output = model(test_image)
    probs = torch.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    class_names = ["陰性 (Benign)", "陽性 (Malignant)"]
    print(f"\n予測結果: {class_names[pred_class]}")
    print(f"  陰性の確率: {probs[0][0].item():.3f}")
    print(f"  陽性の確率: {probs[0][1].item():.3f}")

print("\n--- Step 4 完了! 次は 05_medical_ai_practice.py へ ---")
