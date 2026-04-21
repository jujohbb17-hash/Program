# ============================================================
# Step 2: Matplotlib基礎 - 画像と結果の可視化
# ============================================================
# インストール: pip install matplotlib
# 医療AI研究では結果をグラフや画像で可視化することが重要です

import numpy as np
import matplotlib.pyplot as plt

# ---- 1. 画像の表示 ----
# ランダムなグレースケール画像を作成 (本物のCT画像のような雰囲気)
np.random.seed(42)
fake_xray = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
# 中央に明るい「腫瘍っぽい」領域を追加
fake_xray[25:38, 25:38] = np.random.randint(200, 255, size=(13, 13))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(fake_xray, cmap='gray')
axes[0].set_title('グレースケール画像')
axes[0].axis('off')

axes[1].imshow(fake_xray, cmap='hot')
axes[1].set_title('熱マップ (hot)')
axes[1].colorbar = plt.colorbar(axes[1].images[0], ax=axes[1])
axes[1].axis('off')

axes[2].imshow(fake_xray, cmap='viridis')
axes[2].set_title('viridisカラーマップ')
axes[2].axis('off')

plt.suptitle('医療画像の可視化例', fontsize=14)
plt.tight_layout()
plt.savefig('output_02_images.png', dpi=100)
print("画像を 'output_02_images.png' に保存しました")
plt.show()

# ---- 2. 学習曲線のプロット ----
# 機械学習の訓練過程でloss(損失)とaccuracy(精度)を記録する
epochs = list(range(1, 21))
train_loss = [0.8 * np.exp(-0.15 * e) + 0.05 * np.random.randn() for e in epochs]
val_loss   = [0.9 * np.exp(-0.12 * e) + 0.08 + 0.05 * np.random.randn() for e in epochs]
train_acc  = [1 - 0.7 * np.exp(-0.18 * e) + 0.02 * np.random.randn() for e in epochs]
val_acc    = [1 - 0.75 * np.exp(-0.15 * e) + 0.02 * np.random.randn() for e in epochs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_loss, 'b-o', label='訓練損失', markersize=4)
ax1.plot(epochs, val_loss, 'r-o', label='検証損失', markersize=4)
ax1.set_xlabel('エポック')
ax1.set_ylabel('損失 (Loss)')
ax1.set_title('学習曲線: 損失')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_acc, 'b-o', label='訓練精度', markersize=4)
ax2.plot(epochs, val_acc, 'r-o', label='検証精度', markersize=4)
ax2.set_xlabel('エポック')
ax2.set_ylabel('精度 (Accuracy)')
ax2.set_title('学習曲線: 精度')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.suptitle('CNNの学習過程', fontsize=14)
plt.tight_layout()
plt.savefig('output_02_training.png', dpi=100)
print("学習曲線を 'output_02_training.png' に保存しました")
plt.show()

# ---- 3. 混同行列 (Confusion Matrix) ----
# 医療AIでは「見逃し(False Negative)」が特に重要
from matplotlib.colors import LinearSegmentedColormap

confusion = np.array([
    [85,  5],   # 実際: 陰性 → 予測: 陰性85, 陽性5
    [ 8, 92],   # 実際: 陽性 → 予測: 陰性8,  陽性92
])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(confusion, cmap='Blues')
plt.colorbar(im)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['予測: 陰性', '予測: 陽性'])
ax.set_yticklabels(['実際: 陰性', '実際: 陽性'])

for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion[i, j], ha='center', va='center',
                fontsize=18, color='black')

ax.set_title('混同行列 (Confusion Matrix)\n感度=92%, 特異度=94%')
plt.tight_layout()
plt.savefig('output_02_confusion.png', dpi=100)
print("混同行列を 'output_02_confusion.png' に保存しました")
plt.show()

# ---- 4. ヒストグラム (画像の輝度分布) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(fake_xray, cmap='gray')
axes[0].set_title('元画像')
axes[0].axis('off')

axes[1].hist(fake_xray.flatten(), bins=50, color='steelblue', edgecolor='white')
axes[1].set_xlabel('画素値 (0=黒, 255=白)')
axes[1].set_ylabel('頻度')
axes[1].set_title('輝度ヒストグラム')
axes[1].axvline(fake_xray.mean(), color='red', linestyle='--', label=f'平均={fake_xray.mean():.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('output_02_histogram.png', dpi=100)
print("ヒストグラムを 'output_02_histogram.png' に保存しました")
plt.show()

print("\n--- Step 2 完了! 次は 03_cnn_concept.py へ ---")
