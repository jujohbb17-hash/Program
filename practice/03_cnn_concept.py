# ============================================================
# Step 3: CNNの仕組みを理解する (NumPyで実装)
# ============================================================
# PyTorchを使う前に、CNNが「何をしているか」を理解しましょう
# ここではNumPyだけでCNNの核心操作を実装します

import numpy as np
import matplotlib.pyplot as plt

# ---- 1. 畳み込み (Convolution) とは？ ----
# フィルタ（カーネル）を画像上でスライドさせて特徴を検出する

def convolve2d(image, kernel):
    """2次元畳み込みの手動実装"""
    ih, iw = image.shape
    kh, kw = kernel.shape
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

# 簡単な画像を作成
np.random.seed(0)
image = np.zeros((16, 16))
image[4:12, 4:12] = 1.0   # 中央に白い四角形

# 様々なフィルタ
filters = {
    "エッジ検出 (縦)": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    "エッジ検出 (横)": np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]]),
    "ぼかし (平均)":   np.ones((3, 3)) / 9.0,
    "シャープ化":      np.array([[ 0,-1, 0], [-1, 5,-1], [ 0,-1, 0]]),
}

fig, axes = plt.subplots(1, 5, figsize=(16, 3))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('元画像')
axes[0].axis('off')

for ax, (name, kernel) in zip(axes[1:], filters.items()):
    result = convolve2d(image, kernel)
    ax.imshow(result, cmap='gray')
    ax.set_title(name, fontsize=9)
    ax.axis('off')

plt.suptitle('畳み込みフィルタの効果', fontsize=13)
plt.tight_layout()
plt.savefig('output_03_conv.png', dpi=100)
print("畳み込み結果を 'output_03_conv.png' に保存しました")
plt.show()

# ---- 2. プーリング (Pooling) とは？ ----
# 特徴マップを小さくまとめて計算量を削減し、位置不変性を高める

def max_pooling2d(feature_map, pool_size=2):
    """Max Poolingの手動実装"""
    h, w = feature_map.shape
    out_h = h // pool_size
    out_w = w // pool_size
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = feature_map[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size
            ].max()
    return output

sample_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 1, 4, 7],
    [2, 8, 3, 5],
], dtype=float)

pooled = max_pooling2d(sample_map, pool_size=2)
print("Max Pooling の例:")
print("入力 (4x4):\n", sample_map)
print("出力 (2x2):\n", pooled)
print("  → 各2x2領域の最大値を取る")

# ---- 3. 活性化関数 (Activation Function) ----
# 非線形性を導入して複雑なパターンを学習できるようにする

x = np.linspace(-4, 4, 200)

relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh_v = np.tanh(x)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(x, relu, 'b-', lw=2)
axes[0].axhline(0, color='k', lw=0.5)
axes[0].axvline(0, color='k', lw=0.5)
axes[0].set_title('ReLU\n(CNNで最もよく使われる)')
axes[0].set_ylim(-1, 4)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, sigmoid, 'r-', lw=2)
axes[1].axhline(0.5, color='gray', lw=0.5, linestyle='--')
axes[1].set_title('Sigmoid\n(二値分類の出力層)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(x, tanh_v, 'g-', lw=2)
axes[2].set_title('Tanh\n(-1〜1の範囲)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('活性化関数の比較', fontsize=13)
plt.tight_layout()
plt.savefig('output_03_activations.png', dpi=100)
print("活性化関数を 'output_03_activations.png' に保存しました")
plt.show()

# ---- 4. CNNの全体構造の説明 ----
print("""
CNNの処理フロー:

入力画像 (例: 224x224x3)
    ↓
[畳み込み層] フィルタで特徴を検出 → (222x222x32)
    ↓
[ReLU] 負の値を0にして非線形性を追加
    ↓
[MaxPooling] 特徴マップを半分に縮小 → (111x111x32)
    ↓
[畳み込み層] より複雑な特徴を検出
    ↓
[MaxPooling]
    ↓
[Flatten] 2D→1Dに変換 (全結合層への入力)
    ↓
[全結合層 (FC Layer)] 特徴を組み合わせて判断
    ↓
[Softmax/Sigmoid] 確率に変換
    ↓
出力 (例: 悪性の確率 = 0.87)
""")

print("--- Step 3 完了! 次は 04_pytorch_intro.py へ ---")
