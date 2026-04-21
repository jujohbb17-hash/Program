# ============================================================
# Step 1: NumPy基礎 - 画像データの数値処理
# ============================================================
# インストール: pip install numpy
# NumPyは画像を「数値の配列」として扱うための必須ライブラリです

import numpy as np

# ---- 1. NumPy配列の基本 ----
# 画像は数字の行列として表現される
arr = np.array([1, 2, 3, 4, 5])
print("1次元配列:", arr)
print("型:", arr.dtype)

# ---- 2. 2次元配列 = グレースケール画像のイメージ ----
# 4x4ピクセルの小さな「画像」を作る
image_4x4 = np.array([
    [120, 130, 140, 150],
    [110, 200, 210, 160],
    [105, 195, 205, 155],
    [100, 115, 125, 145]
], dtype=np.uint8)  # uint8 = 0〜255の整数

print("\n4x4 グレースケール画像 (画素値):")
print(image_4x4)
print("形状 (shape):", image_4x4.shape)  # (行数, 列数)

# ---- 3. 3次元配列 = カラー画像 ----
# 形状: (高さ, 幅, チャンネル数)  ← チャンネル = RGB
color_image = np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8)
print("\nカラー画像の形状:", color_image.shape)  # (28, 28, 3)
print("  → 28x28ピクセル、3チャンネル(R, G, B)")

# ---- 4. 医療AIでよく使う前処理 ----

# (a) 正規化: 画素値を 0〜255 から 0.0〜1.0 に変換
normalized = image_4x4.astype(np.float32) / 255.0
print("\n正規化後:")
print(normalized.round(2))

# (b) 統計量の計算
print("\n統計:")
print(f"  平均値: {image_4x4.mean():.1f}")
print(f"  標準偏差: {image_4x4.std():.1f}")
print(f"  最小値: {image_4x4.min()}, 最大値: {image_4x4.max()}")

# (c) 標準化 (平均0、標準偏差1に変換) - CNNの学習を安定させる
mean = image_4x4.mean()
std = image_4x4.std()
standardized = (image_4x4.astype(np.float32) - mean) / std
print("\n標準化後の平均:", standardized.mean().round(4))
print("標準化後の標準偏差:", standardized.std().round(4))

# ---- 5. 配列のスライス ----
# 画像の一部（関心領域: ROI）を切り取る
roi = image_4x4[1:3, 1:3]  # 行1〜2、列1〜2
print("\nROI (中央2x2):")
print(roi)

# ---- 6. バッチ処理 (複数画像をまとめて処理) ----
# CNNでは複数画像を「バッチ」としてまとめて学習する
# 形状: (バッチサイズ, チャンネル, 高さ, 幅) ← PyTorchの形式
batch_size = 8
batch = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
print("\nバッチの形状:", batch.shape)
print("  → 8枚の28x28グレースケール画像")

# ---- 7. ゼロパディング (画像の周囲を0で埋める) ----
padded = np.pad(image_4x4, pad_width=1, mode='constant', constant_values=0)
print("\nパディング後 (6x6):")
print(padded)

print("\n--- Step 1 完了! 次は 02_matplotlib_basics.py へ ---")
