# ============================================================
# Step 0: Python基礎 - AI研究に必要な基本文法
# ============================================================
# このファイルを上から順に読んで、実行してみましょう。
# VS Codeでは行を選択してShift+Enterで1行ずつ実行できます。

# ---- 1. 変数と基本的なデータ型 ----
patient_age = 45          # 整数 (int)
tumor_size = 2.3          # 小数 (float)
patient_name = "Tanaka"   # 文字列 (str)
is_malignant = True       # 真偽値 (bool)

print("患者名:", patient_name)
print("年齢:", patient_age)
print("腫瘍サイズ (cm):", tumor_size)
print("悪性フラグ:", is_malignant)

# ---- 2. リスト (複数のデータをまとめる) ----
# 画像の画素値のようなデータをリストで扱う
pixel_values = [128, 200, 55, 180, 90]
print("\n画素値リスト:", pixel_values)
print("最初の画素値:", pixel_values[0])   # インデックスは0始まり
print("最後の画素値:", pixel_values[-1])  # -1で最後の要素
print("リストの長さ:", len(pixel_values))

# ---- 3. 辞書 (キーと値のペア) ----
# 患者データのような構造化データに使う
patient_data = {
    "name": "Tanaka",
    "age": 45,
    "diagnosis": "benign",
    "image_path": "data/scan_001.png"
}
print("\n患者データ:", patient_data)
print("診断結果:", patient_data["diagnosis"])

# ---- 4. 条件分岐 ----
accuracy = 0.92

if accuracy >= 0.95:
    print("\nモデル精度: 優秀")
elif accuracy >= 0.85:
    print("\nモデル精度: 良好 →", accuracy)
else:
    print("\nモデル精度: 要改善")

# ---- 5. ループ ----
# データセットの各サンプルを処理するイメージ
labels = ["benign", "malignant", "benign", "malignant"]

print("\nラベル確認:")
for i, label in enumerate(labels):
    print(f"  サンプル{i}: {label}")

# ---- 6. 関数 ----
def calculate_accuracy(correct, total):
    """正解率を計算する関数"""
    return correct / total * 100

correct_predictions = 92
total_samples = 100
acc = calculate_accuracy(correct_predictions, total_samples)
print(f"\n正解率: {acc:.1f}%")

# ---- 7. リスト内包表記 (Pythonらしい書き方) ----
# 全画素値を0〜1の範囲に正規化 (AIでよく使う)
raw_pixels = [128, 200, 55, 180, 90]
normalized = [p / 255.0 for p in raw_pixels]
print("\n正規化前:", raw_pixels)
print("正規化後:", [f"{v:.3f}" for v in normalized])

print("\n--- Step 0 完了! 次は 01_numpy_basics.py へ ---")
