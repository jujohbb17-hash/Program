# CNN・医療AI 学習プログラム (Python初心者向け)

## 学習の進め方

ファイルを **番号順** に実行してください。各ファイルは前のステップを前提としています。

| ファイル | 内容 | 必要なライブラリ |
|---------|------|----------------|
| `00_python_basics.py` | Python基礎文法 (変数、リスト、ループ、関数) | なし |
| `01_numpy_basics.py` | NumPy - 画像の数値処理 | numpy |
| `02_matplotlib_basics.py` | Matplotlib - 画像・グラフの可視化 | matplotlib |
| `03_cnn_concept.py` | CNNの仕組み (畳み込み・プーリング・活性化関数) | numpy, matplotlib |
| `04_pytorch_intro.py` | PyTorch入門 - CNNの実装と学習ループ | torch |
| `05_medical_ai_practice.py` | 医療AI実践 - 乳がん診断データセット | torch, scikit-learn |

## セットアップ

```bash
pip install numpy matplotlib torch torchvision scikit-learn
```

## VS Codeでの実行方法

- ファイル全体を実行: `Ctrl+F5` (Mac: `Fn+F5`)
- 選択行を実行: `Shift+Enter` (Jupyter的に使える)
- ターミナルで実行: `python 00_python_basics.py`

## 学習後の次のステップ

1. **転移学習**: ResNet18/VGG16など学習済みモデルを医療画像に適用
2. **データ拡張**: `torchvision.transforms`で学習データを増やす
3. **公開データセット**: NIH Chest X-ray、ISIC皮膚病変データセット
4. **Grad-CAM**: CNNがどこを見ているか可視化する手法
