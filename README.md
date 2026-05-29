# XAI Misclassification Analysis

## 概要

本リポジトリは、画像分類モデルの誤分類に対してXAI手法を適用し、
モデルがどの領域に注目して誤分類したのかを分析するための研究用リポジトリである。

## 目的

- Grad-CAMなどのXAI手法を用いて、分類モデルの判断根拠を可視化する
- 正分類画像と誤分類画像における注目領域の違いを比較する
- XAI結果が誤分類原因の分析に有効か検討する

## 使用予定の手法

- ResNet18
- Grad-CAM
- LIME
- SHAP
- 誤分類画像の抽出
- 可視化結果の比較

## ディレクトリ構成

```text
src/          実験用スクリプト
notebooks/    試行錯誤用ノートブック
experiments/  実験ごとの設定・結果
docs/         論文メモ・研究メモ
data/         データセット配置用
outputs/      出力画像・モデルなど
```

## 実行方法

```bash
git clone https://github.com/kkacha624-netizen/xai_misclassification_project
pip install -r requirements.txt
pip install -r requirements_torch.txt
python src/train.py
python src/explain.py
```

## 参考文献
- Selvaraju et al., Grad-CAM
- Ribeiro et al., LIME
- Lundberg and Lee, SHAP
