# ComfyUI-LongLive-Plug（日本語概要）

NVlabs の [LongLive-Plug](https://github.com/NVlabs/LongLive/tree/main/LongLive-Plug) アダプタを、
ComfyUI 標準の Wan2.1 実装で正しく使うための**非公式**コミュニティ統合です。
NVIDIA・Wan チーム・Comfy Org とは無関係です。詳細は英語版 [README.md](README.md) を参照してください。

## インストール

Comfy Registry に `longlive-plug`（表示名 "LongLive-Plug for ComfyUI"）として公開しています。
ComfyUI-Manager で `LongLive-Plug` を検索するか、`comfy node install longlive-plug` でインストールできます。
git clone でも導入できます。手順は README.md にあります。

## 何をするか

- **LongLive-Plug Apply Adapter Pair**: 公式の Few-Step LoRA（重み 1.0）と CFG LoRA（重み 0.5）を
  Wan2.1-T2V-14B に適用します。バックボーン形状、PEFT/ネイティブのキー変換、A/B の向き、
  rank と alpha、未使用キーを検査し、適用率（14B では各 400/400）を JSON で報告します。
  量子化ベースには適用を拒否します。
- **LongLive-Plug Sampling Recipe**: 上流で確認したサンプリングレシピ（GUIDER / SAMPLER / SIGMAS）を出力します。
  既定は LongLive-Plug の 4 ステップ FlowUniPC（CFG 1.0、条件付きパスのみ）です。
  上流スケジューラとの CPU 上でのビット一致をテストで確認しています。
  比較用に Wan2.1 公式の 50 ステップ（CFG 5）も入っています。

`steps=4` にするだけでは公式レシピになりません。LoRA の CFG 重み（0.5）と
サンプラーの CFG スケール（1.0）は別物です。

## 検証済みの範囲

- 実生成: Windows 11、RTX 5090 32GB、ComfyUI v0.38.0、Wan2.1-T2V-14B（bf16）、832×480・81 フレーム。
  ComfyUI は `--bf16-unet --bf16-text-enc --fp32-vae` で起動しています。
- CI（モデル不要）: Ubuntu / Windows。
- 実測値・画質比較・手順は README.md と docs/ にあります。数値はすべて今回の実測です。

## ライセンス

このリポジトリのコードは Apache-2.0 です。アダプタ（Apache-2.0）とベースモデル Wan2.1（Apache-2.0、利用上の注意あり）は
各自で取得し、それぞれの条件に従ってください。内訳は [docs/LICENSING.md](docs/LICENSING.md) にあります。
