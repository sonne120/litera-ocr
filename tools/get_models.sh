#!/usr/bin/env bash
# Requires: pip install paddlepaddle paddle2onnx
set -euo pipefail

OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

BASE="https://paddleocr.bj.bcebos.com/PP-OCRv3"

echo "== detection model (multilingual) =="
curl -L "$BASE/chinese/ch_PP-OCRv3_det_infer.tar" -o "$TMP/det.tar"
tar xf "$TMP/det.tar" -C "$TMP"
paddle2onnx --model_dir "$TMP/ch_PP-OCRv3_det_infer" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --opset_version 14 \
    --save_file "$OUT_DIR/det_db.onnx"

echo "== recognition model (cyrillic: uk, ru, be, ...) =="
curl -L "$BASE/multilingual/cyrillic_PP-OCRv3_rec_infer.tar" -o "$TMP/rec.tar"
tar xf "$TMP/rec.tar" -C "$TMP"
paddle2onnx --model_dir "$TMP/cyrillic_PP-OCRv3_rec_infer" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --opset_version 14 \
    --save_file "$OUT_DIR/rec_cyrillic.onnx"

echo "Done:"
ls -lh "$OUT_DIR"/det_db.onnx "$OUT_DIR"/rec_cyrillic.onnx
echo
echo "Run: ./build/litera-ocr models/det_db.onnx models/rec_cyrillic.onnx \\"
echo "         assets/dicts/cyrillic_dict.txt photo.jpg"
