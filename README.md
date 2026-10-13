# Litera OCR

C++20 OCR library that extracts Ukrainian text from images such as
real-estate ads and screenshots of social media posts.

Pipeline: **DBNet text detection → reading-order sort → perspective crop →
CRNN recognition → CTC decode**, all ONNX models running on ONNX Runtime.
OpenCV is used only for image preprocessing and IO.

```
image ──► DbDetector (DBNet, language-independent)
              │  quad boxes
              ▼
        sortReadingOrder ──► cropTextRegion (perspective warp, per box)
              │  line crops
              ▼
        CrnnRecognizer (PP-OCRv3 cyrillic) ──► CtcDecoder ──► OcrResult
```

## Layout

```
include/litera/ocr/     public headers (OcrEngine is the entry point)
src/                implementation
assets/dicts/       cyrillic_dict.txt — official PaddleOCR charset (uk/ru/...)
models/             ONNX models (see below)
tools/get_models.sh downloads + converts production PP-OCRv3 models
tests/              GTest unit + integration tests
cli/                litera-ocr command-line tool
```

## Build (desktop)

```bash
# deps: OpenCV (core/imgproc/imgcodecs) + ONNX Runtime release tarball
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
ctest --test-dir build            
```

## Models

Bundled for tests (small, fetched from public GitHub repos):

| file                  | role                          | note                     |
|-----------------------|-------------------------------|--------------------------|
| `dbnet.onnx`          | text detection                | language-independent     |
| `crnn_lite_lstm.onnx` | recognition (Chinese charset) | integration tests only   |

Production models — run `tools/get_models.sh` on a machine with access to
`paddleocr.bj.bcebos.com` (needs `pip install paddlepaddle paddle2onnx`):

* `det_db.onnx` — PP-OCRv3 detection
* `rec_cyrillic.onnx` — PP-OCRv3 cyrillic recognition (Ukrainian, …)

`assets/dicts/cyrillic_dict.txt` has 517 characters, so the recognition model
must output 519 classes (blank + 517 chars + space). The engine validates this
and fails loudly on mismatch.

Then:

```bash
./build/litera-ocr models/det_db.onnx models/rec_cyrillic.onnx \
    assets/dicts/cyrillic_dict.txt ad_photo.jpg
```

## Usage

```cpp
#include "litera/ocr/ocr_engine.hpp"

litera::ocr::OcrEngine engine("models/det_db.onnx", "models/rec_cyrillic.onnx",
                          "assets/dicts/cyrillic_dict.txt");

cv::Mat image = cv::imread("ad_photo.jpg", cv::IMREAD_COLOR);
litera::ocr::OcrResult result = engine.recognize(image);
std::string text = result.fullText();
```

* `recognizeEncoded(data, size)` accepts raw JPEG/PNG bytes.
* `recognizeTiled(image)` splits tall images into overlapping strips,
  recognizes them in parallel and merges the results — use it for long
  screenshots where small text would otherwise be downscaled away.
* `OcrEngine` is not thread-safe for concurrent calls — serialize them or
  create one engine per worker thread.
* An optional `uk_lexicon.txt` placed next to the dictionary enables
  Ukrainian spell correction of the recognized text.

## Tuning knobs

* `DbDetector::Config` — `binThresh`/`boxThresh` (detection sensitivity),
  `unclipRatio` (box padding; raise to ~2.0 if characters get clipped),
  `maxSide` (speed/accuracy trade-off, 960 default).
* `CrnnRecognizer::Config` — defaults match PP-OCRv3 (3×48×W, [-1,1]);
  for 32-px CRNNs set `inputHeight = 32`.
* `OcrEngine::Config::minLineConfidence` — drop noisy lines before parsing.
