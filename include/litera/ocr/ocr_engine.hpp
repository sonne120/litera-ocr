#pragma once

#include "litera/ocr/crnn_recognizer.hpp"
#include "litera/ocr/db_detector.hpp"
#include "litera/ocr/types.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace litera::ocr {

// Not thread-safe: serialize calls or use one engine per thread.
class OcrEngine {
public:
    struct Config {
        DbDetector::Config detector;
        CrnnRecognizer::Config recognizer;
        float minLineConfidence = 0.50F;
    };

    struct TileConfig {
        int tileSize = 768;
        float overlap = 0.125F;
        float nmsIoU = 0.30F;
        int minImageSide = 900;
        bool enhance = true;
        double upscale = 1.0;  // >1 also requires a larger detector maxSide
    };

    OcrEngine(const std::filesystem::path& detModelPath,
              const std::filesystem::path& recModelPath,
              const std::filesystem::path& dictPath);
    OcrEngine(const std::filesystem::path& detModelPath,
              const std::filesystem::path& recModelPath,
              const std::filesystem::path& dictPath, Config config);
    ~OcrEngine();
    OcrEngine(OcrEngine&&) noexcept;
    OcrEngine& operator=(OcrEngine&&) noexcept;

    [[nodiscard]] OcrResult recognize(const cv::Mat& imageBgr) const;

    [[nodiscard]] OcrResult recognizeEncoded(const std::uint8_t* data,
                                             std::size_t size) const;

    [[nodiscard]] OcrResult recognizeTiled(const cv::Mat& imageBgr) const;
    [[nodiscard]] OcrResult recognizeTiled(const cv::Mat& imageBgr,
                                           const TileConfig& tiling) const;

    [[nodiscard]] OcrResult recognizeEncodedTiled(const std::uint8_t* data,
                                                  std::size_t size) const;
    [[nodiscard]] OcrResult recognizeEncodedTiled(const std::uint8_t* data,
                                                  std::size_t size,
                                                  const TileConfig& tiling) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace litera::ocr
