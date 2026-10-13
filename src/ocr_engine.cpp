#include "litera/ocr/ocr_engine.hpp"

#include "litera/ocr/preprocess.hpp"
#include "litera/ocr/spell_corrector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litera::ocr {

struct OcrEngine::Impl {
    DbDetector detector;
    CrnnRecognizer recognizer;
    Config config;
    SpellCorrector corrector;  // no-op if uk_lexicon.txt is missing

    Impl(const std::filesystem::path& detModelPath,
         const std::filesystem::path& recModelPath,
         const std::filesystem::path& dictPath, Config cfg)
        : detector(detModelPath, cfg.detector),
          recognizer(recModelPath, dictPath, cfg.recognizer),
          config(cfg),
          corrector(dictPath.parent_path() / "uk_lexicon.txt") {}
};

OcrEngine::OcrEngine(const std::filesystem::path& detModelPath,
                     const std::filesystem::path& recModelPath,
                     const std::filesystem::path& dictPath)
    : OcrEngine(detModelPath, recModelPath, dictPath, Config{}) {}

OcrEngine::OcrEngine(const std::filesystem::path& detModelPath,
                     const std::filesystem::path& recModelPath,
                     const std::filesystem::path& dictPath, Config config)
    : impl_(std::make_unique<Impl>(detModelPath, recModelPath, dictPath,
                                   config)) {}

OcrEngine::~OcrEngine() = default;
OcrEngine::OcrEngine(OcrEngine&&) noexcept = default;
OcrEngine& OcrEngine::operator=(OcrEngine&&) noexcept = default;

OcrResult OcrEngine::recognize(const cv::Mat& imageBgr) const {
    std::vector<TextBox> boxes = impl_->detector.detect(imageBgr);
    sortReadingOrder(boxes);

    OcrResult result;
    result.lines.reserve(boxes.size());
    for (const auto& box : boxes) {
        cv::Mat crop = cropTextRegion(imageBgr, box);
        CtcDecoder::Decoded decoded = impl_->recognizer.recognize(crop);
        if (decoded.text.empty() ||
            decoded.confidence < impl_->config.minLineConfidence) {
            continue;
        }
        result.lines.push_back(
            {box, std::move(decoded.text), decoded.confidence});
    }
    for (auto& line : result.lines) line.text = impl_->corrector.correct(line.text);
    return result;
}

namespace {

std::vector<int> tileStarts(int total, int tile, int stride) {
    std::vector<int> starts;
    if (total <= tile) {
        starts.push_back(0);
        return starts;
    }
    for (int s = 0;; s += stride) {
        if (s + tile >= total) {
            starts.push_back(total - tile);
            break;
        }
        starts.push_back(s);
    }
    return starts;
}

std::vector<cv::Rect> computeTiles(cv::Size size, int tileHeight,
                                     float overlap) {
    tileHeight = std::max(tileHeight, 64);
    overlap = std::clamp(overlap, 0.0F, 0.9F);
    const int stride = std::max(
        1, static_cast<int>(std::lround(tileHeight * (1.0F - overlap))));

    const std::vector<int> ys = tileStarts(size.height, tileHeight, stride);

    std::vector<cv::Rect> tiles;
    tiles.reserve(ys.size());
    for (const int y : ys) {
        const int h = std::min(tileHeight, size.height - y);
        tiles.emplace_back(0, y, size.width, h);
    }
    return tiles;
}

cv::Rect2f boundingBox(const TextBox& box) {
    float minX = box.points[0].x;
    float maxX = box.points[0].x;
    float minY = box.points[0].y;
    float maxY = box.points[0].y;
    for (std::size_t i = 1; i < box.points.size(); ++i) {
        minX = std::min(minX, box.points[i].x);
        maxX = std::max(maxX, box.points[i].x);
        minY = std::min(minY, box.points[i].y);
        maxY = std::max(maxY, box.points[i].y);
    }
    return {minX, minY, maxX - minX, maxY - minY};
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float ix = std::max(0.0F, std::min(a.x + a.width, b.x + b.width) -
                                        std::max(a.x, b.x));
    const float iy = std::max(0.0F, std::min(a.y + a.height, b.y + b.height) -
                                        std::max(a.y, b.y));
    const float inter = ix * iy;
    const float uni = a.area() + b.area() - inter;
    return uni > 0.0F ? inter / uni : 0.0F;
}

void nmsMerge(std::vector<OcrLine>& lines, float iouThresh) {
    if (lines.size() < 2) return;

    std::vector<cv::Rect2f> rects;
    rects.reserve(lines.size());
    for (const auto& line : lines) rects.push_back(boundingBox(line.box));

    std::vector<int> order(lines.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return lines[a].confidence > lines[b].confidence;
    });

    std::vector<char> suppressed(lines.size(), 0);
    std::vector<OcrLine> kept;
    kept.reserve(lines.size());
    for (const int i : order) {
        if (suppressed[i]) continue;
        for (const int j : order) {
            if (j == i || suppressed[j]) continue;
            if (iou(rects[i], rects[j]) > iouThresh) suppressed[j] = 1;
        }
        kept.push_back(std::move(lines[i]));
    }
    lines = std::move(kept);
}

void sortLinesReadingOrder(std::vector<OcrLine>& lines) {
    if (lines.size() < 2) return;

    std::sort(lines.begin(), lines.end(), [](const OcrLine& a, const OcrLine& b) {
        return a.box.center().y < b.box.center().y;
    });

    std::vector<std::vector<OcrLine>> rows;
    for (auto& line : lines) {
        if (!rows.empty()) {
            auto& row = rows.back();
            float avgY = 0.0F;
            float avgH = 0.0F;
            for (const auto& l : row) {
                avgY += l.box.center().y;
                avgH += l.box.height();
            }
            avgY /= static_cast<float>(row.size());
            avgH /= static_cast<float>(row.size());
            if (std::abs(line.box.center().y - avgY) < avgH * 0.5F) {
                row.push_back(std::move(line));
                continue;
            }
        }
        rows.push_back({std::move(line)});
    }

    lines.clear();
    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [](const OcrLine& a, const OcrLine& b) {
            return a.box.center().x < b.box.center().x;
        });
        for (auto& l : row) lines.push_back(std::move(l));
    }
}

cv::Mat decodeImage(const std::uint8_t* data, std::size_t size) {
    cv::Mat raw(1, static_cast<int>(size), CV_8UC1,
                const_cast<std::uint8_t*>(data));
    cv::Mat image = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("OcrEngine: failed to decode image buffer");
    }
    return image;
}

}  // namespace

OcrResult OcrEngine::recognizeTiled(const cv::Mat& imageBgr) const {
    return recognizeTiled(imageBgr, TileConfig{});
}

OcrResult OcrEngine::recognizeTiled(const cv::Mat& imageBgr,
                                    const TileConfig& tiling) const {
    CV_Assert(!imageBgr.empty() && imageBgr.type() == CV_8UC3);

    const cv::Mat work =
        tiling.enhance ? enhanceForOcr(imageBgr, tiling.upscale) : imageBgr;

    if (std::max(imageBgr.cols, imageBgr.rows) < tiling.minImageSide) {
        return recognize(work);
    }

    const std::vector<cv::Rect> tiles =
        computeTiles(work.size(), tiling.tileSize, tiling.overlap);

    if (std::getenv("LITERA_OCR_VERBOSE")) {
        std::fprintf(stderr,
                     "[litera-ocr] work=%dx%d tiles=%zu tileSize=%d overlap=%.3f\n",
                     work.cols, work.rows, tiles.size(), tiling.tileSize,
                     static_cast<double>(tiling.overlap));
    }

    std::vector<std::vector<OcrLine>> perTile(tiles.size());
    cv::parallel_for_(cv::Range(0, static_cast<int>(tiles.size())),
                      [&](const cv::Range& range) {
        for (int t = range.start; t < range.end; ++t) {
            const cv::Rect& roi = tiles[t];
            const cv::Mat view = work(roi);

            std::vector<TextBox> boxes = impl_->detector.detect(view);
            std::vector<OcrLine>& out = perTile[t];
            out.reserve(boxes.size());
            for (auto& box : boxes) {
                for (auto& p : box.points) {
                    p.x += static_cast<float>(roi.x);
                    p.y += static_cast<float>(roi.y);
                }
                // Crop from the full image so boxes on tile edges stay complete.
                cv::Mat crop = cropTextRegion(imageBgr, box);
                CtcDecoder::Decoded decoded = impl_->recognizer.recognize(crop);
                if (decoded.text.empty() ||
                    decoded.confidence < impl_->config.minLineConfidence) {
                    continue;
                }
                out.push_back({box, std::move(decoded.text), decoded.confidence});
            }
        }
    });

    OcrResult result;
    std::size_t total = 0;
    for (const auto& lines : perTile) total += lines.size();
    result.lines.reserve(total);
    for (auto& lines : perTile) {
        for (auto& line : lines) result.lines.push_back(std::move(line));
    }

    nmsMerge(result.lines, tiling.nmsIoU);
    sortLinesReadingOrder(result.lines);
    for (auto& line : result.lines) line.text = impl_->corrector.correct(line.text);
    return result;
}

OcrResult OcrEngine::recognizeEncoded(const std::uint8_t* data,
                                      std::size_t size) const {
    cv::Mat imageBgr = decodeImage(data, size);

    cv::Mat processed;
    cv::resize(imageBgr, processed, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(gray, gray);

    cv::Mat finalImage;
    cv::cvtColor(gray, finalImage, cv::COLOR_GRAY2BGR);

    return recognize(finalImage);
}

OcrResult OcrEngine::recognizeEncodedTiled(const std::uint8_t* data,
                                           std::size_t size) const {
    cv::Mat imageBgr = decodeImage(data, size);

    cv::Mat processed;
    cv::resize(imageBgr, processed, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(gray, gray);

    cv::Mat finalImage;
    cv::cvtColor(gray, finalImage, cv::COLOR_GRAY2BGR);

    return recognizeTiled(finalImage, TileConfig{});
}

OcrResult OcrEngine::recognizeEncodedTiled(const std::uint8_t* data,
                                           std::size_t size,
                                           const TileConfig& tiling) const {
    cv::Mat imageBgr = decodeImage(data, size);

    cv::Mat processed;
    cv::resize(imageBgr, processed, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(gray, gray);

    cv::Mat finalImage;
    cv::cvtColor(gray, finalImage, cv::COLOR_GRAY2BGR);

    return recognizeTiled(finalImage, tiling);
}

}  // namespace litera::ocr
