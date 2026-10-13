#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace litera::ocr {

// Points: top-left, top-right, bottom-right, bottom-left.
struct TextBox {
    std::array<cv::Point2f, 4> points{};
    float score = 0.0F;

    [[nodiscard]] cv::Point2f center() const {
        return (points[0] + points[1] + points[2] + points[3]) * 0.25F;
    }
    [[nodiscard]] float height() const {
        return static_cast<float>(
            (cv::norm(points[3] - points[0]) + cv::norm(points[2] - points[1])) / 2.0);
    }
    [[nodiscard]] float width() const {
        return static_cast<float>(
            (cv::norm(points[1] - points[0]) + cv::norm(points[2] - points[3])) / 2.0);
    }
};

struct OcrLine {
    TextBox box;
    std::string text;
    float confidence = 0.0F;
};

struct OcrResult {
    std::vector<OcrLine> lines;

    [[nodiscard]] std::string fullText() const {
        std::string out;
        for (const auto& l : lines) {
            if (!out.empty()) out += '\n';
            out += l.text;
        }
        return out;
    }
};

}  // namespace litera::ocr
