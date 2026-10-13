#include "litera/ocr/preprocess.hpp"

#include <opencv2/imgproc.hpp>

// OpenCV 5 moved geometry functions out of imgproc.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <cmath>

namespace litera::ocr {

cv::Mat resizeForDetection(const cv::Mat& srcBgr, int maxSide, float& scaleX,
                           float& scaleY) {
    const int w = srcBgr.cols;
    const int h = srcBgr.rows;

    float ratio = 1.0F;
    if (std::max(w, h) > maxSide) {
        ratio = static_cast<float>(maxSide) / static_cast<float>(std::max(w, h));
    }

    auto roundUp32 = [](float v) {
        int r = static_cast<int>(std::round(v / 32.0F)) * 32;
        return std::max(r, 32);
    };
    const int newW = roundUp32(static_cast<float>(w) * ratio);
    const int newH = roundUp32(static_cast<float>(h) * ratio);

    cv::Mat resized;
    cv::resize(srcBgr, resized, cv::Size(newW, newH));

    scaleX = static_cast<float>(w) / static_cast<float>(newW);
    scaleY = static_cast<float>(h) / static_cast<float>(newH);
    return resized;
}

std::vector<float> toNchwBlob(const cv::Mat& bgr, const cv::Scalar& meanRgb,
                              const cv::Scalar& stdRgb) {
    CV_Assert(bgr.type() == CV_8UC3 || bgr.type() == CV_8UC1);

    const int h = bgr.rows;
    const int w = bgr.cols;
    const int channels = bgr.channels();
    std::vector<float> blob(static_cast<size_t>(channels) * h * w);

    for (int c = 0; c < channels; ++c) {
        const int srcC = channels == 3 ? 2 - c : 0;  // BGR -> RGB
        const float mean = static_cast<float>(meanRgb[c]);
        const float stdv = static_cast<float>(stdRgb[c]);
        float* plane = blob.data() + static_cast<size_t>(c) * h * w;
        for (int y = 0; y < h; ++y) {
            const uchar* row = bgr.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                const float v = static_cast<float>(row[x * channels + srcC]) / 255.0F;
                plane[static_cast<size_t>(y) * w + x] = (v - mean) / stdv;
            }
        }
    }
    return blob;
}

cv::Mat enhanceForOcr(const cv::Mat& srcBgr, double upscale) {
    CV_Assert(!srcBgr.empty());

    cv::Mat work = srcBgr;
    if (upscale > 1.0) {
        cv::resize(work, work, cv::Size(), upscale, upscale, cv::INTER_CUBIC);
    }

    cv::Mat gray;
    cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);

    cv::Mat out;
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    return out;
}

cv::Mat cropTextRegion(const cv::Mat& srcBgr, const TextBox& box) {
    const int cropW = std::max(1, static_cast<int>(std::round(box.width())));
    const int cropH = std::max(1, static_cast<int>(std::round(box.height())));

    const cv::Point2f src[4] = {box.points[0], box.points[1], box.points[2],
                                box.points[3]};
    const cv::Point2f dst[4] = {
        {0.0F, 0.0F},
        {static_cast<float>(cropW), 0.0F},
        {static_cast<float>(cropW), static_cast<float>(cropH)},
        {0.0F, static_cast<float>(cropH)}};

    cv::Mat m = cv::getPerspectiveTransform(src, dst);
    cv::Mat crop;
    cv::warpPerspective(srcBgr, crop, m, cv::Size(cropW, cropH),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    if (static_cast<float>(crop.rows) / static_cast<float>(crop.cols) > 1.5F) {
        cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return crop;
}

void sortReadingOrder(std::vector<TextBox>& boxes) {
    if (boxes.size() < 2) return;

    std::sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b) {
        return a.center().y < b.center().y;
    });

    std::vector<std::vector<TextBox>> lines;
    for (auto& box : boxes) {
        if (!lines.empty()) {
            auto& line = lines.back();
            float avgY = 0.0F;
            float avgH = 0.0F;
            for (const auto& p : line) {
                avgY += p.center().y;
                avgH += p.height();
            }
            avgY /= static_cast<float>(line.size());
            avgH /= static_cast<float>(line.size());
            if (std::abs(box.center().y - avgY) < avgH * 0.5F) {
                line.push_back(box);
                continue;
            }
        }
        lines.push_back({box});
    }

    boxes.clear();
    for (auto& line : lines) {
        std::sort(line.begin(), line.end(), [](const TextBox& a, const TextBox& b) {
            return a.center().x < b.center().x;
        });
        boxes.insert(boxes.end(), line.begin(), line.end());
    }
}

}  // namespace litera::ocr
