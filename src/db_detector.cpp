#include "litera/ocr/db_detector.hpp"

#include "litera/ocr/preprocess.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

// OpenCV 5 moved geometry functions out of imgproc.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>

namespace litera::ocr {

struct DbDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "litera-ocr-det"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::string outputName;
    Config config;

    Impl(const std::filesystem::path& modelPath, Config cfg) : config(cfg) {
        options.SetIntraOpNumThreads(2);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);

        Ort::AllocatorWithDefaultOptions alloc;
        inputName = session->GetInputNameAllocated(0, alloc).get();
        outputName = session->GetOutputNameAllocated(0, alloc).get();
    }
};

DbDetector::DbDetector(const std::filesystem::path& modelPath)
    : DbDetector(modelPath, Config{}) {}

DbDetector::DbDetector(const std::filesystem::path& modelPath, Config config)
    : impl_(std::make_unique<Impl>(modelPath, config)) {}

DbDetector::~DbDetector() = default;
DbDetector::DbDetector(DbDetector&&) noexcept = default;
DbDetector& DbDetector::operator=(DbDetector&&) noexcept = default;

namespace {

float boxScore(const cv::Mat& probMap, const cv::RotatedRect& rect) {
    cv::Rect bounds = rect.boundingRect() &
                      cv::Rect(0, 0, probMap.cols, probMap.rows);
    if (bounds.empty()) return 0.0F;

    cv::Mat mask = cv::Mat::zeros(bounds.size(), CV_8UC1);
    std::array<cv::Point2f, 4> pts;
    rect.points(pts.data());
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (const auto& p : pts) {
        poly.emplace_back(static_cast<int>(p.x) - bounds.x,
                          static_cast<int>(p.y) - bounds.y);
    }
    cv::fillConvexPoly(mask, poly, cv::Scalar(1));
    return static_cast<float>(cv::mean(probMap(bounds), mask)[0]);
}

// DB predicts shrunken text masks; expand them back.
cv::RotatedRect unclip(const cv::RotatedRect& rect, float ratio) {
    const float area = rect.size.width * rect.size.height;
    const float perimeter = 2.0F * (rect.size.width + rect.size.height);
    if (perimeter <= 0.0F) return rect;
    const float d = area * ratio / perimeter;
    cv::RotatedRect out = rect;
    out.size.width += 2.0F * d;
    out.size.height += 2.0F * d;
    return out;
}

std::array<cv::Point2f, 4> orderedCorners(const cv::RotatedRect& rect) {
    std::array<cv::Point2f, 4> pts;
    rect.points(pts.data());
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    const bool leftSwap = pts[0].y > pts[1].y;
    cv::Point2f tl = leftSwap ? pts[1] : pts[0];
    cv::Point2f bl = leftSwap ? pts[0] : pts[1];
    const bool rightSwap = pts[2].y > pts[3].y;
    cv::Point2f tr = rightSwap ? pts[3] : pts[2];
    cv::Point2f br = rightSwap ? pts[2] : pts[3];
    return {tl, tr, br, bl};
}

}  // namespace

std::vector<TextBox> DbDetector::detect(const cv::Mat& imageBgr) const {
    CV_Assert(!imageBgr.empty() && imageBgr.type() == CV_8UC3);
    const auto& cfg = impl_->config;

    float scaleX = 1.0F;
    float scaleY = 1.0F;
    cv::Mat resized = resizeForDetection(imageBgr, cfg.maxSide, scaleX, scaleY);

    std::vector<float> blob = toNchwBlob(resized, {0.485, 0.456, 0.406},
                                         {0.229, 0.224, 0.225});
    const std::array<int64_t, 4> shape{1, 3, resized.rows, resized.cols};

    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), shape.data(), shape.size());

    const char* inNames[] = {impl_->inputName.c_str()};
    const char* outNames[] = {impl_->outputName.c_str()};
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, inNames, &input,
                                       1, outNames, 1);

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto dims = info.GetShape();
    const int mapH = static_cast<int>(dims[dims.size() - 2]);
    const int mapW = static_cast<int>(dims[dims.size() - 1]);
    cv::Mat probMap(mapH, mapW, CV_32FC1,
                    const_cast<float*>(outputs[0].GetTensorData<float>()));

    cv::Mat binary;
    cv::threshold(probMap, binary, cfg.binThresh, 1.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1, 255);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const float mapToImgX = static_cast<float>(resized.cols) / mapW;
    const float mapToImgY = static_cast<float>(resized.rows) / mapH;

    std::vector<TextBox> boxes;
    for (const auto& contour : contours) {
        if (contour.size() < 3) continue;
        cv::RotatedRect rect = cv::minAreaRect(contour);
        if (std::min(rect.size.width, rect.size.height) <
            static_cast<float>(cfg.minBoxSize)) {
            continue;
        }

        const float score = boxScore(probMap, rect);
        if (score < cfg.boxThresh) continue;

        rect = unclip(rect, cfg.unclipRatio);

        TextBox box;
        box.score = score;
        auto corners = orderedCorners(rect);
        for (size_t i = 0; i < 4; ++i) {
            box.points[i].x = std::clamp(
                corners[i].x * mapToImgX * scaleX, 0.0F,
                static_cast<float>(imageBgr.cols - 1));
            box.points[i].y = std::clamp(
                corners[i].y * mapToImgY * scaleY, 0.0F,
                static_cast<float>(imageBgr.rows - 1));
        }
        boxes.push_back(box);
    }
    return boxes;
}

}  // namespace litera::ocr
