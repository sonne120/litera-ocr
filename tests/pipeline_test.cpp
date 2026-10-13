#include "litera/ocr/crnn_recognizer.hpp"
#include "litera/ocr/db_detector.hpp"
#include "litera/ocr/preprocess.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <filesystem>

namespace litera::ocr {
namespace {

namespace fs = std::filesystem;

const fs::path kDetModel = LITERA_OCR_TEST_DATA_DIR "/models/dbnet.onnx";
const fs::path kRecModel = LITERA_OCR_TEST_DATA_DIR "/models/crnn_lite_lstm.onnx";
const fs::path kDict = LITERA_OCR_TEST_DATA_DIR "/assets/dicts/cyrillic_dict.txt";

cv::Mat makeAdImage() {
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(245, 245, 245));
    cv::putText(img, "RENT 2-room flat", {40, 120},
                cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(20, 20, 20), 3);
    cv::putText(img, "Kyiv, Obolon, 650$", {40, 220},
                cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(20, 20, 20), 3);
    cv::putText(img, "tel 067 123 45 67", {40, 320},
                cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(20, 20, 20), 3);
    return img;
}

TEST(DbDetectorIT, FindsTextLinesOnSyntheticAd) {
    if (!fs::exists(kDetModel)) GTEST_SKIP() << "det model not present";

    DbDetector detector(kDetModel);
    cv::Mat img = makeAdImage();
    auto boxes = detector.detect(img);

    EXPECT_GE(boxes.size(), 3U);
    for (const auto& b : boxes) {
        EXPECT_GT(b.score, 0.5F);
        EXPECT_GT(b.width(), b.height());
        for (const auto& p : b.points) {
            EXPECT_GE(p.x, 0.0F);
            EXPECT_LT(p.x, static_cast<float>(img.cols));
            EXPECT_GE(p.y, 0.0F);
            EXPECT_LT(p.y, static_cast<float>(img.rows));
        }
    }
}

TEST(DbDetectorIT, EmptyImageYieldsNoBoxes) {
    if (!fs::exists(kDetModel)) GTEST_SKIP() << "det model not present";

    DbDetector detector(kDetModel);
    cv::Mat blank(320, 320, CV_8UC3, cv::Scalar(250, 250, 250));
    auto boxes = detector.detect(blank);
    EXPECT_TRUE(boxes.empty());
}

// Model has a Chinese charset: only the inference path is checked, not the text.
TEST(CrnnRecognizerIT, RunsInferenceEndToEnd) {
    if (!fs::exists(kRecModel)) GTEST_SKIP() << "rec model not present";

    const fs::path dummyDict = fs::temp_directory_path() / "dummy_dict.txt";
    {
        std::FILE* f = std::fopen(dummyDict.c_str(), "w");
        ASSERT_NE(f, nullptr);
        for (int i = 0; i < 5529; ++i) std::fprintf(f, "x\n");
        std::fclose(f);
    }

    CrnnRecognizer::Config cfg;
    cfg.inputChannels = 3;
    cfg.inputHeight = 32;
    cfg.useSpaceChar = true;
    CrnnRecognizer rec(kRecModel, dummyDict, cfg);

    cv::Mat line(32, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(line, "650", {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 0), 2);

    auto out = rec.recognize(line);
    EXPECT_GE(out.confidence, 0.0F);
    EXPECT_LE(out.confidence, 1.0F);
}

TEST(CrnnRecognizerIT, ThrowsOnDictModelMismatch) {
    if (!fs::exists(kRecModel)) GTEST_SKIP() << "rec model not present";

    CrnnRecognizer::Config cfg;
    cfg.inputChannels = 3;
    cfg.inputHeight = 32;
    CrnnRecognizer rec(kRecModel, kDict, cfg);
    cv::Mat line(32, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    EXPECT_THROW(rec.recognize(line), std::invalid_argument);
}

}  // namespace
}  // namespace litera::ocr
