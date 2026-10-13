#include "litera/ocr/preprocess.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

namespace litera::ocr {
namespace {

TextBox makeBox(float x, float y, float w, float h) {
    TextBox b;
    b.points = {cv::Point2f{x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}};
    return b;
}

TEST(Preprocess, ResizeKeepsMultipleOf32AndScales) {
    cv::Mat img(700, 1300, CV_8UC3, cv::Scalar(10, 20, 30));
    float sx = 0.0F;
    float sy = 0.0F;
    cv::Mat out = resizeForDetection(img, 960, sx, sy);

    EXPECT_EQ(out.cols % 32, 0);
    EXPECT_EQ(out.rows % 32, 0);
    EXPECT_LE(std::max(out.cols, out.rows), 960 + 16);
    EXPECT_NEAR(out.cols * sx, 1300.0F, 32.0F);
    EXPECT_NEAR(out.rows * sy, 700.0F, 32.0F);
}

TEST(Preprocess, BlobNormalizesToMinusOneOne) {
    cv::Mat img(2, 2, CV_8UC3, cv::Scalar(255, 255, 255));
    auto blob = toNchwBlob(img, cv::Scalar::all(0.5), cv::Scalar::all(0.5));
    ASSERT_EQ(blob.size(), 3U * 2 * 2);
    for (float v : blob) EXPECT_FLOAT_EQ(v, 1.0F);

    img.setTo(cv::Scalar(0, 0, 0));
    blob = toNchwBlob(img, cv::Scalar::all(0.5), cv::Scalar::all(0.5));
    for (float v : blob) EXPECT_FLOAT_EQ(v, -1.0F);
}

TEST(Preprocess, BlobUsesRgbPlaneOrder) {
    cv::Mat img(1, 1, CV_8UC3, cv::Scalar(255, 0, 0));
    auto blob = toNchwBlob(img, cv::Scalar::all(0.0), cv::Scalar::all(1.0));
    ASSERT_EQ(blob.size(), 3U);
    EXPECT_FLOAT_EQ(blob[0], 0.0F);
    EXPECT_FLOAT_EQ(blob[1], 0.0F);
    EXPECT_FLOAT_EQ(blob[2], 1.0F);
}

TEST(Preprocess, ReadingOrderSortsLinesThenColumns) {
    auto A = makeBox(10, 10, 80, 20);
    auto B = makeBox(120, 12, 80, 20);
    auto C = makeBox(15, 60, 80, 20);
    auto D = makeBox(130, 58, 80, 20);

    std::vector<TextBox> boxes = {D, A, C, B};
    sortReadingOrder(boxes);

    EXPECT_FLOAT_EQ(boxes[0].points[0].x, 10.0F);
    EXPECT_FLOAT_EQ(boxes[1].points[0].x, 120.0F);
    EXPECT_FLOAT_EQ(boxes[2].points[0].x, 15.0F);
    EXPECT_FLOAT_EQ(boxes[3].points[0].x, 130.0F);
}

TEST(Preprocess, CropExtractsRegionContent) {
    cv::Mat img(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(img, {50, 40}, {150, 80}, cv::Scalar(255, 255, 255),
                  cv::FILLED);

    auto box = makeBox(50, 40, 100, 40);
    cv::Mat crop = cropTextRegion(img, box);

    ASSERT_FALSE(crop.empty());
    EXPECT_NEAR(crop.cols, 100, 2);
    EXPECT_NEAR(crop.rows, 40, 2);
    EXPECT_GT(cv::mean(crop)[0], 200.0);
}

TEST(Preprocess, CropRotatesVerticalRegions) {
    cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    auto box = makeBox(100, 50, 30, 200);
    cv::Mat crop = cropTextRegion(img, box);
    EXPECT_GT(crop.cols, crop.rows);
}

}  // namespace
}  // namespace litera::ocr
