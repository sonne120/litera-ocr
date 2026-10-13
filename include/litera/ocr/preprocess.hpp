#pragma once

#include "litera/ocr/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace litera::ocr {

cv::Mat resizeForDetection(const cv::Mat& srcBgr, int maxSide, float& scaleX,
                           float& scaleY);

std::vector<float> toNchwBlob(const cv::Mat& bgr, const cv::Scalar& meanRgb,
                              const cv::Scalar& stdRgb);

cv::Mat enhanceForOcr(const cv::Mat& srcBgr, double upscale);

cv::Mat cropTextRegion(const cv::Mat& srcBgr, const TextBox& box);

void sortReadingOrder(std::vector<TextBox>& boxes);

}  // namespace litera::ocr
