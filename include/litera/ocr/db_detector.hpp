#pragma once

#include "litera/ocr/types.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace litera::ocr {

class DbDetector {
public:
    struct Config {
        int maxSide = 960;
        float binThresh = 0.3F;
        float boxThresh = 0.5F;
        float unclipRatio = 2.2F;
        int minBoxSize = 5;
    };

    explicit DbDetector(const std::filesystem::path& modelPath);
    DbDetector(const std::filesystem::path& modelPath, Config config);
    ~DbDetector();
    DbDetector(DbDetector&&) noexcept;
    DbDetector& operator=(DbDetector&&) noexcept;

    [[nodiscard]] std::vector<TextBox> detect(const cv::Mat& imageBgr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace litera::ocr
