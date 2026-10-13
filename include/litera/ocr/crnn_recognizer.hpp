#pragma once

#include "litera/ocr/ctc_decoder.hpp"
#include "litera/ocr/types.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace litera::ocr {

class CrnnRecognizer {
public:
    struct Config {
        int inputChannels = 3;
        int inputHeight = 48;
        int maxWidth = 512;
        float mean = 0.5F;
        float stdv = 0.5F;
        bool useSpaceChar = true;
    };

    CrnnRecognizer(const std::filesystem::path& modelPath,
                   const std::filesystem::path& dictPath);
    CrnnRecognizer(const std::filesystem::path& modelPath,
                   const std::filesystem::path& dictPath, Config config);
    ~CrnnRecognizer();
    CrnnRecognizer(CrnnRecognizer&&) noexcept;
    CrnnRecognizer& operator=(CrnnRecognizer&&) noexcept;

    [[nodiscard]] CtcDecoder::Decoded recognize(const cv::Mat& lineImage) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  
