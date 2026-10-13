#include "litera/ocr/ctc_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace litera::ocr {

namespace {

std::vector<std::string> loadDict(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("CtcDecoder: cannot open dictionary: " + path.string());
    }
    std::vector<std::string> chars;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        chars.push_back(line);
    }
    if (chars.empty()) {
        throw std::runtime_error("CtcDecoder: empty dictionary: " + path.string());
    }
    return chars;
}

}  

CtcDecoder::CtcDecoder(const std::filesystem::path& dictPath, bool useSpaceChar)
    : CtcDecoder(loadDict(dictPath), useSpaceChar) {}

CtcDecoder::CtcDecoder(std::vector<std::string> characters, bool useSpaceChar) {
    charset_.reserve(characters.size() + 2);
    charset_.emplace_back("");
    for (auto& c : characters) charset_.push_back(std::move(c));
    if (useSpaceChar) charset_.emplace_back(" ");
}

CtcDecoder::Decoded CtcDecoder::decode(const float* logits, int timesteps,
                                       int numClasses) const {
    if (numClasses != classCount()) {
        throw std::invalid_argument(
            "CtcDecoder: model outputs " + std::to_string(numClasses) +
            " classes, charset has " + std::to_string(classCount()) +
            " (check dictionary / use_space_char)");
    }

    Decoded out;
    int prevIndex = 0;
    double confSum = 0.0;
    int emitted = 0;

    for (int t = 0; t < timesteps; ++t) {
        const float* row = logits + static_cast<size_t>(t) * numClasses;
        const float* maxIt = std::max_element(row, row + numClasses);
        const int idx = static_cast<int>(maxIt - row);

        if (idx != 0 && idx != prevIndex) {
            out.text += charset_[static_cast<size_t>(idx)];

            // PP-OCRv3 outputs are already softmaxed; avoid double softmax.
            double rowSum = 0.0;
            bool normalized = true;
            for (int c = 0; c < numClasses; ++c) {
                const double v = static_cast<double>(row[c]);
                rowSum += v;
                if (v < -1e-3 || v > 1.0 + 1e-3) normalized = false;
            }
            normalized = normalized && std::abs(rowSum - 1.0) < 1e-2;

            double prob;
            if (normalized) {
                prob = static_cast<double>(*maxIt);
            } else {
                double denom = 0.0;
                for (int c = 0; c < numClasses; ++c) {
                    denom += std::exp(static_cast<double>(row[c]) - *maxIt);
                }
                prob = 1.0 / denom;
            }
            confSum += prob;
            ++emitted;
        }
        prevIndex = idx;
    }

    out.confidence = emitted > 0 ? static_cast<float>(confSum / emitted) : 0.0F;
    return out;
}

}  // namespace litera::ocr
