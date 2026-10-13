#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace litera::ocr {

// Class 0 is the CTC blank, then dictionary chars, then optional space.
class CtcDecoder {
public:
    struct Decoded {
        std::string text;
        float confidence = 0.0F;
    };

    CtcDecoder(const std::filesystem::path& dictPath, bool useSpaceChar = true);

    explicit CtcDecoder(std::vector<std::string> characters, bool useSpaceChar = true);

    [[nodiscard]] Decoded decode(const float* logits, int timesteps, int numClasses) const;

    [[nodiscard]] int classCount() const { return static_cast<int>(charset_.size()); }

private:
    std::vector<std::string> charset_;
};

}  // namespace litera::ocr
