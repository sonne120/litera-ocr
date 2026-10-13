#include "litera/ocr/ctc_decoder.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace litera::ocr {
namespace {

const std::vector<std::string> kChars = {"к", "и", "ї", "в", "2", "м", "."};

std::vector<float> makeLogits(const std::vector<int>& path, int numClasses) {
    std::vector<float> logits(path.size() * numClasses, 0.0F);
    for (size_t t = 0; t < path.size(); ++t) {
        logits[t * numClasses + path[t]] = 10.0F;
    }
    return logits;
}

TEST(CtcDecoder, DecodesSimpleSequence) {
    CtcDecoder decoder(kChars, true);
    const std::vector<int> path = {1, 2, 3, 4};
    auto logits = makeLogits(path, decoder.classCount());

    auto out = decoder.decode(logits.data(), 4, decoder.classCount());
    EXPECT_EQ(out.text, "київ");
    EXPECT_GT(out.confidence, 0.9F);
}

TEST(CtcDecoder, CollapsesRepeatsAndBlanks) {
    CtcDecoder decoder(kChars, true);
    const std::vector<int> path = {1, 1, 0, 2, 0, 2, 4};
    auto logits = makeLogits(path, decoder.classCount());

    auto out = decoder.decode(logits.data(), static_cast<int>(path.size()),
                              decoder.classCount());
    EXPECT_EQ(out.text, "киив");
}

TEST(CtcDecoder, EmitsSpaceAsLastClass) {
    CtcDecoder decoder(kChars, true);
    const int space = decoder.classCount() - 1;
    const std::vector<int> path = {5, space, 6, 7};
    auto logits = makeLogits(path, decoder.classCount());

    auto out = decoder.decode(logits.data(), 4, decoder.classCount());
    EXPECT_EQ(out.text, "2 м.");
}

TEST(CtcDecoder, AllBlanksGiveEmptyText) {
    CtcDecoder decoder(kChars, true);
    const std::vector<int> path = {0, 0, 0};
    auto logits = makeLogits(path, decoder.classCount());

    auto out = decoder.decode(logits.data(), 3, decoder.classCount());
    EXPECT_TRUE(out.text.empty());
    EXPECT_FLOAT_EQ(out.confidence, 0.0F);
}

TEST(CtcDecoder, RejectsClassCountMismatch) {
    CtcDecoder decoder(kChars, true);
    std::vector<float> logits(5 * 3, 0.0F);
    EXPECT_THROW(decoder.decode(logits.data(), 5, 3), std::invalid_argument);
}

TEST(CtcDecoder, LoadsRealCyrillicDict) {
    CtcDecoder decoder(LITERA_OCR_TEST_DATA_DIR "/assets/dicts/cyrillic_dict.txt", true);
    EXPECT_EQ(decoder.classCount(), 519);
}

}  // namespace
}  // namespace litera::ocr
