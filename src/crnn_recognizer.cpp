#include "litera/ocr/crnn_recognizer.hpp"

#include "litera/ocr/preprocess.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace litera::ocr {

namespace {

// Map Latin look-alikes to Cyrillic in Cyrillic-dominant lines.
std::string normalizeCyrillicHomoglyphs(const std::string& s) {
    int cyr = 0;
    int lat = 0;
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            if (std::isalpha(c)) ++lat;
            ++i;
        } else if ((c == 0xD0 || c == 0xD1) && i + 1 < s.size()) {
            ++cyr;
            i += 2;
        } else {
            ++i;
        }
    }
    if (cyr == 0 || cyr < lat) return s;

    static const std::unordered_map<char, const char*> homoglyph = {
        {'A', "А"}, {'B', "В"}, {'C', "С"}, {'E', "Е"}, {'H', "Н"}, {'I', "І"},
        {'K', "К"}, {'M', "М"}, {'O', "О"}, {'P', "Р"}, {'T', "Т"}, {'X', "Х"},
        {'Y', "У"}, {'a', "а"}, {'b', "в"}, {'c', "с"}, {'e', "е"}, {'i', "і"},
        {'o', "о"}, {'p', "р"}, {'t', "т"}, {'x', "х"}, {'y', "у"}};

    std::string out;
    out.reserve(s.size() + s.size() / 2);
    for (const char ch : s) {
        const auto it = homoglyph.find(ch);
        if (it != homoglyph.end()) {
            out += it->second;
        } else {
            out += ch;
        }
    }
    return out;
}

bool hasRecognizableContent(const std::string& s) {
    for (const unsigned char c : s) {
        if (c >= 0x80) return true;
        if (std::isalnum(c)) return true;
    }
    return false;
}

}  // namespace

struct CrnnRecognizer::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "litera-ocr-rec"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::string outputName;
    Config config;
    CtcDecoder decoder;

    Impl(const std::filesystem::path& modelPath,
         const std::filesystem::path& dictPath, Config cfg)
        : config(cfg), decoder(dictPath, cfg.useSpaceChar) {
        options.SetIntraOpNumThreads(2);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);

        Ort::AllocatorWithDefaultOptions alloc;
        inputName = session->GetInputNameAllocated(0, alloc).get();
        outputName = session->GetOutputNameAllocated(0, alloc).get();
    }
};

CrnnRecognizer::CrnnRecognizer(const std::filesystem::path& modelPath,
                               const std::filesystem::path& dictPath)
    : CrnnRecognizer(modelPath, dictPath, Config{}) {}

CrnnRecognizer::CrnnRecognizer(const std::filesystem::path& modelPath,
                               const std::filesystem::path& dictPath,
                               Config config)
    : impl_(std::make_unique<Impl>(modelPath, dictPath, config)) {}

CrnnRecognizer::~CrnnRecognizer() = default;
CrnnRecognizer::CrnnRecognizer(CrnnRecognizer&&) noexcept = default;
CrnnRecognizer& CrnnRecognizer::operator=(CrnnRecognizer&&) noexcept = default;

CtcDecoder::Decoded CrnnRecognizer::recognize(const cv::Mat& lineImage) const {
    CV_Assert(!lineImage.empty());
    const auto& cfg = impl_->config;

    cv::Mat img;
    if (cfg.inputChannels == 1 && lineImage.channels() == 3) {
        cv::cvtColor(lineImage, img, cv::COLOR_BGR2GRAY);
    } else if (cfg.inputChannels == 3 && lineImage.channels() == 1) {
        cv::cvtColor(lineImage, img, cv::COLOR_GRAY2BGR);
    } else {
        img = lineImage;
    }

    // Model expects dark text on a light background.
    {
        const cv::Scalar m = cv::mean(img);
        const double lum = img.channels() == 1
                               ? m[0]
                               : 0.114 * m[0] + 0.587 * m[1] + 0.299 * m[2];
        if (lum < 110.0) {
            cv::bitwise_not(img, img);
        }
    }

    const float ratio =
        static_cast<float>(img.cols) / static_cast<float>(img.rows);
    int targetW = static_cast<int>(std::ceil(cfg.inputHeight * ratio));
    targetW = std::clamp(targetW, 8, cfg.maxWidth);
    targetW = (targetW + 7) / 8 * 8;
    cv::resize(img, img, cv::Size(targetW, cfg.inputHeight), 0, 0,
               cv::INTER_LINEAR);

    std::vector<float> blob =
        toNchwBlob(img, cv::Scalar::all(cfg.mean), cv::Scalar::all(cfg.stdv));
    const std::array<int64_t, 4> shape{1, cfg.inputChannels, cfg.inputHeight,
                                       targetW};

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
    const float* data = outputs[0].GetTensorData<float>();

    int timesteps = 0;
    int numClasses = 0;
    if (dims.size() == 3) {
        timesteps = static_cast<int>(dims[0] == 1 ? dims[1] : dims[0]);
        numClasses = static_cast<int>(dims[2]);
    } else if (dims.size() == 2) {
        timesteps = static_cast<int>(dims[0]);
        numClasses = static_cast<int>(dims[1]);
    } else {
        throw std::runtime_error("CrnnRecognizer: unexpected output rank " +
                                 std::to_string(dims.size()));
    }

    CtcDecoder::Decoded decoded = impl_->decoder.decode(data, timesteps, numClasses);
    decoded.text = normalizeCyrillicHomoglyphs(decoded.text);
    if (!hasRecognizableContent(decoded.text)) {
        decoded.text.clear();
    }
    return decoded;
}

}  
