#include "litera/ocr/ocr_engine.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstring>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <det.onnx> <rec.onnx> <dict.txt> <image> "
                     "[--tiled]\n"
                     "  --tiled  split large images into overlapping tiles and\n"
                     "           recognize them in parallel (Map-Reduce + NMS)\n"
                     "example:\n"
                     "  litera-ocr models/det_db.onnx models/rec_cyrillic.onnx \\\n"
                     "         assets/dicts/cyrillic_dict.txt ad_photo.jpg\n",
                     argv[0]);
        return 2;
    }

    const bool tiled = argc > 5 && std::strcmp(argv[5], "--tiled") == 0;

    try {
        litera::ocr::OcrEngine engine(argv[1], argv[2], argv[3]);

        cv::Mat image = cv::imread(argv[4], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::fprintf(stderr, "error: cannot read image %s\n", argv[4]);
            return 1;
        }

        litera::ocr::OcrResult result =
            tiled ? engine.recognizeTiled(image) : engine.recognize(image);
        std::printf("-- %zu line(s) --\n", result.lines.size());
        for (const auto& line : result.lines) {
            std::printf("[%.2f] %s\n", line.confidence, line.text.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
