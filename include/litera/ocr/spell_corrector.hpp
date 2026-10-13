#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace litera::ocr {

// Fixes only tokens one edit away from a clearly dominant lexicon word.
class SpellCorrector {
public:
    SpellCorrector() = default;
    explicit SpellCorrector(const std::filesystem::path& lexiconPath) {
        load(lexiconPath);
    }

    void load(const std::filesystem::path& lexiconPath) {
        words_.clear();
        std::ifstream in(lexiconPath);
        if (!in) return;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string word;
            long freq = 1;
            ss >> word >> freq;
            if (!word.empty()) words_[word] = freq;
        }
    }

    [[nodiscard]] bool loaded() const { return !words_.empty(); }

    [[nodiscard]] std::string correct(const std::string& text) const {
        if (words_.empty()) return text;
        std::string out;
        out.reserve(text.size());
        std::string token;
        std::size_t i = 0;
        while (i < text.size()) {
            const std::string cp = nextCodePoint(text, i);
            if (isWordChar(cp)) {
                token += cp;
            } else {
                if (!token.empty()) { out += correctToken(token); token.clear(); }
                out += cp;
            }
        }
        if (!token.empty()) out += correctToken(token);
        return out;
    }

private:
    std::unordered_map<std::string, long> words_;

    static constexpr long kDominance = 3;
    static constexpr long kMinFreq = 3;

    static std::string nextCodePoint(const std::string& s, std::size_t& i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = 1;
        if ((c & 0x80U) == 0x00U) len = 1;
        else if ((c & 0xE0U) == 0xC0U) len = 2;
        else if ((c & 0xF0U) == 0xE0U) len = 3;
        else if ((c & 0xF8U) == 0xF0U) len = 4;
        len = std::min(len, s.size() - i);
        std::string cp = s.substr(i, len);
        i += len;
        return cp;
    }

    static std::uint32_t decode(const std::string& cp) {
        const unsigned char c = static_cast<unsigned char>(cp[0]);
        if (cp.size() == 1) return c;
        if (cp.size() == 2) {
            return (static_cast<std::uint32_t>(c & 0x1FU) << 6) |
                   static_cast<std::uint32_t>(cp[1] & 0x3F);
        }
        return 0;
    }

    static std::string encode(std::uint32_t cp) {
        std::string s;
        if (cp < 0x80U) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800U) {
            s.push_back(static_cast<char>(0xC0U | (cp >> 6)));
            s.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
        return s;
    }

    static bool isCyrillic(const std::string& cp) {
        const std::uint32_t v = decode(cp);
        return v >= 0x0400U && v <= 0x04FFU;
    }
    static bool isAsciiDigit(const std::string& cp) {
        return cp.size() == 1 && cp[0] >= '0' && cp[0] <= '9';
    }
    static bool isAsciiAlpha(const std::string& cp) {
        return cp.size() == 1 &&
               std::isalpha(static_cast<unsigned char>(cp[0])) != 0;
    }
    static bool isApostrophe(const std::string& cp) {
        return cp == "'" || cp == "’";
    }
    static bool isWordChar(const std::string& cp) {
        return isCyrillic(cp) || isAsciiDigit(cp) || isApostrophe(cp) ||
               isAsciiAlpha(cp);
    }

    static std::uint32_t toLowerCp(std::uint32_t v) {
        if (v >= 0x0410U && v <= 0x042FU) return v + 0x20U;
        if (v >= 0x0400U && v <= 0x040FU) return v + 0x50U;
        if (v >= 0x0460U && v <= 0x04FFU && (v % 2U == 0U)) return v + 1U;
        return v;
    }
    static std::uint32_t toUpperCp(std::uint32_t v) {
        if (v >= 0x0430U && v <= 0x044FU) return v - 0x20U;
        if (v >= 0x0450U && v <= 0x045FU) return v - 0x50U;
        if (v >= 0x0460U && v <= 0x04FFU && (v % 2U == 1U)) return v - 1U;
        return v;
    }
    static bool isUpperLetter(const std::string& cp) {
        if (isCyrillic(cp)) {
            const std::uint32_t v = decode(cp);
            return toLowerCp(v) != v;
        }
        return cp.size() == 1 && cp[0] >= 'A' && cp[0] <= 'Z';
    }
    static std::string lowerCp(const std::string& cp) {
        if (isCyrillic(cp)) return encode(toLowerCp(decode(cp)));
        if (cp.size() == 1 && cp[0] >= 'A' && cp[0] <= 'Z') {
            return std::string(1, static_cast<char>(cp[0] + 32));
        }
        return cp;
    }
    static std::string upperCp(const std::string& cp) {
        if (isCyrillic(cp)) return encode(toUpperCp(decode(cp)));
        if (cp.size() == 1 && cp[0] >= 'a' && cp[0] <= 'z') {
            return std::string(1, static_cast<char>(cp[0] - 32));
        }
        return cp;
    }

    static std::vector<std::string> toCps(const std::string& s) {
        std::vector<std::string> v;
        std::size_t i = 0;
        while (i < s.size()) v.push_back(nextCodePoint(s, i));
        return v;
    }
    static std::size_t utf8Len(const std::string& s) {
        std::size_t n = 0;
        std::size_t i = 0;
        while (i < s.size()) { nextCodePoint(s, i); ++n; }
        return n;
    }

    static const std::vector<std::string>& alphabet() {
        static const std::vector<std::string> a = makeAlphabet();
        return a;
    }
    static std::vector<std::string> makeAlphabet() {
        static const std::uint32_t cps[] = {
            0x0430, 0x0431, 0x0432, 0x0433, 0x0491, 0x0434, 0x0435, 0x0454,
            0x0436, 0x0437, 0x0438, 0x0456, 0x0457, 0x0439, 0x043A, 0x043B,
            0x043C, 0x043D, 0x043E, 0x043F, 0x0440, 0x0441, 0x0442, 0x0443,
            0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044C, 0x044E,
            0x044F};
        std::vector<std::string> a;
        a.reserve(sizeof(cps) / sizeof(cps[0]) + 1);
        for (const std::uint32_t c : cps) a.push_back(encode(c));
        a.emplace_back("'");
        return a;
    }

    std::string correctToken(const std::string& token) const {
        const std::vector<std::string> cps = toCps(token);
        int letters = 0;
        int digits = 0;
        int cyr = 0;
        for (const auto& cp : cps) {
            if (isAsciiDigit(cp)) {
                ++digits;
            } else if (isCyrillic(cp)) {
                ++letters;
                ++cyr;
            } else if (isAsciiAlpha(cp)) {
                ++letters;
            }
        }
        if (letters < 3 || digits > letters || cyr * 2 < letters) return token;

        std::string lower;
        for (const auto& cp : cps) lower += lowerCp(cp);
        if (words_.find(lower) != words_.end()) return token;

        const std::vector<std::string> lcps = toCps(lower);
        std::unordered_map<std::string, long> found;

        auto consider = [&](const std::string& w) {
            const auto it = words_.find(w);
            if (it != words_.end()) found[w] = it->second;
        };

        for (std::size_t i = 0; i < lcps.size(); ++i) {
            std::string w;
            for (std::size_t j = 0; j < lcps.size(); ++j) {
                if (j != i) w += lcps[j];
            }
            if (utf8Len(w) >= 3) consider(w);
        }
        for (std::size_t i = 0; i < lcps.size(); ++i) {
            for (const auto& a : alphabet()) {
                if (a == lcps[i]) continue;
                std::string w;
                for (std::size_t j = 0; j < lcps.size(); ++j) {
                    w += (j == i) ? a : lcps[j];
                }
                consider(w);
            }
        }
        for (std::size_t i = 0; i <= lcps.size(); ++i) {
            for (const auto& a : alphabet()) {
                std::string w;
                for (std::size_t j = 0; j < i; ++j) w += lcps[j];
                w += a;
                for (std::size_t j = i; j < lcps.size(); ++j) w += lcps[j];
                consider(w);
            }
        }

        if (found.empty()) return token;

        std::string best;
        long bestFreq = -1;
        long secondFreq = -1;
        for (const auto& [w, f] : found) {
            if (f > bestFreq) {
                secondFreq = bestFreq;
                bestFreq = f;
                best = w;
            } else if (f > secondFreq) {
                secondFreq = f;
            }
        }
        if (bestFreq < kMinFreq) return token;
        if (secondFreq >= 0 && bestFreq < kDominance * secondFreq) return token;
        return recase(best, cps);
    }

    static std::string recase(const std::string& word,
                              const std::vector<std::string>& origCps) {
        int upper = 0;
        int lower = 0;
        std::string firstLetter;
        for (const auto& cp : origCps) {
            if (isCyrillic(cp) || isAsciiAlpha(cp)) {
                if (isUpperLetter(cp)) ++upper; else ++lower;
                if (firstLetter.empty()) firstLetter = cp;
            }
        }
        const std::vector<std::string> wcps = toCps(word);
        if (upper > 1 && lower == 0) {
            std::string s;
            for (const auto& cp : wcps) s += upperCp(cp);
            return s;
        }
        if (!firstLetter.empty() && isUpperLetter(firstLetter) && !wcps.empty()) {
            std::string s = upperCp(wcps[0]);
            for (std::size_t j = 1; j < wcps.size(); ++j) s += wcps[j];
            return s;
        }
        return word;
    }
};

}  // namespace litera::ocr
