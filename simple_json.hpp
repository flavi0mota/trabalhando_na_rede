#ifndef SIMPLE_JSON_HPP
#define SIMPLE_JSON_HPP

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

namespace SimpleJson {

inline std::string double_to_string(double val) {
  std::string s = std::to_string(val);
  size_t dot_pos = s.find('.');

  if (dot_pos != std::string::npos) {
    size_t last_non_zero = s.find_last_not_of('0');
    if (last_non_zero == dot_pos) {
      s.erase(dot_pos);
    } else if (last_non_zero != std::string::npos) {
      s.erase(last_non_zero + 1);
    }
  }

  return s;
}

inline std::string build_request(const std::vector<double>& features) {
  std::ostringstream oss;
  oss << "{\"features\":[";

  for (size_t i = 0; i < features.size(); i++) {
    if (i > 0) oss << ",";
    oss << double_to_string(features[i]);
  }

  oss << "]}";
  return oss.str();
}

inline int parse_prediction(const std::string& json) {
    const std::string key = "\"prediction\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("Chave 'prediction' não encontrada");
    }
    pos += key.length();

    while (pos < json.size() && (std::isspace(json[pos]) || json[pos] == ':')) {
        ++pos;
    }

    size_t start = pos;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-')) {
        ++pos;
    }
    if (start == pos) {
        throw std::runtime_error("Número não encontrado após 'prediction'");
    }
    std::string num_str = json.substr(start, pos - start);
    return std::stoi(num_str);
}

}

#endif
