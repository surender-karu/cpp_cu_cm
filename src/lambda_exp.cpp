#include "lambda_exp.h"

namespace cpp_fin {
void hw() {
  std::string cVar("Hello");

  auto hello = [&cVar](const std::string &s) {
    std::cout << cVar << " " << s << std::endl;
  };

  hello(std::string("C"));
  hello(std::string("C++"));
}
} // namespace cpp_fin