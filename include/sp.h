#ifndef CPP_FIN_SP_H
#define CPP_FIN_SP_H

#include "common.h"

namespace cpp_fin {

using value_type = double;

template <typename T> struct Deleter {
  void operator()(T *t) const {
    std::cout << "delete memory from function object" << std::endl;
    delete t;
  }
};

template <typename T> using SP = std::shared_ptr<T>;

template <typename T> using UP = std::unique_ptr<T>;
template <typename T, typename Dp> using UPD = std::unique_ptr<T, Dp>;

struct Point2d {
  double x, y;
  Point2d() : x(0.0), y(0.0) {}
  Point2d(double x_val, double y_val) : x(x_val), y(y_val) {}
  void print() const { std::cout << "(" << x << "," << y << ")" << std::endl; }
  ~Point2d() { std::cout << "Point destroyed" << std::endl; }
};

void sp_1(void);
void sp_2(void);
void up(void);
void wp_0(void);
}

#endif
