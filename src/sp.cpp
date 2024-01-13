#include "sp.h"
#include "common.h"

namespace cpp_fin {

void basic_sp() {
  // Creating shared pointers with default deleters
  SP<value_type> sp1;          // empty shared ptr
  SP<value_type> sp2(nullptr); // empty shared ptr for
                               // C++11 nullptr

  SP<value_type> sp3(new value_type(148.413)); // ptr owning raw ptr
  SP<value_type> sp4(sp3);                     // share ownership with sp3
  SP<value_type> sp5(sp4);                     // share ownership with sp4
                                               // and sp3

  // The number of shared owners
  std::cout << "sp2 shared # " << sp2.use_count() << std::endl;
  std::cout << "sp3 shared # " << sp3.use_count() << std::endl;
  std::cout << "sp4 shared # " << sp4.use_count() << std::endl;

  sp3 = sp2; // sp3 now shares ownership with sp2;
             // sp3 no longer has ownership of its previous resource
  std::cout << "sp3 shared # " << sp3.use_count() << std::endl;
  std::cout << "sp4 shared # " << sp4.use_count() << std::endl;
}

void sp_deleter() {
  // Creating shared pointers with user-defined deleters
  // Deleter as function object
  SP<value_type> sp(new value_type(148.413), Deleter<value_type>());

  // Deleter as lambda function
  SP<value_type> sp2(new value_type(148.413), [](value_type *p) {
    std::cout << "bye" << std::endl;
    delete p;
  });

  // Stored lambda function as deleter
  auto deleter = [](value_type *p) {
    std::cout << "bye from stored lambda." << std::endl;
    delete p;
  };

  SP<value_type> sp32(new value_type(148.413), deleter);
}

void sp_1() {
  {
    auto sp = std::make_shared<int>(42);
    (*sp)++;
    std::cout << "sp: " << *sp << std::endl; // 43

    auto sp2 = std::make_shared<Point2d>(-1.0, 2.0);
    (*sp2).print(); // (-1, 2)

    auto sp3 = std::make_shared<Point2d>();
    (*sp3).print(); // (0,0)
  }

  {
    // More efficient ways to construct shared pointers
    auto sp = std::allocate_shared<int>(std::allocator<int>(), 42);
    (*sp)++;
    std::cout << "sp: " << *sp << std::endl; // 43

    auto sp2 = std::allocate_shared<Point2d>(std::allocator<int>(), -1.0, 2.0);
    (*sp2).print(); // (-1, 2)
  }
}

void sp_2() {
  // Reset
  std::cout << "Reset\n";
  SP<value_type> sp1(new value_type(148.413));
  SP<value_type> sp2(sp1);
  SP<value_type> sp3(sp2);

  std::cout << "sp3 shared # " << sp3.use_count() << std::endl; // 3

  SP<value_type> sp4(new value_type(42.0));
  SP<value_type> sp5(sp4);

  std::cout << "sp5 shared # " << sp5.use_count() << std::endl; // 2

  sp3.reset();
  std::cout << "sp3 shared # " << sp3.use_count() << std::endl; // 0
  std::cout << "sp2 shared # " << sp2.use_count() << std::endl; // 2

  sp3.reset(new value_type(3.1415));
  std::cout << "sp3 shared # " << sp3.use_count() << std::endl; // 1
  std::cout << "sp2 shared # " << sp2.use_count() << std::endl; // 2

  sp2.reset(new value_type(3.1415), Deleter<value_type>());
  std::cout << "sp2 shared # " << sp2.use_count() << std::endl; // 1

  std::cout << "sp2 sole owner? " << std::boolalpha << sp2.unique()
            << std::endl;
  // true

  std::cout << "Which on is last?" << std::endl;
}

void up() {
  {
    auto up = std::make_unique<int>(42);
    (*up)++;
    std::cout << "up: " << *up << std::endl;

    auto up2 = std::make_unique<Point2d>(-1.0, 2.0);
    (*up2).print();

    auto up3 = std::make_unique<Point2d>();
    (*up3).print();
  }

  std::cout << "Out of scope" << std::endl;

  {
    UP<value_type> up1(new value_type(148.413));
    up1.reset();
    assert(up1 == nullptr);
    // std::cout << "reset: " << *up1 << '\n';

    up1.reset(new value_type(3.1415));
    std::cout << "reset: " << *up1 << std::endl;
    // Give ownership back to caller without calling deleter
    std::cout << "Release unique pointer\n";
    auto up = std::make_unique<Point2d>(42.0, 44.5);
    Point2d *fp = up.release();

    assert(up.get() == nullptr);
    std::cout << "No longer owned by unique_ptr..." << std::endl;

    (*fp).print();

    delete fp; // Destructor of Point2d called
  }

  try {
    // Unique pointers

    // Stored lambda function as deleter
    auto deleter = [](value_type *p) {
      std::cout << "bye, bye unique pointer\n";
      delete p;
    };

    UPD<value_type, decltype(deleter)> upd(new value_type(148.413), deleter);

    throw -1;
  } catch (int &n) {
    std::cout << "error but memory is cleaned up" << std::endl;
  }
}

void wp_0() {
  // Create a default weak pointer
  std::weak_ptr<value_type> wp;
  std::cout << "Expired wp? " << std::boolalpha << wp.expired() << std::endl;

  // Create a weak pointer from a shared pointer
  std::shared_ptr<value_type> sp(new value_type(3.1415));
  std::cout << "Reference count: " << sp.use_count() << std::endl;

  // Assign weak pointer to shared pointer
  wp = sp;
  std::cout << "Reference count: " << sp.use_count() << std::endl;

  std::weak_ptr<value_type> wp2(sp);
  std::cout << "Reference count: " << sp.use_count() << std::endl;

  wp = sp;
  std::shared_ptr<value_type> sp2(wp);
  std::cout << "Reference count, sp2: " << sp2.use_count() << std::endl;
  std::cout << "Expired wp? " << wp.expired() << std::endl;

  std::shared_ptr<value_type> sp3 = wp.lock();
  std::cout << "Reference count: " << sp3.use_count() << std::endl;
  std::cout << "Reference count: " << sp.use_count() << std::endl;

  // Event notification (Observer) pattern and weak pointers
  std::shared_ptr<value_type> spA(new value_type(3.1415));

  std::weak_ptr<value_type> wA(spA);
  std::weak_ptr<value_type> wB(spA);

  std::cout << "wA expired: " << wA.expired() << std::endl;
  std::cout << "wB expired: " << wB.expired() << std::endl;

  spA.reset();
  std::cout << "After reset: " << std::endl;
  std::cout << "wA expired: " << wA.expired() << std::endl;
  std::cout << "wB expired: " << wB.expired() << std::endl;

  std::cout << "Reference count: " << spA.use_count() << std::endl;
}

} // namespace cpp_fin