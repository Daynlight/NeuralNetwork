#pragma once
#include "conf.h"
#include <fmt/core.h>
#include <fmt/color.h>
#include <string>
#include "dsl.h"

namespace UnitTests {
// =============== Helping Struct ===============
struct Point { int x, y; };
template<typename T>
class TestVector : public Essentials::Vector<T> {
public:
    unsigned int getBackIndex() { return this->_back; }
};


// =============== Comparators & Predicates ===============
int cmp_int(const int &a, const int &b) { return a - b; }
int is_even(const int &val) { return (val % 2) == 0; }
int point_cmp(const Point &a, const Point &b) { return a.x - b.x; }
void add(const int &el, int &ud) { ud += el; }
void multiply(int &el) { el *= 2; }


// ========================= ASSERT =========================
bool ASSERT_EQ_INT(std::string name, int expected, int actual) noexcept {
#ifdef LoudTests
  if(expected == actual) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={:d} got={:d}\n", name, expected, actual);  
#endif
  if(expected != actual)
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={:d} got={:d}\n", name, expected, actual); 
  return expected == actual;
};

bool ASSERT_EQ_SIZE(std::string name, size_t expected, size_t actual) noexcept {
#ifdef LoudTests
  if(expected == actual) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={} got={}\n", name, expected, actual);  
#endif
  if(expected != actual)
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={} got={}\n", name, expected, actual);
  return expected == actual;
};

bool ASSERT_EQ_FLOAT(std::string name, float expected, float actual, float eps) noexcept {
  float diff = std::abs(expected - actual);
#ifdef LoudTests
  if(diff < eps) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={} got={} (eps={})\n", name, expected, actual, eps); 
#endif   
  if(diff >= eps)
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={} got={} (eps={})\n", name, expected, actual, eps);
  return diff < eps;
};

bool ASSERT_TRUE(std::string name, bool cond) noexcept {
#ifdef LoudTests
  if(cond) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s}\n", name);
#endif
  if(!cond)     
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s}\n", name);
  return cond;
};






//=========================================================
//========================= Tests =========================
//=========================================================

// ===== INIT & DESTROY =====
bool test_vector_init() {
  Essentials::Vector<int> v;
  ASSERT_EQ_SIZE("init size", 0, v.getSize());
  ASSERT_TRUE("init capacity>=1", v.getCapacity() >= 1);
  return true;
}

// ===== RESIZE & CAPACITY =====
bool test_vector_resize() {
  Essentials::Vector<int> v;
  unsigned initial = v.getCapacity();
  v.resize();
  ASSERT_TRUE("resize doubles capacity", v.getCapacity() >= initial*2);
  v.resize();
  ASSERT_TRUE("resize quadruple capacity", v.getCapacity() >= initial*4);
  return true;
}

bool test_vector_reorder() {
  TestVector<int> v;
  v.reserve(5);
  v.pushHead(21);
  v.pushHead(24);
  v.pushBack(21);
  unsigned int back = v.getBackIndex();
  ASSERT_TRUE("reorder check back before reorder", back != 0);
  v.reorder();
  back = v.getBackIndex();
  ASSERT_TRUE("reorder check back after reorder", back == 0);
  return true;
}

bool test_vector_reserve() {
  Essentials::Vector<int> v;
  unsigned old = v.getCapacity();
  v.reserve(10);
  ASSERT_TRUE("reserve increases capacity", v.getCapacity()>=old+10);
  return true;
}

bool test_vector_set_capacity() {
  Essentials::Vector<int> v;
  v.setCapacity(20);
  ASSERT_EQ_SIZE("set_capacity=20", 20, v.getCapacity());
  v.setCapacity(2);
  ASSERT_EQ_SIZE("set_capacity=2", 2, v.getCapacity());
  return true;
}

bool test_vector_shrink() {
  Essentials::Vector<int> v;
  v.setCapacity(60);
  v.pushHead(1);
  v.pushHead(1);
  v.pushHead(1);
  ASSERT_EQ_SIZE("shrink check before shrink", 60, v.getCapacity());
  v.shrink();
  ASSERT_EQ_SIZE("shrink size==capacity", v.getSize(), v.getCapacity());
  return true;
}

// ===== PUSH METHODS =====
bool test_vector_push_head() {
  Essentials::Vector<int> v;
  v.reserve(10);
  v.pushHead(12);
  v.pushHead(32);
  v.pushHead(54);
  v.pushHead(11);
  v.pushHead(77);
  ASSERT_EQ_INT("push head value", 77, v.head());
  return true;
}

bool test_vector_push_back() {
  Essentials::Vector<int> v;
  v.pushBack(1);
  v.pushBack(2);
  v.pushBack(67);
  v.pushBack(99);
  v.pushBack(32);
  v.pushBack(12454);
  ASSERT_EQ_INT("push back value", 12454, v.back());
  return true;
}

bool test_vector_push_at() {
  Essentials::Vector<int> v;
  v.setCapacity(20);
  v.pushHead(1); v.pushHead(2); v.pushHead(3); v.pushHead(5); v.pushHead(3);
  v.pushBack(12); v.pushBack(123);
  v.pushAt(1, 77);
  ASSERT_EQ_INT("push at 1", 77, v.at(1));
  v.pushAt(3, 12);
  ASSERT_EQ_INT("push at 3", 12, v.at(3));
  ASSERT_EQ_INT("push at back", 123, v.back());
  ASSERT_EQ_INT("push at head", 3, v.head());
  return true;
}

// ===== POP METHODS =====
bool test_vector_pop_head() {
  Essentials::Vector<int> v;
  v.pushHead(5); v.pushHead(6);
  ASSERT_EQ_INT("pop head", 6, v.popHead());
  ASSERT_EQ_INT("pop head size", 1, v.getSize());
  return true;
}

bool test_vector_pop_back() {
  Essentials::Vector<int> v;
  v.pushHead(5); v.pushHead(6);
  ASSERT_EQ_INT("pop front", 5, v.popBack());
  ASSERT_EQ_INT("pop back size", 1, v.getSize());
  return true;
}

bool test_vector_pop_at() {
  Essentials::Vector<int> v;
  v.setCapacity(20);
  v.pushHead(1); v.pushHead(2); v.pushHead(3); v.pushHead(5);
  v.pushBack(12); v.pushBack(54);
  ASSERT_EQ_INT("pop_at 1", 12, v.popAt(1));
  ASSERT_EQ_INT("pop back size", 5, v.getSize());
  ASSERT_EQ_INT("pop_at 2", 2, v.popAt(2));
  ASSERT_EQ_INT("pop back size", 4, v.getSize());
  ASSERT_EQ_INT("pop at back", 54, v.back());
  ASSERT_EQ_INT("pop at head", 5, v.head());
  return true;
}

// ===== ERASE & CLEAR =====
bool test_vector_erase() {
  Essentials::Vector<int> v;
  v.setCapacity(10);
  v.pushHead(231); v.pushHead(2); v.pushHead(3);
  v.pushBack(1); v.pushBack(2);
  v.erase(1);
  ASSERT_EQ_INT("erase at 1", 231, v.at(1));
  ASSERT_EQ_INT("pop back size", 4, v.getSize());
  return true;
}

bool test_vector_clear() {
  Essentials::Vector<int> v;
  v.pushHead(1);
  v.clear();
  ASSERT_EQ_SIZE("clear resets size", 0, v.getSize());
  return true;
}

// ===== FIRST / LAST / AT =====
bool test_vector_first_last_at() {
  Essentials::Vector<int> v;
  v.pushHead(11); v.pushHead(22); v.pushHead(33);
  ASSERT_EQ_INT("last", 33, v.head());
  ASSERT_EQ_INT("first", 11, v.back());
  ASSERT_EQ_INT("at[1]", 22, v.at(1));
  return true;
}

// ===== GETTERS =====
bool test_vector_get_size() {
  Essentials::Vector<int> v;
  ASSERT_EQ_SIZE("get_size empty",0,v.getSize());
  v.pushHead(5);
  ASSERT_EQ_SIZE("get_size=1",1,v.getSize());
  return true;
}

bool test_vector_get_capacity() {
  Essentials::Vector<int> v;
  ASSERT_TRUE("get_capacity>=1",v.getCapacity()>=1);
  return true;
}

bool test_vector_is_empty() {
  Essentials::Vector<int> v;
  ASSERT_TRUE("is_empty true",v.isEmpty());
  v.pushHead(3);
  ASSERT_TRUE("is_empty false",!v.isEmpty());
  return true;
}

// ===== MIXED PUSH / POP =====
bool test_vector_mixed_push_pop() {
  Essentials::Vector<int> v;
  v.pushHead(1); v.pushBack(2); v.pushHead(3); v.pushBack(4);
  ASSERT_EQ_INT("mixed back",4,v.back());
  ASSERT_EQ_INT("mixed head",3,v.head());
  ASSERT_EQ_INT("pop Back",4,v.popBack());
  ASSERT_EQ_INT("pop Head",3,v.popHead());
  return true;
}

// ===== LARGE DATASET TESTS =====
bool test_vector_large_push() {
  Essentials::Vector<int> v;
  int N=1000;
  for(int i=0;i<N;i++) v.pushHead(i);
  ASSERT_EQ_SIZE("large push size",N,v.getSize());
  ASSERT_EQ_INT("large push head",N-1,v.head());
  ASSERT_EQ_INT("large push back",0,v.back());
  return true;
}

bool test_vector_large_push_front() {
  Essentials::Vector<int> v;
  int N=1000;
  for(int i=0;i<N;i++) v.pushBack(i);
  ASSERT_EQ_SIZE("large push head size",N,v.getSize());
  ASSERT_EQ_INT("large push head back",N-1,v. back());
  ASSERT_EQ_INT("large push head head",0,v.head());
  return true;
}

// ===== RUN ALL =====
bool Vector() {
  int total=0, passed=0;
#define RUN(f) total++; if(f()) passed++;

  RUN(test_vector_init);

  RUN(test_vector_resize); 
  RUN(test_vector_reorder); 
  RUN(test_vector_reserve);
  RUN(test_vector_set_capacity);
  RUN(test_vector_shrink);

  RUN(test_vector_push_head);
  RUN(test_vector_push_back);
  RUN(test_vector_push_at);
  RUN(test_vector_pop_head);
  RUN(test_vector_pop_back);
  RUN(test_vector_pop_at);

  RUN(test_vector_erase);
  RUN(test_vector_clear);

  RUN(test_vector_first_last_at);
  RUN(test_vector_get_size);
  RUN(test_vector_get_capacity);
  RUN(test_vector_is_empty);
  RUN(test_vector_mixed_push_pop);
  RUN(test_vector_large_push);
  RUN(test_vector_large_push_front);

#undef RUN
  if(passed == total) 
    fmt::print(fg(fmt::color::blue_violet), "=== Essentials::Vector Test Score: {:d}\\{:d} passed ===\n", passed, total);
  else 
    fmt::print(fg(fmt::color::dark_orange), "=== Essentials::Vector Test Score: {:d}\\{:d} passed ===\n", passed, total);
  return passed == total;
};
};