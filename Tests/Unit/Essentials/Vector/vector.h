#pragma once
#include <fmt/core.h>
#include <fmt/color.h>
#include <string>
#include "dsl.h"

namespace UnitTests {
// =============== Helping Struct ===============
struct Point { int x, y; };



// =============== Comparators & Predicates ===============
int cmp_int(const int &a, const int &b) { return a - b; }
int is_even(const int &val) { return (val % 2) == 0; }
int point_cmp(const Point &a, const Point &b) { return a.x - b.x; }
void add(const int &el, int &ud) { ud += el; }
void multiply(int &el) { el *= 2; }



// ========================= ASSERT =========================
bool ASSERT_EQ_INT(std::string name, int expected, int actual) noexcept {
  if(expected == actual) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={:d} got={:d}\n", name, expected, actual);
  else    
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={:d} got={:d}\n", name, expected, actual); 
  return expected == actual;
};

bool ASSERT_EQ_SIZE(std::string name, size_t expected, size_t actual) noexcept {
  if(expected == actual) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={} got={}\n", name, expected, actual);
  else    
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={} got={}\n", name, expected, actual);
  return expected == actual;
};

bool ASSERT_EQ_FLOAT(std::string name, float expected, float actual, float eps) noexcept {
  float diff = std::abs(expected - actual);
  if(diff < eps) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={} got={} (eps={})\n", name, expected, actual, eps);
  else    
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={} got={} (eps={})\n", name, expected, actual, eps);
  return diff < eps;
};

bool ASSERT_TRUE(std::string name, bool cond) noexcept {
  if(cond) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s}\n", name);
  else    
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
  return true;
}

bool test_vector_shrink() {
  Essentials::Vector<int> v;
  v.pushHead(1);
  v.pushHead(1);
  v.pushHead(1);
  v.shrink();
  ASSERT_EQ_SIZE("shrink size==capacity", v.getSize(), v.getCapacity());
  return true;
}

// ===== PUSH METHODS =====
bool test_vector_push() {
  Essentials::Vector<int> v;
  v.pushHead(42);
  ASSERT_EQ_INT("push value", 42, v.first());
  return true;
}

bool test_vector_push_front() {
  Essentials::Vector<int> v;
  v.pushHead(1);
  v.pushHead(2);
  v.pushBack(99);
  ASSERT_EQ_INT("push_front value", 99, v.first());
  return true;
}

bool test_vector_push_at() {
  Essentials::Vector<int> v;
  v.pushHead(1); v.pushHead(2); v.pushHead(3);
  v.pushAt(1, 77);
  ASSERT_EQ_INT("push_at middle", 77, v.at(1));
  return true;
}

// ===== POP METHODS =====
bool test_vector_pop() {
  Essentials::Vector<int> v;
  v.pushHead(5); v.pushHead(6);
  ASSERT_EQ_INT("pop last", 6, v.popHead());
  return true;
}

bool test_vector_pop_front() {
  Essentials::Vector<int> v;
  v.pushHead(5); v.pushHead(6);
  ASSERT_EQ_INT("pop front", 5, v.popBack());
  return true;
}

bool test_vector_pop_at() {
  Essentials::Vector<int> v;
  v.pushHead(1); v.pushHead(2); v.pushHead(3);
  ASSERT_EQ_INT("pop_at middle", 2, v.popAt(1));
  return true;
}

// ===== ERASE & CLEAR =====
bool test_vector_erase() {
  Essentials::Vector<int> v;
  v.pushHead(1); v.pushHead(2); v.pushHead(3);
  v.erase(1);
  ASSERT_EQ_INT("erase shifts", 3, v.at(1));
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
  ASSERT_EQ_INT("last", 33, v.last());
  ASSERT_EQ_INT("first", 11, v.first());
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
  ASSERT_EQ_INT("mixed first",4,v.first());
  ASSERT_EQ_INT("mixed last",3,v.last());
  ASSERT_EQ_INT("pop_front",4,v.popBack());
  ASSERT_EQ_INT("pop_back",3,v.popHead());
  return true;
}

// ===== LARGE DATASET TESTS =====
bool test_vector_large_push() {
  Essentials::Vector<int> v;
  int N=1000;
  for(int i=0;i<N;i++) v.pushHead(i);
  ASSERT_EQ_SIZE("large push size",N,v.getSize());
  ASSERT_EQ_INT("large push last",N-1,v.last());
  ASSERT_EQ_INT("large push first",0,v.first());
  return true;
}

bool test_vector_large_push_front() {
  Essentials::Vector<int> v;
  int N=1000;
  for(int i=0;i<N;i++) v.pushBack(i);
  ASSERT_EQ_SIZE("large push_front size",N,v.getSize());
  ASSERT_EQ_INT("large push_front first",N-1,v.first());
  ASSERT_EQ_INT("large push_front last",0,v.last());
  return true;
}

// ===== RUN ALL =====
bool Vector() {
  int total=0, passed=0;
#define RUN(f) total++; if(f()) passed++;

  RUN(test_vector_init);
  RUN(test_vector_resize);
  RUN(test_vector_reserve);
  RUN(test_vector_set_capacity);
  RUN(test_vector_shrink);
  RUN(test_vector_push);
  RUN(test_vector_push_front);
  RUN(test_vector_push_at);
  RUN(test_vector_pop);
  RUN(test_vector_pop_front);
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
    fmt::print(fg(fmt::color::blue_violet), "\n=== Essentials::Vector Test Score: {:d}\\{:d} passed ===\n", passed, total);
  else 
    fmt::print(fg(fmt::color::dark_orange), "\n=== Essentials::Vector Test Score: {:d}\\{:d} passed ===\n", passed, total);
  return passed == total;
};
};