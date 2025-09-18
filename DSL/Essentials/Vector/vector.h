/*
 * MyProgram - short description
 * Copyright (C) 2025 Your Name
 *
 * This file is part of MyProgram.
 *
 * MyProgram is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MyProgram is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */



///#################### memory #####################///
///[ | | | | | | | | | | | | | | | | | | | | | | | ]///
/// ^ head                                   back ^ ///




#pragma once
#include <stdlib.h>
#include <stdexcept>
#include <string.h>
#include "Math/ModuloZ/ModuloZ.h"

//namespace Essential
namespace Essentials {
// VectorClass
template<typename T>
class Vector {
private:
  T* data = new T[2];
  unsigned int capacity = 2;
  unsigned int size = 0;

protected:
  Math::ModuloZ _head = Math::ModuloZ(2, 1);
  Math::ModuloZ _back = Math::ModuloZ(2, 0);

private:
  constexpr bool inRange(unsigned int index) const noexcept;

public:
  // Constructor/Destructor
  ~Vector() noexcept;    // O(1)

  // Vector management
  void setCapacity(unsigned int new_capacity);    // O(log2(n))
  void reorder();    // O(log2(n))
  void reserve(unsigned int additional);    // O(log2(n))
  void resize();    // O(log2(n))
  void shrink();    // O(log2(n))

  // Get Info
  constexpr unsigned int getSize() const noexcept;    // O(1)
  constexpr unsigned int getCapacity() const noexcept;    // O(1)
  constexpr bool isEmpty() const noexcept;    // O(1)

  // Pushers
  void pushHead(T el);    // O(log2(n))
  void pushBack(T el);    // O(log2(n))
  void pushAt(unsigned int index, T el);    // O(log2(n))

  // Popers
  const T popHead();    // O(1)
  const T popBack();    // O(1)
  const T popAt(unsigned int index);    // O(n)

  // Removers
  void clear() noexcept;    // O(1)
  void erase(unsigned int index) noexcept;    // O(n)

  // Get Without Removing       
  T& back() const;    // O(1)
  T& head() const;    // O(1)
  T& at(unsigned int index) const;    // O(1)

  // operators
};
};


#include "vector.hpp"