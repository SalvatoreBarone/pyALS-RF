// Copyright 2020-2021 Salvatore Barone <salvatore.barone@unina.it>
// 
// This is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 3 of the License, or any later version.
// 
// This is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
// 
// You should have received a copy of the GNU General Public License along with
// RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
// Street, Fifth Floor, Boston, MA 02110-1301, USA.
#include <iostream>
#include <bitset>
#include <limits>
#include <random>
#include <cassert>

void print_binary_64(std::ostream& stream, void * const data)
{
  unsigned long * udata = (unsigned long *)data;
  stream << std::bitset<64>(*udata);
}

int main()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib;

  for (int i = 0; i < 10000; i++){
    double data_1;
    double data_2;
    data_1 = distrib(gen);
    data_2 = distrib(gen);
    print_binary_64(std::cout, &data_1);
    std::cout << " ";
    print_binary_64(std::cout, &data_2);
    std::cout << " " << (data_1 == data_2) << " " << (data_1 < data_2) << " " << (data_1 > data_2) << std::endl;
  }
}
