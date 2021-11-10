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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

void print_binary(int amount, int num)
{
  assert(amount <= 32);
  for (int i = amount-1; i >= 0; i--)
    printf("%d", (num & (1<<i)) ? 1 : 0);
}

int main()
{
  srand(time(NULL));
  for (int i = 0; i < 10000; i++){
    unsigned data_1 = rand();
    unsigned data_2 = rand();
    print_binary(32, data_1);
    printf(" ");
    print_binary(32, data_2);
    printf(" %d %d %d\n", data_1 == data_2, data_1 < data_2, data_1 > data_2);
  }
}
