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
#include <string.h>
#include <assert.h>

void print_binary(int amount, int num)
{
  assert(amount <= 32);
  for (int i = amount-1; i >= 0; i--)
    printf("%d", (num & (1<<i)) ? 1 : 0);
}

int main ()
{
	for (unsigned i = 0; i < (1<<16); i++)
	{
		unsigned number = i, one_count = 0;
		for (unsigned j = 0; j < 16; j++)
			one_count += 1U & (number >> j);		
    print_binary(16, number);
		printf(" %d\n", one_count >= 8);
	}
	return 0;
}