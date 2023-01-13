"""
Copyright 2021-2022 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
class ErrorConfig:
    def __init__(self, test_dataset, max_loss_perc, max_eprob = None, nvectors = None, dataset = None):
        self.test_dataset = test_dataset
        self.max_loss_perc = max_loss_perc
        self.max_eprob = max_eprob
        self.nvectors = nvectors
        self.dataset = dataset
        