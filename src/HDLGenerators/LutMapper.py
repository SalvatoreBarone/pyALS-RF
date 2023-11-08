"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

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
from itertools import islice 
from typing import List

class LutMapper:
    def __init__(self, k : int = 6):
        self.k = k
        self.and_conf = 'X\"8' + '0' * (2**(self.k-2)-1) + "\""
        self.or_conf = 'X\"' + 'F' * (2**(self.k-2)-1) + "E\""

    def map(self, minterms : List[str], signal_name : str):
        and_luts = {}
        or_luts = {}
        stack = []
        to_be_ored = []
        
        for m in minterms:
            inputs = m.strip("()").replace("not ", "not_").split(" and ")
            inputs.sort()
            stack.append(inputs)
            
        while len(stack) > 0:
            m = stack.pop()
            print(f"Processing minterm {m}")
            slices = LutMapper.split_minterm(m, self.k)
            and_of_and = []
            for s in slices:
                if 1 == len(s) and s[0] not in and_of_and:
                    and_of_and.append(s[0])
                    continue
                elif len(s) < self.k:
                    inputs = tuple(s + ["'1'"] * (self.k - len(s)))
                else:
                    inputs = tuple(s)
                if inputs not in and_luts:
                    and_luts[inputs] = {"inst" : f"and{self.k}_{signal_name}_inst_{len(and_luts)}", "conf": self.and_conf, "o": f"and{self.k}_{signal_name}_inst_{len(and_luts)}_o"}
                    print(f"\tInstantiating {and_luts[inputs]['inst']} with inputs {inputs}")
                and_of_and.append(and_luts[inputs]["o"])
            if len(and_of_and) > 1:
                stack.append(and_of_and)
            else:
                to_be_ored.append(and_of_and[0])
                
        if len(to_be_ored) > 1:
            stack.append(to_be_ored)
            while len(stack) > 0:
                m = stack.pop()
                print(f"Processing maxterm {m}")
                slices = LutMapper.split_minterm(m, self.k)
                if len(slices) > 1:
                    or_of_or = []
                    for s in slices:
                        if 1 == len(s) and s[0] not in or_of_or:
                            or_of_or.append(s[0])
                            continue
                        elif len(s) < self.k:
                            inputs = tuple(s + ["'0'"] * (self.k - len(s)))
                        else:
                            inputs = tuple(s)
                        if inputs not in or_luts:
                            or_luts[inputs] = {"inst" : f"or{self.k}_{signal_name}_inst_{len(or_luts)}", "conf": self.or_conf, "o": f"or{self.k}_{signal_name}_inst_{len(or_luts)}_o"}
                            print(f"\tInstantiating {or_luts[inputs]['inst']} with inputs {inputs}")
                        or_of_or.append(or_luts[inputs]['o'])
                    stack.append(or_of_or)
                else:
                    if len(slices[0]) < self.k:
                        inputs = tuple(slices[0] + ["'0'"] * (self.k - len(slices[0])))
                    else:
                        inputs = tuple(slices[0])
                    or_luts[inputs] = {"inst" : f"or{self.k}_{signal_name}_inst_{len(or_luts)}", "conf": self.or_conf, "o": f"class_{signal_name}"}
                    print(f"\tInstantiating {or_luts[inputs]['inst']} with inputs {inputs}")
        else:
            and_luts[ list(and_luts.keys())[-1] ]['o'] = f"class_{signal_name}"    
        return [ {"inst" : lut['inst'], "type": f"LUT{self.k}", "conf" : lut['conf'], 'o' : lut['o'], "pi": pis} for pis, lut in {**and_luts, **or_luts}.items() ]

        
    @staticmethod
    def split_minterm(literals : list, lut_tech : int):
        slice_len = [lut_tech] * (len(literals) // lut_tech)
        reminder = len(literals) % lut_tech
        if reminder > 0:
            slice_len += [reminder]
        iterator = iter(literals)
        slices = [list(islice(iterator, elem)) for elem in slice_len]
        return slices