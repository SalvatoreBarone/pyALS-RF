"""
Copyright 2021 Salvatore Barone <salvatore.barone@unina.it>

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
import sqlite3
from pyosys import libyosys as ys
from .ALSSMT import *

class ALSCatalog:
  def __init__(self, file_name):
    self.__file_name = file_name
    self.__connection = None
    try:
      self.__connection = sqlite3.connect(self.__file_name)
      self.__cursor = self.__connection.cursor()
      self.__init_db()
    except sqlite3.Error as e:
      print(e)
      exit()

  """
  @brief Catalog generation procedure

  @details
  Starting from the exact specification of each unique LUT in the considered circuit, we progressively increase the
  Hamming distance between the function being implemented by original LUT (cut) and the approximate one, while 
  performing Exact Synthesis. 
  
  The procedure stops when, due to the approximation itself, the synthesis becomes trivial, i.e. it results in a catalog
  entry of size zero.
  @returns An appropriate set of catolog entries, as a list of list. The catalog is structured as follows:
   - Each element of the returned list is a list containing catalog entries for a given LUT specification
   - Each entry of the 2nd-level list is a function specification at a determined Hamming distance from the original
     non-approximate specification i.e. the element position within the list gives the Hamming distance from the 
     original specification; therefore, elements in position [0] represent non-approximate function specification.
  Example:
  [
    # LUT specs
    [
      {"spec": function specification (string), "gates" : AND-gates required to synthesize the spec (integer)}, <-- non-approx. specification
      {"spec": function specification (string), "gates" : AND-gates required to synthesize the spec (integer)}, <-- approx-spec at distance 1
      {"spec": function specification (string), "gates" : AND-gates required to synthesize the spec (integer)}, <-- approx-spec at distance 2
      ...
      {"spec": function specification (string), "gates" : AND-gates required to synthesize the spec (integer)}  <-- approx-spec at distance N
    ],
    ...
  ]

  @note This class implements LUT caching, so the actual synthesis of a LUT is performed i.f.f. the latter is not yet
  in the database.
  """
  def generate_catalog(self, design, es_timeout):
    # Building the set of unique luts
    luts_set = set()
    for module in design.selected_whole_modules_warn():
      for cell in module.selected_cells():
        if ys.IdString("\LUT") in cell.parameters:     
          luts_set.add(cell.parameters[ys.IdString("\LUT")].as_string()[::-1])

    # TODO: This for loop should be partitioned among multiple threads. 
    catalog = []
    for lut in luts_set:
      lut_specifications = []
      # Sinthesizing the baseline (non-approximate) LUT
      hamming_distance = 0
      synt_spec, S, P, out_p, out = self.get_synthesized_lut(lut, hamming_distance, es_timeout)
      gates = len(S[0])
      lut_specifications.append({"spec": synt_spec, "gates": gates, "S": S, "P": P, "out_p": out_p, "out": out})
      #  and, then, approximate ones
      while gates > 0:
        hamming_distance += 1
        synt_spec, S, P, out_p, out = self.get_synthesized_lut(lut, hamming_distance, es_timeout)
        gates = len(S[0])
        lut_specifications.append({"spec": synt_spec, "gates": gates, "S": S, "P": P, "out_p": out_p, "out": out})
      catalog.append(lut_specifications)
      # Speculation...
      for i in range(1, len(lut_specifications)):
        self.__add_lut(lut_specifications[i]["spec"], 0, lut_specifications[i]["spec"], lut_specifications[i]["S"], lut_specifications[i]["P"], lut_specifications[i]["out_p"], lut_specifications[i]["out"])
        for j in range(i+1, len(lut_specifications)):
          self.__add_lut(lut_specifications[i]["spec"], j-i, lut_specifications[j]["spec"], lut_specifications[j]["S"], lut_specifications[j]["P"], lut_specifications[j]["out_p"], lut_specifications[j]["out"])
    return catalog

  """
  @brief Queries the database for a particular lut specification. 

  @param [in] lut
              exact specification of the lut; combined with distance makes up the actual specification of the 
              synthesized LUT to be searched.

  @param [in] distance
              Hamming distance of the LUT to be searched against the exact specification in lut; combined with the 
              latter makes up the actual specification of the sy thesized to be searched.

  @details 
  If the lut exists, it is returned, otherwise the function performs the exact synthesis of the lut and adds it
  to the catalog before returning it to the caller.
  
  @return If the lut exists, it is returned, otherwise the function performs the exact synthesis of the lut and adds it
  to the catalog before returning it to the caller.
  """
  def get_synthesized_lut(self, lut_spec, dist, es_timeout):
    result = self.__get_lut_at_dist(lut_spec, dist)
    if result is None:
      ys.log(f"Cache miss for {lut_spec}@{dist}\n")
      ys.log(f"Performing SMT-ES for {lut_spec}@{dist}\n")
      synth_spec, S, P, out_p, out = ALSSMT(lut_spec, dist, es_timeout).synthesize()
      gates = len(S[0])
      print(f"Done! {lut_spec}@{dist} Satisfied using {gates} gates. Synth. spec.: {synth_spec}")
      self.__add_lut(lut_spec, dist, synth_spec, S, P, out_p, out)
      return synth_spec, S, P, out_p, out
    else:
      synth_spec = result[0]
      gates = len(result[1][0])
      print(f"Cache hit for {lut_spec}@{dist}, which is implemented as {synth_spec} using {gates} gates")
      return result[0], result[1], result[2], result[3], result[4]

  """ 
  @brief Inits the database
  """
  def __init_db(self):
    self.__cursor.execute("create table if not exists luts (spec text not null, distance integer not null, synth_spec text, S text, P text, out_p integer, out integer, primary key (spec, distance))")
    self.__connection.commit()
  
  """
  @brief Queries the database for a particular lut specification. 
  """
  def __get_lut_at_dist(self, spec, dist):
    self.__cursor.execute(f"select synth_spec, S, P, out_p, out from luts where spec = '{spec}' and distance = {dist};")
    result = self.__cursor.fetchone()
    if result is not None:
      return result[0], string_to_nested_list_int(result[1]), string_to_nested_list_int(result[2]), result[3], result[4]
    return None

  """
  @brief Insert a synthesized LUT into the database
  """
  def __add_lut(self, spec, dist, synth_spec, S, P, out_p, out):
    self.__cursor.execute(f"insert or ignore into luts (spec, distance, synth_spec, S, P, out_p, out) values ('{spec}', {dist}, '{synth_spec}', '{S}', '{P}', {out_p}, {out});")
    self.__connection.commit()


def string_to_nested_list_int(s):
  if s == '[[], []]':
    return [[], []]
  l = [sl.strip('[]').split(',') for sl in s.split('], [')]
  return [[int(i) for i in l[0]], [int(i) for i in l[1]]]