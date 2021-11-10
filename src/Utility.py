import numpy as np

def list_partitioning(a_list, num_of_partitions):
  list_of_list = []
  np_split = np.array_split(a_list, num_of_partitions)
  for item in np_split:
    list_of_list.append(list(item))
  return list_of_list