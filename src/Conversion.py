import struct

def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def double_to_hex(d):
  return hex(struct.unpack('<Q', struct.pack('<d', d))[0])

def float_to_bin(f):
  return format(struct.unpack('<I', struct.pack('<f', f))[0], "032b")

def double_to_bin(d):
  return format(struct.unpack('<Q', struct.pack('<d', d))[0], "064b")

def bin_to_double(b):
  return struct.unpack('d', struct.pack('Q', int(b, 2)))[0]

def bin_to_float(b):
  return struct.unpack('f', struct.pack('I', int(b, 2)))[0]

def apply_mask_to_double(f, nab):
  return bin_to_double(double_to_bin(f)[:-nab] + "0" * nab)

def apply_mask_to_float(f, nab):
  return bin_to_float(float_to_bin(f)[:-nab] + "0" * nab)

def apply_mask_to_int(i, nab):
  return i & (~((1<<nab)-1))

