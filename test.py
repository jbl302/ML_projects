import sys
if sys.base_prefix == sys.prefix:
  print("yes")
  print(sys.prefix)
  print(sys.base_prefix)
else:
  print("No")
  print(sys.prefix)
  print(sys.base_prefix)
  
  