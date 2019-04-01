#test.py
try:
  for k in range(100000000):
    for j in range(k):
      for l in range(j):
        d = l + 1
except:
  print('hmmm')
  raise

print('not even quitting though')