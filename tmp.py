from io import open
import sys

file_path = '/Users/wumengling/PycharmProjects/ward_heapq/test'
f = open(file_path)
i = 0
total_type = 0
for line in f.readlines():
    print(type(line))
    print(bytes(line.encode('ascii', errors='replace')))
    print(line.encode('ascii', errors='replace'))
    print(line.encode('utf-8', errors='replace'))
    total_type += sys.getsizeof(line)
    i += 1

print(i)
print(total_type)
f.close()
