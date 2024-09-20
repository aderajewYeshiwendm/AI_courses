from functools import cmp_to_key


def cmp(a, b):
    if a < b:
        return -1
    else:
        return 1
    
print(sorted([1,4,2,5,6,3], key = cmp_to_key(cmp)))
        