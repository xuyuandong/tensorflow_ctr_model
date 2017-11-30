#coding=utf-8
import sys

for line in sys.stdin:
    ts = line.strip().split('\t')
    if len(ts) == 2:
        print ts[1]
