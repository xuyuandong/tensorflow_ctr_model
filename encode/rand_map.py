#coding=utf-8
import sys
import random

for line in sys.stdin:
    key = '%.10f'%(random.random())
    print key[2:] + '\t' + line.strip()
