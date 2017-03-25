#!/usr/bin/python
import random
with open("edges_after_reduce.txt", 'r') as f:
    with open("edges_with_prob.txt", 'w') as fout:
        for line in f:
            if line[0] is '#':
                continue
            line = line.strip()
            words = line.split(' ')
            r = random.randint(1, 5)/100.0
            fout.write(words[0] + ' ' + words[1] + ' ' + str(r) + '\n')
