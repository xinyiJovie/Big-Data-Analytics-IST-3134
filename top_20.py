# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:29:46 2024

@author: HP
"""

import sys
from collections import Counter

word_counts = Counter()

# Read the input from the Hadoop output
for line in sys.stdin:
    word, count = line.strip().split('\t')
    word_counts[word] = int(count)

# Get the top 20 most common words
top_20 = word_counts.most_common(20)

# Output the results
for word, count in top_20:
    print(f'{word}\t{count}')

