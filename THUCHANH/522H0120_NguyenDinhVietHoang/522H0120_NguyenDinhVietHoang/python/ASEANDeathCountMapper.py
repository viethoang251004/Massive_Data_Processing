#!/usr/bin/env python

import sys

# Read input from standard input
for line in sys.stdin:
    # Remove leading and trailing whitespace
    line = line.strip()

    # Split the line into columns
    columns = line.split('\t')

    # Check if the country is in ASEAN (South-East Asia)
    if columns[1] == "South-East Asia":
        country = columns[0]
        deaths = columns[7]

        # Emit the country and death count as key-value pairs
        print(f"{country}\t{deaths}")
