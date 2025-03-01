#!/usr/bin/env python

import sys

current_country = None
total_deaths = 0

# Read input from standard input
for line in sys.stdin:
    # Remove leading and trailing whitespace
    line = line.strip()

    # Split the input into country and death count
    country, deaths = line.split('\t')

    # Convert the death count to an integer
    deaths = int(deaths)

    # If the country is the same as the previous one, increment the death count
    if country == current_country:
        total_deaths += deaths
    else:
        # If the country is different, emit the previous country and its total death count
        if current_country:
            print(f"{current_country}\t{total_deaths}")

        # Update the current country and total death count
        current_country = country
        total_deaths = deaths

# Emit the last country and its total death count
if current_country:
    print(f"{current_country}\t{total_deaths}")