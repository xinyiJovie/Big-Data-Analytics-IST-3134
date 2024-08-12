# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:28:33 2024

@author: HP
"""

import os
import pandas as pd

# Define the directory where your files are located
directory = os.path.expanduser('~/workspace/python/reviews/')

# List of input and output files
files = [
    ('reviews_0-250.csv', 'selected_reviews_0-250.csv'),
    ('reviews_250-500.csv', 'selected_reviews_250-500.csv'),
    ('reviews_500-750.csv', 'selected_reviews_500-750.csv'),
    ('reviews_750-1250.csv', 'selected_reviews_750-1250.csv'),
    ('reviews_1250-end.csv', 'selected_reviews_1250-end.csv'),
]

# Loop over each file
for input_filename, output_filename in files:
    input_file = os.path.join(directory, input_filename)
    output_file = os.path.join(directory, output_filename)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Adjust the column names as necessary
    selected_columns = ['product_name', 'brand_name', 'cleaned_review']

    # Ensure the column names match your dataset
    df_selected = df[selected_columns]

    # Save the new DataFrame to a CSV file
    df_selected.to_csv(output_file, index=False)

    print(f"Selected columns saved to {output_file}")
