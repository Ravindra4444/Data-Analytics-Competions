import pandas as pd
import numpy as np
import csv



input_file = "MF_IIT_DATA_V3.csv"
entry_file = "partB_data.csv"

number = []

with open(entry_file, 'wb') as f:
	writer = csv.writer(f)
	with open(input_file, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			print(row[29])
			number[12] = row[29]
			writer.writerow(number)

f.close()
csvfile.close()
