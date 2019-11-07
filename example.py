import pandas as pd
from Seeds2Features.Exstract import Tsv2Df
from Seeds2Features.Preprocessing import RenameClasses
import os
import argparse

parser = argparse.ArgumentParser("Extract Lire color descriptors from csv file(s) with RGB pixels")
parser.add_argument('-i', '--input', required=True, nargs='+', help="input csv file(s)")
parser.add_argument('-o', '--output', nargs='?',
                    help="output file name. Input file name with '.LireColorDescriptors.csv' extension by default")

args = parser.parse_args()

input_files = args.input
if not all([os.path.isfile(f) for f in input_files]):
    print("Error: input file(s) not found")
    exit(1)

output_file = args.output
if output_file is None:
    output_file = os.path.splitext(input_files[0])[0] + '.LireColorDescriptors.csv'

result = pd.DataFrame()
print(input_files, output_file)
for input_file in input_files:
    df = RenameClasses(input_file, Tsv2Df(input_file).extract()).rename_from_dict()
    result = pd.concat([result, df])
result.to_csv(output_file)
