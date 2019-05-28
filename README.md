# lire-color-descriptors
Extract Lire color descriptors from csv file(s) with RGB pixels.

#### Usage: 
```
       [-h] -i INPUT [INPUT ...] [-o [OUTPUT]]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        input csv file(s)
  -o [OUTPUT], --output [OUTPUT]
                        output file name. Input file name with
                        '.LireColorDescriptors.csv' extension by default
```

#### Example:
`./example.py -i RGB_LINEAR_RGB_class1_class2_class3_class4_exp2_cc_1.tsv RGB_LINEAR_RGB_LINEAR_class1_cc_ctrl_1.tsv -o result.csv`
