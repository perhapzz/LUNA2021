RESOURCES_PATH = './data/input'
# some directory for output the results:
OUTPUT_PATH = './data/output'

# Resource path which contains: annotations.csv, candidates.csv,
# and subdirectories containing .mhd files.
# This is the directory structure needed to run the code:
# (The code will use all .mhd and .raw files inside subdirectories which their name is in annotations or candidates)

'''
[RESOURCES_PATH]/
    my_custom_subset/
        *.mhd
        *.raw
'''
