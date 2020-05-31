import indexed_dataset
import os
import sys

#input_folder = "pretrain_data/data/"
#output_folder = "pretrain_data/merge/"
#output_bin_file = "merge.bin"
#output_idx_file = "merge.idx"

def get_merge_file(input_folder, output_folder, output_bin_file, output_idx_file):
    builder = indexed_dataset.IndexedDatasetBuilder(output_folder + output_bin_file)
    for filename in os.listdir(input_folder):
        if filename[-4:] == '.bin':
            builder.merge_file_(input_folder + filename[:-4])
    builder.finalize(output_folder + output_idx_file)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print ("Usage: python3 merge.py input_folder output_folder output_bin_file output_idx_file")
    input_folder = sys.argv[1] + "/" 
    output_folder = sys.argv[2] + "/"
    output_bin_file = sys.argv[3]
    output_idx_file = sys.argv[4]
    
    if not os.path.exists(input_folder):
        logging.error("input_folder doesn't exist. input_folder=%s pwd=%s" % (input_folder, os.path.abspath(__file__)))
        exit(0)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    get_merge_file(input_folder, output_folder, output_bin_file, output_idx_file)
    
    