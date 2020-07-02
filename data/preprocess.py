import argparse
import os


def split_manifest(root_path, manifest_file_path):
    """
    This function split the attribute manifest file of CelebA dataset into
    three files, train, test and valid file which are used 
    
    Params:
    
    -- root_path                : root path of your system as string.
    -- manifest_file_path       :  attributes file path
    
    
    Returns:
    This function returns None. But it saved the splited files in dataset
    directory.
    
    """

    train_manifest = open(os.path.join(root_path,"dataset", "train_manifest.txt"), "w+")
    test_manifest = open(os.path.join(root_path, "dataset","test_manifest.txt"), "w+")
    val_manifest = open(os.path.join(root_path,"dataset" ,"valid_manifest.txt"), "w+")
    with open(os.path.join(root_path, manifest_file_path), 'r') as f:
        data_manifest = f.read().strip().split('\n')
    data_len = len(data_manifest)
    k = 0
    for i in data_manifest:
        if k == 0:
            k = k+1
            continue
        elif k == 1:
            train_manifest.write(i+'\n')
            test_manifest.write(i+'\n')
            val_manifest.write(i+'\n')
        elif k <= data_len*0.6: # 60% on train set
            train_manifest.write(i+'\n')
        elif k > data_len*0.6 and k <= data_len*0.8: # 20 % on test
            test_manifest.write(i+'\n')
        else: #20 % on test
            val_manifest.write(i+'\n')
        k = k+1
    print("Spliting attritutes Done!")
    
def preproces_parser(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='root path type here',
                        default='/home/incentive/Desktop/FINAL Project/ojt-project')
    parser.add_argument('--manifest_file_path', type=str,
                        help='attribute test file name',
                        default='dataset/list_attr_celeba.txt')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = preproces_parser()
    print(args)
    split_manifest(args.root_path, args.manifest_file_path)
