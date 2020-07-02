#!/bin/bash
# Ask the user for file name

python data/preprocess.py --root_path '/home/incentive/Music/Final Checking' --manifest_file_path 'data/list_attr_celeba.txt' ;;

echo "write the file name you want to run (example: train/test/eval)?"
read filename
#Pass the variable in string 
case "$filename" in 
    #case 1 
    "train") python train.py --root_path '/home/incentive/Music/Final Checking' --image_container_path "data/CelebA" --train_manifest_file_path 'data/train_manifest.txt' --validation_manifest_file_path 'data/valid_manifest.txt' ;; 
      
    #case 2 
    "test") python test.py --root_path '/home/incentive/Music/Final Checking' --weights_path 'src/weights/weights.149.pth' ;; 
      
    #case 3 
    "eval") python inference.py --root_path '/home/incentive/Music/Final Checking' --image_folder_name 'custom' --image_fname 'Akbar_jm.jpg' --weights_name 'src/weights/weights.149.pth' --input_img_attr 1 0 0 0 1 1 0 1 1 0 1 0 0 --index 3 10 ;; 
esac