wget https://www.dropbox.com/s/g8o27nd3lw51ewp/hw2_2_best.pkl?dl=1 -O hw2_2_best.pkl
python3 semantic_segmentation/test.py --img_dir $1 --save_dir $2 --ckp_path hw2_2_best.pkl --model_type FCN8s