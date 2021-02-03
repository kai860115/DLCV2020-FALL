wget https://www.dropbox.com/s/cu6cp1wttor999s/hw2_1_model.pkl?dl=1  -O hw2_1_model.pkl
python3 image_classification/test.py --img_dir $1 --save_dir $2 --ckp_path hw2_1_model.pkl 