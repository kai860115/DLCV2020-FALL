# TODO: create shell script for running your GAN model

# Example
wget https://www.dropbox.com/s/lekb6dm55q2istf/G.pth?dl=1 -O G.pth
python3 ./gan/test.py --save_path $1 --ckp_path ./G.pth
