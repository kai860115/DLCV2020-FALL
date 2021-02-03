# TODO: create shell script for running your VAE model

# Example
wget https://www.dropbox.com/s/qb9k2n3bqvelie7/vae.pth?dl=1  -O vae.pth
python3 ./vae/test.py --save_path $1 --ckp_path ./vae.pth
