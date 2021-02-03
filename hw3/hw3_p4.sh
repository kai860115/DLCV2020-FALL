# TODO: create shell script for running your improved UDA model

# Example
if [ "$2" = "mnistm" ]
then
    wget https://www.dropbox.com/s/oquikuq2bg8qllp/u2m-dsn.pth?dl=1 -O u2m-dsn.pth
    python3 ./dsn/test.py --img_dir $1 --save_path $3 --ckp_path ./u2m-dsn.pth
elif [ "$2" = "usps" ]
then
    wget https://www.dropbox.com/s/7c3ut7v23xob4ey/s2u-dsn.pth?dl=1 -O s2u-dsn.pth
    python3 ./dsn/test.py --img_dir $1 --save_path $3 --ckp_path ./s2u-dsn.pth
else
    wget https://www.dropbox.com/s/6ld7kukxx8qoqia/m2s-dsn.pth?dl=1 -O m2s-dsn.pth
    python3 ./dsn/test.py --img_dir $1 --save_path $3 --ckp_path ./m2s-dsn.pth
fi