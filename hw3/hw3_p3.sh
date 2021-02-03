# TODO: create shell script for running your DANN model

# Example
if [ "$2" = "mnistm" ]
then
    wget https://www.dropbox.com/s/8ahf0cz6e3gjpgo/u2m-dann.pth?dl=1 -O u2m-dann.pth
    python3 ./dann/test.py --img_dir $1 --save_path $3 --ckp_path ./u2m-dann.pth
elif [ "$2" = "usps" ]
then
    wget https://www.dropbox.com/s/7k0pk05o6z01xxz/s2u-dann.pth?dl=1 -O s2u-dann.pth
    python3 ./dann/test.py --img_dir $1 --save_path $3 --ckp_path ./s2u-dann.pth
else
    wget https://www.dropbox.com/s/sntpys0tgmr80dc/m2s-dann.pth?dl=1 -O m2s-dann.pth
    python3 ./dann/test.py --img_dir $1 --save_path $3 --ckp_path ./m2s-dann.pth
fi