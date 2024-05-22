# Linum

## Train

Place the training data in `data` folder and run:

`python train.py --bs 80 --lr 0.0001 --epochs 25 --log_iter 20 --data_dir data --log`

This uses wnadb for logging

## Validation

Trained model is saved as `unet.pth`. Place the validation data in `data` folder and run:

Run `python run_validation --model_checkpoint unet.pth`