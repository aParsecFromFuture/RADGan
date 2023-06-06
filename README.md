# RADGan
Simple conditional GAN for MRI to CT image synthesis

![](results.png)

## Training

`python train.py --train_dir your_train_dir --val_dir your_val_dir --num_epochs 100`

## Evaluation

`python eval.py --test_dir your_test_dir --model_path your_model.pth`
