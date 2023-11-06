# DSA4266-project2

- [Virtual_environment](#virtual_environment)
- [Training](#training)
- [Prediction](#prediction)

### Virtual_environment

1. Install python3 virtual environment
```bash
sudo apt install python3.8-venv
```

2. Create and activate virtual environment
```bash
python3 -m venv venv
source ./venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```


### Training

```console
> python train.py 
usage: train.py [--epochs E] [--batch_size B] [--data D] [--test_data T]
                
Train the model

optional arguments:
  --epochs E      Number of epochs default is 32
  --batch-size B  Batch size default is 200
  --data D        Training data directiory in json format default is ./data/train_OG.csv
  --test_data T   Test data directory in json format default is ./data/test_OG.csv
```


### Prediction
```console
> python detect_json.py 
usage: detect_json.py [--model M] [--data D]
                
Train the model

optional arguments:
  --model M     model directory in h5 format
  --data  D     data directory in json format
```