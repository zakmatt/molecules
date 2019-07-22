# Toxicity prediction

Run models training with  
```python
python train.py [-d] [-e]
```
`-d` and `-e` are parameters specifying dataset path and number of epochs. 
By default the `-d` is set to `./data/data.csv` and `-e` to 50.

Evaluate models training with  
```python
python train.py [-m] [-f] [-t]
```
`-m`, `-f` and `-t` are parameters specifying model type (DenseNet or ConvNet), 
input features type (morgan - morgan fingerprint binary representation, vect - 
using bag of words) and training dataset path. 
By default the `-m` is set to `DenseNet`, `-f` to `morgan` and `-t` to 
`./data/test_set.csv`.