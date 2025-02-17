
# Autocomplete Using LSTM

Here we will be creating a Bidirectional LSTM Neural Network and training it on some pre processed data to implement an autcomplete/next word prediction feature.


## Dataset

I have used holmes.txt dataset which I got from kaggle.

To import the dataset here is what you do:
```bash
  pip install opendatasets
```

after installing that in command line:
```bash
  import opendatasets as od
```

```bash
  od.download("kaggle.com/datasets/noorsaeed/holmes")
```
```bash
input_file = 'path to downloaded file'
with open(input_file, 'r', encoding='utf-8') as infile:
    data = infile.read()
```

Refer to the rest of the .py files for the rest of the information.
