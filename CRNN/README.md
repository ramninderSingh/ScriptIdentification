
## Environment Setup
Activate your conda or virtual Environment with python=3.12 installed and do the following:
```bash
pip install -r requirements.txt
```


## Training

In the config.py's common config pass the number of hidden units in the fully connected layer that maps the CNN output to the sequence input for the RNN(map_to_seq_hidden) and pass the number of hidden units in each LSTM layer of your RNN (rnn_hidden).

Also in train_config dictinory keep the dataset which contains images along with specifying the number of images to take from each folder and the number of classes.Also specify the directory where the trained model with graphs will be stored.

To start the training 

```bash
python train_crnn.py
```


## Testing 
Similar to above in the config.py file under test_config pass the images folder path and number of classes. Here you also need to pass the model path

To test
```bash
python test_crnn.py
```

## Inference
The model is assumed to be trained on 2 or 3 classes in which the first two classes are to be Hindi and English and 3rd if present can be any one of the remaining.

In the config.py file for the infer_config dictionary pass the trained model path(CRNN) and img_path.Also update the map_to_seq_hidden and rnn_hidden as defined above based on the trained model.Now:

```bash
python infer.py
```
### Output:
| Prediction    | Class         |
| ------------- | ------------- |
| 0 | Hindi  |
| 1 | English|
| 2 | Other Language|




## API's
Start the localhost FastAPI server:
uvicorn app:app --reload

To get the prediction for image using the CRNN model

- METHOD : POST
 
- URL
```bash
"http://127.0.0.1:8000/predict_image/{model_key}"

```
- Parameters: file: image_file to be classified

Where model key is as follows to get the model for testing on image:
- hep: for hindi english and punjabi
- he: for hindi english 
- heg: for hindi english and gujarati

