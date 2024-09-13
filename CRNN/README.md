
## Environment Setup
Activate your conda or virtual Environment with python=3.12 installed and do the following:
```bash
pip install -r requirements.txt
```


## Training
In the config.py's train_config dictinory keep the dataset which contains images along with specifying the number of images to take from each folder and the number of classes.Also specify the directory where the trained model with graphs will be stored.

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

