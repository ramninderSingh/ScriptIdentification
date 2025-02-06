## Environment Setup
Activate your conda or virtual Environment with python=3.12 installed and do the following:
```bash
pip install -r requirements.txt
```



### Vit Model's folder trained on BSTD 12 classes can be downloaded from:
    "https://drive.google.com/file/d/1u66s0G60j2tKRUYKiH05XQRmp1YCcwjW/view?usp=sharing"

Following are the languages it is trained on: Hindi, English, Gujarati, Punjabi, Marathi, Kannada, Telugu, Tamil, Malayalam, Assamese, Bengali, and Odia.


## Training
In config.py, specify the desired pretrained ViT model by setting the pretrained_vit_model key within the common_config dictionary.
Further in train_config dictionary pass the dataset which contains images along with specifying the maximum number of images to take from each folder and the number of classes.Also specify the directory where the trained model with graphs will be stored.

To start the training 

```bash
python train.py
```
After training, two folders will be created: one containing all checkpoints and another with the best model saved during the entire training process. For testing, the path to the best model folder will be used."

## Testing 
In config.py, specify the desired pretrained ViT model by setting the pretrained_vit_model key within the common_config dictionary.
Similar to above in the config.py file under test_config pass the images folder path and number of classes. Here you also need to pass the best_model's folder which is to be tested

To test
```bash
python test_crnn.py
```

The code will output class-wise accuracy along with macro precision, recall, and F1-score. Additionally, a confusion matrix will be saved in the best_model's folder

## Inference
In the config.py file for the infer_config dictionary pass the trained model folder_path and img_path.Now to get prediction:

```bash
python infer.py
```


## API
Start the localhost FastAPI server:
uvicorn app:app --reload


To get the prediction for image using the ViT model

- METHOD : POST

- URL
```bash
"http://127.0.0.1:8000/predict_lang/"

```
- Parameters: file: image_file to be classified

The model will predict the language.