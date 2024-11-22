# Script Identification
Script Identification is designed to identify the script (language) of text within images from natural scenes. This repository includes the inference code and models needed for prediction. Each model is structured as a triplet, with Hindi and English as common languages alongside a third language listed below. 

## Installation
Create a conda environment and install the dependencies.
```
conda create -n scriptdetect python=3.9 -y
conda activate scriptdetect

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

cd clip
pip install -r requirements.txt
```

## Inference
To get started, clone the repository and make "models" direcotory so that model gets downloaded as we inference script. The models can be downloaded directly from github [assets](https://github.com/anikde/STscriptdetect/releases/tag/V1).
```
git clone https://github.com/anikde/STscriptdetect.git
cd STscriptdetect
mkdir models
```

Script detection can be done using ```infer.py``` on a single image as input.

```python
python infer.py --image_path demo_images/D_image_149_9.jpg --model_name odia
# {'predicted_class': 'odia'}
```
Simply replace ```demo_images/D_image_149_9.jpg``` with your image path and ```odia``` with the model name for desired language detection.

To process a batch of images
```python
python infer.py --image_dir demo_images/ --model_name odia --batch
# predictions.csv
```
using the ```image_dir``` and ```--batch``` argument predictions can be made on a batch of images.


## Evaluation

```python
python evaluation.py <path/to/test.csv> <path/to/predictions.csv>
``` 

## Supported Languages
Each model includes Hindi and English, with the third language varying as follows:

- hindi: Hindi, English
- assamese: Hindi, English, Assamese
- bengali: Hindi, English, Bengali
- gujarati: Hindi, English, Gujarati
- kannada: Hindi, English, Kannada
- malayalam: Hindi, English, Malayalam
- marathi: Hindi, English, Marathi
- meitei: Hindi, English, Meitei
- odia: Hindi, English, Odia
- punjabi: Hindi, English, Punjabi
- tamil: Hindi, English, Tamil
- telugu: Hindi, English, Telugu
- urdu: Hindi, English, Urdu
