# Serving_Using_BentoML
This project helps in serving simple model using BentoML

Requirements:
bentoml==0.13.1
imageio==2.8.0
torch==1.9.0+cu102
torchvision==0.10.0+cu102

# Change the paths in code:
Step-1 : Run the "loading_datasets.py" file, it'll dowload the required datasets (mnist dataset)
Step-2 : Run the "mnist.py" file, used in packaging the model.
Step-3 : Run the "utils1.py" file, it'll create the save_model and load_model functions.
Step-4 : Run the "trained_model.py" file, it'll create, train a simple model and save the checkpoint file.
Step-5 : Run the "saveToBento.py" file, it'll create functions needed to create the bento file.
Step-6 : Run the "final.py" file, it'll create the bento file in loaction "saved-path" using the saveToBento module we've created.

Step-7 : Run the following code in command prompt or shell:

"bentoml serve saved-path"

saved-path in above command is printed when you follow step-6.
