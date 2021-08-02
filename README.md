# Serving_Using_BentoML
This project helps in serving simple model using BentoML <br />
 <br />
### Requirements: <br />
bentoml==0.13.1 <br />
imageio==2.8.0 <br />
torch==1.9.0+cu102 <br />
torchvision==0.10.0+cu102 <br />
 <br />
## Please change all the paths in code files and follow below steps: <br />
Step-1 : Run the "loading_datasets.py" file, it'll dowload the required datasets (mnist dataset) <br />
Step-2 : Run the "mnist.py" file, used in packaging the model. <br />
Step-3 : Run the "utils1.py" file, it'll create the save_model and load_model functions. <br />
Step-4 : Run the "trained_model.py" file, it'll create, train a simple model and save the checkpoint file. <br />
Step-5 : Run the "saveToBento.py" file, it'll create functions needed to create the bento file. <br />
Step-6 : Run the "final.py" file, it'll create the bento file in loaction "saved-path" using the saveToBento module we've created, please copy the path location. <br />
 <br />
Step-7 : Run the following code in command prompt or shell: <br />
 <br />
"bentoml serve saved-path" <br />
 <br />
saved-path in above command is printed when you follow step-6. <br />
