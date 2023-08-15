# Positive and Active Gradients to boost Saliency Maps

This is the official implementation of [A matter of attitude: Focusing on positive and active gradients to boost saliency maps]():

```
```



## How to run the experiments

All the code is inside the src folder. Inside the src folder, everything is divided into two folders:

-  train: This is the code for training the same models that have been used for the paper. There is a main file for running the training with the main function. Inside this function the user can choose between the different types of datasets (imagenette or cifar10), models (cnn resnet18 or convnext) and the hyperparameters (pretrained, lr and number of epochs). The datasets will be saved in a folder called "data" that will be created the first time you run a training. The models will be saved in another folder called models that will also be created the first time a training is run. The name of each model will be a combination of the dataset name, the model type and if it has been pretrained. The other hyperparameters are not used since we have used 50 epochs and 1e-3 of learning rate for all the models. For running the main file the following command could be used:

```
python -m src.train.main
```


- explain: This is the code for running the explainability methods and the deletion metrics. There is a main file with a main function. Inside this you will find two boolean variables, generate_examples and generate_graphs. The first one is to indicate if visualizations will be generated. The second one to indicate if the metrics visualizations will be generated. Everything will be saved in a new folder called "visualizations" that will be created the first time the explain main is run. The first time the graphs are generated the computation will be slower since ewe have to generate the classification results for each percentage. This data will be saved inside the data folder, in a folder called "saved" inside each dataset folder. The explainability techniques are all inside the saliency_maps file. For running the main file the following command could be used:

```
python -m src.explain.main
```

## Dependencies
Are listed in the requirements file. For installing them just run:

```
pip install -r requirements.txt
```