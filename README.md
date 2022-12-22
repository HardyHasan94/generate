
# Generative Adversarial Networks

Implementing the original Generative Adversarial Networks paper and running it on three `hello world` problems.

## Structure

```
├── README.md
├── constants.py
├── gan.py
├── main.py
├── /media
├── requirements.txt
└── utils.py
```


## Run
For training on the faces problem, run the following line:

```python3 main.py --problem FACES --seed 2022```

Run ```python3 main.py -h``` for the help message.


## Description
This project contains an implementation of the original `GAN paper` along training it on three small problems.
The implementation is based on pytorch, and logging the training progress is done in `Weights&Biases`.
The problems are as follows:

- `FACES`: The task is to generate faces, where each face is an image of four pixel, two black pixels on the main anti-diagonal line and two white pixels on the diagonal line. A small random variation on the blackness and whiteness is introduced into the data generation process.
- `SINE`: The task is to generate data points on a 2-D plane that will resemble a sine curve.
- `MNIST`: The task is to generate handwritten digits similar to the mnist dataset.




## Images
Examples of the generated data after some training iterations:

![alt text](media/faces.png?raw=true "Faces")
![alt text](media/sine.png?raw=true "Sine")
![alt text](media/mnist.png?raw=true "mnist")

## Author

- Hardy Hasan


## Resources
These resources are of great help to understand how a GAN system works:
- https://realpython.com/generative-adversarial-networks/
- https://www.youtube.com/watch?v=8L11aMN5KY8&ab_channel=Serrano.Academy
- https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f 