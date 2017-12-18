# ANN
Artificial Neural Network for detection handwritten digits from MNIST dataset

MNIST DATASET: http://yann.lecun.com/exdb/mnist/

### Build:

#### CMAKE to create solution and download MNIST data (about 12 MB):

```
md ..\ann-course-practice-build
cd ..\ann-course-practice-build
cmake ..\ann-course-practice
```

Next unzip archives: `..\ann-course-practice-build\download\*.gz`

#### Build solution:

```
call "%VS140COMNTOOLS%\VsDevCmd.bat"
msbuild ..\ann-course-practice-build\ann-course-practice.sln /property:Configuration=Release
```

### Run:

#### CMD line params:
1. `images_relative_dir` - relative path to directory with MNIST dataset (relative to dir with executable file) (default = ../download)
2. `images_dir` - absolute path to directory with MNIST dataset
3. `num_hidden_layers` - number hidden neuron (default = 300)
4. `num_epoch` - number of epochs (default = 20)
5. `learning_rate` (default = 0.008)
6. `sigma` - standard deviation for weight initialization from normal distribution (default = 0.005)

#### Example:
`lab1-mnist sigma=0.01 learn=0.01`

### Accuracy after 20 epochs with default params:  
Train: 0.999917
Test: 0.9808