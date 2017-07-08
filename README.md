# intro-mldl
Introduction Machine Learning and Deep Learning

This repository contains a number of Python notebooks
for a a few Machine Learning and Deep
Learning use-cases.

- Movie recommendation system on MovieLens 100k dataset
  using [Apache Spark](http://spark.apache.org) 2.1.0.
- Recognition of hand-written digits from the MNIST
  dataset with modified LeNet-5 convolutional neural
  network using [Keras](https://keras.io) 2.0.5.


## Movie Recommendation with Spark MLlib
Example using the [MovieLens dataset](https://grouplens.org/datasets/movielens/100k/)
consisting of 100,000 recommendations by 1,000 users
for 1,700 movies.

[Python notebook](./notebooks/Collaborative%20Filtering.ipynb)


#### Installation of Spark
Prerequisites: Python 3.x (tested with Python 3.6.1)
Download the Spark binary (tested 2.1.1 pre-built for Hadoop 2.7 and later) from the [Spark download page](http://spark.apache.org/downloads.html).
Unzip the file.
```
cd <installdir>
tar xfz spark-2.1.1-bin-hadoop2.7.tgz
```

#### Download Dataset
Download MovieLens dataset from [http://files.grouplens.org/datasets/movielens/ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and unzip it by
running the download script. `<repodir>` is the directory
into which this repository was cloned.
```
$  cd <repodir>/data
$  sh download-ml100k.sh
```

#### Starting pyspark in Notebook
Make sure that pyspark is the `PATH` variable and the following
environment variables are set.

```
export SPARK_HOME=<installdir>/spark-2.1.1-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook' pyspark
```

Run pyspark which will automatically bring up the Jupyter
and open a window on the default browser. Open the
[Collaborative Filtering](notebooks/Collaborative%20Filtering.ipynb) notebook.
```
$ cd <repodir>
$ pyspark
```


## Recognition of Hand-written Digits with Keras
Classifier for hand-written digit dataset ([MNIST Dataset](http://yann.lecun.com/exdb/mnist)) with a
a convolution neural network (CNN). The model used is
modified version of Yann LeCun's LeNet-5.   

Yann LeCun, Léon Bouttou, Yoshua Benigo, Patrick Haffner:
[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
In Proc. of the IEEE, Volume 86, Issue 11, 1998.

[Python notebook](./notebooks/MNIST%20LeNet%20in%20Keras.ipynb)


#### Installation of Keras

Prerequisites: Python 2.7-3.6 (tested with Python 3.6.1)

Follow the instructions from the [Keros documentation](https://keras.io/#installation) for
version 2.0.5. Choose between
the [Theano](http://deeplearning.net/software/theano/) (0.9.0) or the [TensorFlow](https://www.tensorflow.org/) 1.2 backend. GPU-acceleration
is optional but requires the installation of the CUDA
runtime and the [cuDNN library](https://developer.nvidia.com/cudnn).


#### Running Notebook
Start Jupyter which will open a new page in the default
browser.
```
$ cd <repodir>
$ jupyter notebook
```

Open the
[Python notebook](./notebooks/MNIST%20LeNet%20in%20Keras.ipynb).
