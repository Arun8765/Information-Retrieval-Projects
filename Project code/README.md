# Next Word prediction using LSTM model

our project involves developing a language model that predicts the next word of the sequence 

To design a language model we follow the set of steps :

Step 1: Tokenizing the words in the text.
Step 2: Preprocessing the tokenized words.
Step 3: Give appropriate integer hash value to each word in the text for
easier processing.
Step 4: Make a 3-gram array of the integer hash value to train the data.
Step 5: Splitting the 3-gram into input and output.
Step 6: Selecting a suitable language model and architecture by selecting the
model parameters. The model used here is LSTM network.
Step 7: Train the LSTM network using the data from step 5.
Step 8: Input a string and expect the output to be possible next word in the
string.

#Software requirements

1. Python 3.7.6 or any other version compatible with keras library/
tensorflow
2. Tensorflow framework
3. Keras library
4. Numpy library
5. Jupyter notebook / suitable IDE

#hardware requirements

1. Central Processing Unit (CPU) — Intel Core i5 6th Generation processor
or higher. An AMD equivalent processor will also be optimal.
2. RAM — 8 GB minimum, 16 GB or higher is recommended.
3. Operating System — Ubuntu or Microsoft Windows 10.
4. NVIDIA GPU to execute keras library.

#runtime

1. To run the .ipynb file make sure the software and hardware requirements in fulfilled and run each cell individually.
2. change the model parameters depending on the nature and size of the dataset.
