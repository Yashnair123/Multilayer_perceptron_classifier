# Neural_net_classifier
Classifies different types of music
Authors: Yash Nair and Raluca Vlad

We built neural networks that can classify two different types of music. We uploaded approximately 5000 seconds of music of each Bach, Taylor Swift, and Electronic Dance Music for training the neural network and 500 seconds from each for the test data. We divided the music in 5 second parts, and each 5 second part represented one training/test input. We ran the neural network against Taylor vs. Bach and Electronic vs. Bach. We used three different neural net architectures (a Multilayer Perceptron, an RNN, and a CNN) and compared the accuracies.

Our audio files with music were saved in ‘.wav’ format. We used the Librosa library, which loads a ‘.wav’ file as a list of numbers, where each number represents the amplitude of the sound at a certain moment in time (our video provides a good visualization of how Librosa works). We also used parallelism in order to load the files with Taylor, Bach, and Electronic music in parallel. With the default sampling-rate of Librosa each 5 seconds of music had associated a list of approximately 110,000 numbers characterizing the amplitude of sound. Because it would have not been reasonable to feed the neural network with 110,000 numbers for each training/test input, we took every 20th number from that list in the ‘load’ function, reducing the length of the list to approximately 5,500. For the output, we put ‘0’ for Taylor/Electronic Music, and ‘1’ for Bach.

The variable ‘train\_input’ contains the merged lists for Taylor and Bach training data (or, Electronic and Bach, when we run Electronic vs. Bach). It can be seen in the video that the lists contains the data for approximately 2000 parts of 5 seconds of music. The variable ‘train\_output’ is a list of 0s and 1s, where 0 corresponds to a 5 second part of Taylor music and 1 corresponds to Bach music. The variables ‘test\_input’ and ‘test\_output’ are the same, but for the test data.

For the Multilayer Perceptron, we used several layers and massaged the parameters to get a maximum of 77% accuracy for both Taylor vs. Bach and Electronic vs. Bach. The parameters shown in the code (batch-size = 15) are the ones that lead to the best accuracy in the Taylor vs. Bach case; a 77% accuracy was obtained in the Electronic vs. Bach case with a batch-size of 30.

For the Recurrent Neural Network, we had to reshape the ‘train\_input’ and ‘test\_input’ to a 3-dimensional np array in order to fit in the LSTM (Long Short-Term Memory) algorithm. The RNN is the algorithm that had the poorest performance: a maximum of 68% accuracy for Taylor vs. Bach; and a maximum of 71% for Electronic vs. Bach.

For the Convolutional Neural Network, we had to create a matrix for each 5 seconds of music (instead of a list with ~5,500 numbers, we rearrange those numbers in a 74x74 matrix). Although a CNN is generally used for image classification, we obtained our best performance with it: 85% accuracy.

We started working on the neural network with much less data and obtained only a maximum of 61% accuracy. Once we uploaded more data, the accuracy suddenly increased, so we are confident that a larger training input would lead to even better results.

The necessary libraries are at the beginning of the neural_net.py file. In order for the code to run, the followings must be
run in the terminal window:

pip install tensorflow
pip install keras
pip install librosa
