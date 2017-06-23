# categorical-probelm

How to run:
example: python problem1.py input1.csv output1.csv
make sure when running command above, problem1.py and input1.csv are in same directory. This command produces output1.csv, which is the output of a perceptron learning a model based on the data in input1.csv.

problem1.py creates a model by using perceptron algorithm to categorize a set of data with two opposite labels. The output1.csv is the output of perceptron learing the model and as it converge in th end, the learning process ends.

problem2.py creates a model by using linear regression with gradient descent algorithm. The output2.csv gives different performances based on the different learning rates and number of iterations.

problem3.py utilizes python library. Given a specific dataset, we can compare the performance of different models given different parameters. The second column gives the accuracy of the training data, and third column gives the accuracy of testing data when such a model after training applies to the testing data.
