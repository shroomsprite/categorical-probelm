import csv
import sys
import numpy
import random
#import matplotlib.pyplot as plt


def main():
    raw_data = load_data()
    weights = [0,0,0]
    result = train(raw_data, weights)
    write_data(result)

def load_data():
    input_file = open(sys.argv[1], 'rb')
    reader = csv.reader(input_file)
    raw_data = numpy.asarray(list(reader), dtype=int)
    input_file.close()
    return raw_data

def write_data(data):
    output_file = open(sys.argv[2], 'wb')
    writer = csv.writer(output_file)
    writer.writerows(data)
    output_file.close()

def sign(x):
    if x >= 0.0:
        return 1
    else:
        return -1

def train(raw_data, weights):
    #learning_rate = 0.1
    b = 0
    new_b = 0.1
    output = []
    while new_b != b:
        b = new_b
        #plt.scatter(x, y, label = 'skitscat', color = 'k')
        #plt.xlabel('x1')
        #plt.ylabel('x2')
        #plt.title('iteration')
        #t = numpy.arange(0., 5., 0.2)
        #plt.plot(t, weights[1]*t+weights[2]*t-weights[0])
        #plt.show()
        
        for i in range(0, len(raw_data)):
            f_x = weights[0] + weights[1] * raw_data[i][0] + weights[2] * raw_data[i][1]
            error = raw_data[i][2]* f_x
            if error <= 0:
                #print error
                weights[0] += raw_data[i][2] * 1
                for j in range(1, len(weights)):
                    weights[j] +=  raw_data[i][2] * raw_data[i][j-1]# * learning_rate
                #print weights
                    #print guesses
        new_weights = [weights[0], weights[1], weights[2]]
        new_b = new_weights[0]
        output.append([new_weights[1], new_weights[2], new_weights[0]])
    #result = numpy.asarray(list(output), dtype=int)
    return output


    


'''
output_file = open(sys.argv[2], 'wb')
writer = csv.writer(output_file)
writer.writerows(raw_data)
input_file.close()
output_file.close()
'''
if __name__ == "__main__":
    main()

