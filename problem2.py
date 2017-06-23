import csv
import sys
import numpy

def main():
    raw_data = load_data()
    #write_data(raw_data)
    #print raw_data
    scaled = scale_data_add_intercept(raw_data)
    #result_array = numpy.asarray(list(scaled), dtype=float)
    #print result_array
    result = gradient_descent(scaled)
    write_data(result)
    #result_array = numpy.asarray(list(result), dtype=float)
    #print result_array
    
def load_data():
    input_file = open(sys.argv[1], 'rb')
    reader = csv.reader(input_file)
    raw_data = numpy.asarray(list(reader), dtype=float)
    input_file.close()
    return raw_data

def write_data(data):
    output_file = open(sys.argv[2], 'wb')
    writer = csv.writer(output_file)
    writer.writerows(data)
    output_file.close()

def scale_data_add_intercept(data):
    first_column =  []
    second_column = []
    for i in range(0, len(data)):
        first_column.append(data[i][0])
        second_column.append(data[i][1])
    first_array = numpy.asarray(list(first_column), dtype=float)
    second_array = numpy.asarray(list(second_column), dtype=float)
    stdev1 = numpy.std(first_array)
    stdev2 = numpy.std(second_array)
    #print stdev1
    #print stdev2
    result1 = []
    result2 = []
    result3 = []
    result = []
    for i in range(0,len(data)):
        result1.append(data[i][0] / stdev1)
        result2.append(data[i][1] / stdev2)
        result3.append(data[i][2])
        result.append([1.0, result1[i], result2[i], result3[i]])
    #print result
    result_array = numpy.asarray(list(result), dtype=float)
    #print result_array
    #print len(result_array)
    return result # the list
    #return result_array # the array

def gradient_descent(data):
    result = []
    for learning_rate in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.003]:
        own_choice = False
        new_beta0 = 0.0
        new_beta1 = 0.0
        new_beta2 = 0.0
        r_beta = 0.0
        #num_of_iterations = 0
        if learning_rate != 0.003:
            num_of_iterations = 100
        else:
            num_of_iterations = 300
        j = num_of_iterations
        while j > 0:
            beta_0 = new_beta0
            beta_1 = new_beta1
            beta_2 = new_beta2
            total_loss = 0.0
            beta0_factor = 0.0
            beta1_factor = 0.0
            beta2_factor = 0.0
            for i in range(0, len(data)):
                f_x = beta_0 * data[i][0] + beta_1 * data[i][1] + beta_2 * data[i][2]
                #print 'f_x(i): %f'%f_x
                total_loss += (f_x - data[i][3]) ** 2
                #print 'total: %f'%total_loss
                beta0_factor += (f_x - data[i][3]) * data[i][0]
                beta1_factor += (f_x - data[i][3]) * data[i][1]
                beta2_factor += (f_x - data[i][3]) * data[i][2]
        
            #print 'total: %f'%total_loss
            #print 'beta0_factor: %f'%beta0_factor
            #print 'beta1_factor: %f'%beta1_factor
            #print 'beta2_factor: %f'%beta2_factor
            r_beta = total_loss / (2 * len(data))
            new_beta0 = beta_0 - learning_rate * beta0_factor / len(data)
            new_beta1 = beta_1 - learning_rate * beta1_factor / len(data)
            new_beta2 = beta_2 - learning_rate * beta2_factor / len(data)
            j = j - 1
            #print 'beta_0: %f'%new_beta0
            #print 'beta_1: %f'%new_beta1
            #print 'beta_2: %f'%new_beta2
            #print 'R(beta): %f'%r_beta
            #print [new_beta0, new_beta1, new_beta2]
            #print 'iteration%d\n'%j
        result.append([learning_rate, num_of_iterations, new_beta0, new_beta1, new_beta2])
    return result


if __name__ == "__main__":
    main()
