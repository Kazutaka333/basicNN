from neuralNetwork import *
import random

def main():
    # number of input, hidden and output nodes

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.15

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("MNIST/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    print("tarining ... (this could take more than 10 minutes)")

    # train the neural network
    epochs = 2
    # go through all records in the training data set
    for i in range(epochs):
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label
            # which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass

    test_data_file = open("MNIST/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    size = len(test_data_list)

    all_values = test_data_list[random.randint(0,size-1)].split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
   
    # test the neural network
    scorecard = []

    for record in test_data_list:
        
        all_values = record.split(',')
        
        correct_label = int(all_values[0])
        
       # print(correct_label, "correct label")
        
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        outputs = n.query(inputs)
        
        label = numpy.argmax(outputs)
        
       # print(label, "network's answer")
        
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass 

    print("performance = " + str(sum(scorecard)/len(scorecard)*100) + "%")
    # numpy.savetxt('wih.txt', n.getWIH())
    # numpy.savetxt('who.txt', n.getWHO())
    
	


if __name__ == "__main__":
    main()
