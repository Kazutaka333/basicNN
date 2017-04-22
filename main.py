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
    
    print("tarining ... (this could take more than several minutes)")

    # train the neural network
    epochs = 2
    for i in range(epochs):
        for record in training_data_list:

            all_values = record.split(',')

            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label
            # which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

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
        
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        outputs = n.query(inputs)
        
        label = numpy.argmax(outputs)
        
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

    print("performance = " + str(sum(scorecard)/len(scorecard)*100) + "%")
    
    # you can save your weights if you comments out theses
    # numpy.savetxt('wih_new.txt', n.getWIH())
    # numpy.savetxt('who_new.txt', n.getWHO())
    
	


if __name__ == "__main__":
    main()
