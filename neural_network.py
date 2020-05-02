import matrix

'''
    File name: main.py
    Author: Michael Berge
    Date created: 7/19/2018
    Date modified: 5/2/2020
    Python Version: 3.8.1
'''

class NeuralNetwork:
    def __init__(self, args):
        if args[3] == 0:
            self.__input_nodes = args[0]
            self.__hidden_1_nodes = args[1]
            self.__output_nodes = args[2]

            # weights
            self.__weights_ih = matrix.Matrix(self.__hidden_1_nodes, self.__input_nodes)
            self.__weights_ho = matrix.Matrix(self.__output_nodes, self.__hidden_1_nodes)
            self.__weights_ih.randomize()
            self.__weights_ho.randomize()

            # bias
            self.__bias_h = matrix.Matrix(self.__hidden_1_nodes, 1)
            self.__bias_o = matrix.Matrix(self.__output_nodes, 1)
            self.__bias_h.randomize()
            self.__bias_o.randomize()

        else:
            self.__input_nodes = args[0]
            self.__hidden_1_nodes = args[1]
            self.__hidden_2_nodes = args[3]
            self.__output_nodes = args[2]

            # weights
            self.__weights_ih1 = matrix.Matrix(self.__hidden_1_nodes, self.__input_nodes)
            self.__weights_h1h2 = matrix.Matrix(self.__hidden_2_nodes, self.__hidden_1_nodes)
            self.__weights_h2o = matrix.Matrix(self.__output_nodes, self.__hidden_2_nodes)
            self.__weights_ih1.randomize()
            self.__weights_h1h2.randomize()
            self.__weights_h2o.randomize()

            # bias
            self.__bias_h1 = matrix.Matrix(self.__hidden_1_nodes, 1)
            self.__bias_h2 = matrix.Matrix(self.__hidden_2_nodes, 1)
            self.__bias_o = matrix.Matrix(self.__output_nodes, 1)
            self.__bias_h1.randomize()
            self.__bias_h2.randomize()
            self.__bias_o.randomize()

        # Learning rate
        self.lr = 0.1


    def feed_forward(self, i, args):
        if args[3] == 0:
            input_ = matrix.Matrix.from_array(i)

            hidden = matrix.Matrix.multiply(self.__weights_ih, input_)
            hidden.add(self.__bias_h)
            hidden.map(matrix.Matrix.sigmoid)

            output = matrix.Matrix.multiply(self.__weights_ho, hidden)
            output.add(self.__bias_o)
            output.map(matrix.Matrix.sigmoid)

            return matrix.Matrix.to_array(output)

        else:
            input_ = matrix.Matrix.from_array(i)

            hidden_1 = matrix.Matrix.multiply(self.__weights_ih1, input_)
            hidden_1.add(self.__bias_h1)
            hidden_1.map(matrix.Matrix.sigmoid)

            hidden_2 = matrix.Matrix.multiply(self.__weights_h1h2, hidden_1)
            hidden_2.add(self.__bias_h2)
            hidden_2.map(matrix.Matrix.sigmoid)

            output = matrix.Matrix.multiply(self.__weights_h2o, hidden_2)
            output.add(self.__bias_o)
            output.map(matrix.Matrix.sigmoid)

            return matrix.Matrix.to_array(output)


    def train(self, i, target_, args, gr):
        if args[3] == 0:
            # Feed-forward
            input_ = matrix.Matrix.from_array(i)

            hidden = matrix.Matrix.multiply(self.__weights_ih, input_)
            hidden.add(self.__bias_h)
            hidden.map(matrix.Matrix.sigmoid)

            output = matrix.Matrix.multiply(self.__weights_ho, hidden)
            output.add(self.__bias_o)
            output.map(matrix.Matrix.sigmoid)

            ###################################################################
            ######################### Backpropagation #########################
            ###################################################################

            # Calculate output errors
            target = matrix.Matrix.from_array(target_)
            output_errors = matrix.Matrix.subtract(target, output)

            # Calculate output gradient
            output_gradient = matrix.Matrix.map_(output, matrix.Matrix.d_sigmoid)
            output_gradient.multiply_(output_errors)
            output_gradient.multiply_(self.lr)

            # Calculate hidden --> output  weight deltas
            hidden_t = matrix.Matrix.transpose(hidden)
            weights_ho_deltas = matrix.Matrix.multiply(output_gradient, hidden_t)

            # Adjust hidden --> output weights
            self.__weights_ho.add(weights_ho_deltas)

            # Adjust output bias
            self.__bias_o.add(output_gradient)

            ###################################################################

            # Calculate hidden layer errors
            weights_ho_transpose = matrix.Matrix.transpose(self.__weights_ho)
            hidden_errors = matrix.Matrix.multiply(weights_ho_transpose, output_errors)

            # Calculate hidden gradient
            hidden_gradient = matrix.Matrix.map_(hidden, matrix.Matrix.d_sigmoid)
            hidden_gradient.multiply_(hidden_errors)
            hidden_gradient.multiply_(self.lr)

            # Calculate input --> hidden weight deltas
            input_t = matrix.Matrix.transpose(input_)
            weights_ih_deltas = matrix.Matrix.multiply(hidden_gradient, input_t)

            # Adjust input --> hidden weights
            self.__weights_ih.add(weights_ih_deltas)

            # Adjust hidden bias
            self.__bias_h.add(hidden_gradient)

            ###################################################################

            # Graphics
            gr.draw1(input_, hidden, output, self.__weights_ih, self.__weights_ho)
        else:
            # Feed-forward
            input_ = matrix.Matrix.from_array(i)

            hidden_1 = matrix.Matrix.multiply(self.__weights_ih1, input_)
            hidden_1.add(self.__bias_h1)
            hidden_1.map(matrix.Matrix.sigmoid)

            hidden_2 = matrix.Matrix.multiply(self.__weights_h1h2, hidden_1)
            hidden_2.add(self.__bias_h2)
            hidden_2.map(matrix.Matrix.sigmoid)

            output = matrix.Matrix.multiply(self.__weights_h2o, hidden_2)
            output.add(self.__bias_o)
            output.map(matrix.Matrix.sigmoid)

            ###################################################################
            ######################### Backpropagation #########################
            ###################################################################

            # Calculate output errors
            target = matrix.Matrix.from_array(target_)
            output_errors = matrix.Matrix.subtract(target, output)

            # Calculate output gradient
            output_gradient = matrix.Matrix.map_(output, matrix.Matrix.d_sigmoid)
            output_gradient.multiply_(output_errors)
            output_gradient.multiply_(self.lr)

            # Calculate hidden 2 --> output weight deltas
            hidden_2_t = matrix.Matrix.transpose(hidden_2)
            weights_h2o_deltas = matrix.Matrix.multiply(output_gradient, hidden_2_t)

            # Adjust hidden 2 --> output weights
            self.__weights_h2o.add(weights_h2o_deltas)

            # Adjust output bias
            self.__bias_o.add(output_gradient)

            ###################################################################

            # Calculate hidden 2 errors
            weights_h2o_transpose = matrix.Matrix.transpose(self.__weights_h2o)
            hidden_2_errors = matrix.Matrix.multiply(weights_h2o_transpose, output_errors)

            # Calculate hidden gradient
            hidden_2_gradient = matrix.Matrix.map_(hidden_2, matrix.Matrix.d_sigmoid)
            hidden_2_gradient.multiply_(hidden_2_errors)
            hidden_2_gradient.multiply_(self.lr)

            # Calculate hidden 1 --> hidden 2 weight deltas
            hidden_1_t = matrix.Matrix.transpose(hidden_1)
            weights_h1h2_deltas = matrix.Matrix.multiply(hidden_2_gradient, hidden_1_t)

            # Adjust hidden 1 --> hidden 2 weights
            self.__weights_h1h2.add(weights_h1h2_deltas)

            # Adjust hidden bias
            self.__bias_h2.add(hidden_2_gradient)

            ###################################################################

            # Calculate hidden 1 layer errors
            weights_h1h2_transpose = matrix.Matrix.transpose(self.__weights_h1h2)
            hidden_1_errors = matrix.Matrix.multiply(weights_h1h2_transpose, hidden_2_errors)

            # Calculate hidden gradient
            hidden_1_gradient = matrix.Matrix.map_(hidden_1, matrix.Matrix.d_sigmoid)
            hidden_1_gradient.multiply_(hidden_1_errors) # <------------------------- error here
            hidden_1_gradient.multiply_(self.lr)

            # Calculate input --> hidden weight deltas
            input_t = matrix.Matrix.transpose(input_)
            weights_ih1_deltas = matrix.Matrix.multiply(hidden_1_gradient, input_t)

            # Adjust input --> hidden weights
            self.__weights_ih1.add(weights_ih1_deltas)

            # Adjust hidden bias
            self.__bias_h1.add(hidden_1_gradient)

            ###################################################################

            # Graphics
            gr.draw2(input_, hidden_1, hidden_2, output, self.__weights_ih1, self.__weights_h1h2, self.__weights_h2o)