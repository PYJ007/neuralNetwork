import numpy as np
import scipy.special as spe
import matplotlib.pyplot as plt


class neuralnetwork:
    # 我们需要去初始化一个神经网络

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        self.activation_function = lambda x: spe.expit(x)  # 返回sigmoid函数

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        targets = np.array(targets_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# input_nodes = 2
# hidden_nodes = 3
# output_nodes = 4
# learn_rate = 0.05
# n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)
# data_file = open("C:/Users/44887/Downloads/crop_train.csv", 'r')
# data_list = data_file.readlines()
# data_file.close()
# file2 = open("C:/Users/44887/Downloads/crop_test.csv")
# answer_data = file2.readlines()
# file2.close()

# data = []
#
# sum_count = 0
# for i in range(15):
#     count = 0
#     for j in range(len(data_list)):
#         target = np.zeros(4) + 0.01
#         line_ = data_list[j].split(',')
#         imagearray = np.asfarray(line_)
#         target[int(imagearray[0])] = 1.0
#         n.train(imagearray[1:] / 100 * 0.99 + 0.01, target)
#     for line in answer_data:
#         all_values = line.split(',')
#         answer = n.query((np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01)
#         if answer[int(all_values[0])] > 0.85:
#             count += 1
#     sum_count += count
#     string = "训练进度 %05f\n本轮准确度 %05f\n总准确度 %05f\n\n" % (
#     i / 120, count / len(answer_data), sum_count / (len(answer_data) * (i + 1)))
#     data.append([i / 120, count / len(answer_data), sum_count / (len(answer_data) * (i + 1))])
#     print(string)
# numb=0
# for line in answer_data:
#     if numb % 100 == 0:
#         all_values = line.split(',')
#         answer = n.query((np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01)
#         max = 0
#         for i in range(10):
#             if answer[max]<answer[i]:
#                 max = i
#         print(all_values[0])
#         print(max)
#         print(answer[int(all_values[0])])
#         print()
#         print()
#     numb = numb+1


# flag = 0
# while flag != -1:
#     x = int(input("输入x"))
#     y = int(input("输入y"))
#     list = [x,y]
#     answer = n.query((np.asfarray(list) / 100 * 0.99) + 0.01)
#     max = 0
#     for i in range(4):
#         if answer[max]<answer[i]:
#             max = i
#     if max == 0:
#         print("1象限")
#     if max == 1:
#         print("2象限")
#     if max == 2:
#         print("3象限")
#     if max == 3:
#         print("4象限")
#     f = int(input("输入flag"))
#     flag = f