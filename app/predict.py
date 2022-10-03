import numpy as np
from django.shortcuts import render, redirect
from test.cropTest import neuralnetwork as nw

flag = 0
input_nodes = 2
hidden_nodes = 3
output_nodes = 4

learn_rate = 0.05

n = nw(input_nodes, hidden_nodes, output_nodes, learn_rate)

def test(request):
    return render(request, 'test1.html')

def index(request):
    global flag
    global n
    if request.method=="GET":
        return render(request,'index.html')
    else:
        if flag == 0:
            n = train()
            flag = 1
        x = request.POST.get("longitude")
        y = request.POST.get("latitude")
        urban = request.POST.get("urban")
        detail = request.POST.get("detail")
        list = [x, y]
        answer = n.query((np.asfarray(list) / 100 * 0.99) + 0.01)
        max = 0
        for i in range(4):
            if answer[max] < answer[i]:
                max = i
        if max == 0:
            result = "水稻"
        if max == 1:
            result = "小麦"
        if max == 2:
            result = "枸杞"
        if max == 3:
            result = "大豆"
        return render(request,'test1.html',{"result":result,"r1":x,"r2":y,"r3":urban,"r4":detail})

def train():
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 4
    learn_rate = 0.05
    n = nw(input_nodes, hidden_nodes, output_nodes, learn_rate)
    data_file = open("C:/Users/44887/Downloads/crop_train.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    for i in range(15):
        for j in range(len(data_list)):
            target = np.zeros(4) + 0.01
            line_ = data_list[j].split(',')
            imagearray = np.asfarray(line_)
            target[int(imagearray[0])] = 1.0
            n.train(imagearray[1:] / 100 * 0.99 + 0.01, target)
    return n