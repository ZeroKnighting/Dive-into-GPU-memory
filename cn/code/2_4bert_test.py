import torch
import torch.nn.functional as F
from transformers import BertModel,BertConfig
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

last_gpu_memory = 0

def get_memory():
  global last_gpu_memory
  last = last_gpu_memory
  now = torch.cuda.memory_allocated()/1024/1024
  last_gpu_memory = now
  return now,now-last
print('begin: ',get_memory())

def train():
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
    print("Model: ", get_memory())
    input = torch.zeros((32,512),dtype=torch.long).to('cuda')
    print("input: ",get_memory())
    out = model(input)
    print("Output and intermediate: ",get_memory())



    loss = torch.sum(out[0])
    print('loss: ',get_memory())
    loss.backward()
    print('backward',get_memory())



    optimizer = optim.AdamW(model.parameters())
    optimizer.step()
    print("optimizer: ",get_memory())
    print(torch.cuda.max_memory_allocated()/1024/1024)


def inference():
    with torch.no_grad():
        model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
        print("Model: ", get_memory())
        input = torch.zeros((32,512),dtype=torch.long).to('cuda')
        print("input: ",get_memory())
        out = model(input)
        print("Output and intermediate: ",get_memory())


def test_batchsize():
    plt.switch_backend('agg')
    x = []
    y = []
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
    for i in range(1,128,8):
        input = torch.zeros((i,128),dtype = torch.long).to('cuda')
        _ = model(input)
        x.append(i)
        y.append(torch.cuda.memory_allocated()/1024/1024)
    plt.plot(x,y)
    plt.show()
    plt.savefig("./test_batchsize.jpg")

    


def test_sentenceLength():
    plt.switch_backend('agg')
    x = []
    y = []
    model = BertModel(BertConfig(max_position_embeddings = 2560)).to('cuda')
    for i in range(16,2560,16):
        input = torch.zeros((1,i),dtype = torch.long).to('cuda')
        _ = model(input)
        x.append(i)
        y.append(torch.cuda.memory_allocated()/1024/1024)

    plt.plot(x,y)
    plt.show()
    plt.savefig("./test_sentenceLength.jpg")
    polyfit = np.polyfit(x,y,2) #拟合为2次
    len = np.poly1d(polyfit)
    print(len)


def main():
    print('input model')
    i = int(input('0:train, 1:inference, 2:test_batchsize, 3:test_sentenceLength\n'))
    if i == 0:
        train()
    elif i == 1:
        inference()
    elif i == 2:
        test_batchsize()
    elif i == 3:
        test_sentenceLength()


if __name__ == '__main__' :
    main()