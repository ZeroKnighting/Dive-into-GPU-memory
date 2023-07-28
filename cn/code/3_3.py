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



plt.switch_backend('agg')
x = []
y = []
model = BertModel(BertConfig(max_position_embeddings = 2560)).to('cuda')
model = model.to_bettertransformer()
for i in range(16,2560,16):
    input = torch.zeros((1,i),dtype = torch.long).to('cuda')
    _ = model(input)
    x.append(i)
    y.append(torch.cuda.memory_allocated()/1024/1024)

plt.plot(x,y)
plt.show()
plt.savefig("./test_sentenceLength.jpg")