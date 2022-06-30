import torch 
#from efficientnet_language import *

def load_subred(path):
    model = torch.load(path)
    print('#### model')  
    print(model)
    return model

def load_subspace_model():
    return load_subred('./models/subreg/subspace.pth')
    
def load_semantic_model():
    return load_subred('./models/subreg/semantic.pth')

#model = EfficientNet()