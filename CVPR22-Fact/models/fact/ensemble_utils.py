import torch
from ..subreg.Subreg import *

def get_logit_from_subReg_ss(data): # from subspace
    model = load_semantic_model()
    result = model(data)
    return result 

def get_logit_from_subReg_sm(data): # from semantic
    model = load_subspace_model()
    result = model(data)
    return result 

def make_inference(logit1, logit2, logit3, voting = 'average'): ## for ensemble 
    if voting == 'max': 
        result = torch.maximum(logit1, logit2, logit3)
    else: # voting == 'average'
        result =logit1+logit2+logit3
        result = result/3
    return result 