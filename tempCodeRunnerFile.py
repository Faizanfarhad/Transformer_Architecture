text = mh.text 
dmodel = 512
encoded =m.embedding_plus_postionalencode(text,dmodel)
_input = encoded
input_text = mh.input_para
multihead_instance = mh.MultiheadAttention(input_text,512)
sublayer = multihead_instance.multihead_output()



def layernorm(x,gamma,beta,epsilon=1e-6):
    mean = torch.mean(x,dim=-1,keepdim=True)
    variance = torch.var(x,dim=-1,keepdims=True,unbiased=False)
    normalized = (x - mean) / torch.sqrt(variance + epsilon) 
    return gamma * normalized + beta

def add_and_norm(x,sub_x,gamma,beta):
    residual = x + sub_x
    return layernorm(residual,gamma,beta)


gamma = torch.ones((1,1,512))
beta = torch.zeros((1,1,512))

layer = add_and_norm(encoded,sublayer,gamma,beta)    
