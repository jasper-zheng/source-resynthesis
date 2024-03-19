import torch

def infer_song(model, x, device, batch_size = 4, chunk_size=32000):
    
    assert x.dim() == 2, "[Channel, T]"
    assert x.shape[1]>=chunk_size
    
    chunk_num = (x.shape[1] - x.shape[1]%chunk_size)//chunk_size
    chunks = x[:,:x.shape[1]-x.shape[1]%chunk_size].chunk(chunk_num,dim=1) 
    chunks_torch = torch.stack(chunks)
    chunks = chunks_torch.split(batch_size)
    # print([chunk.shape for chunk in chunks])
    
    sources = [[] for i in range(model.source_num)]
    model = model.to(device)
    with torch.no_grad():
        
        for chunk in chunks:
            out = model(chunk.to(device))
            for i,source in enumerate(out):
                sources[i].append(source)
    outs = [torch.cat(sources[i],axis=0).view(1,1,-1)[0].cpu() for i in range(model.source_num)]
    return outs


def infer_mask_from_song(model, x, device, batch_size = 4, chunk_size=32000):
    
    assert x.dim() == 2, "[Channel, T]"
    assert x.shape[1]>=chunk_size
    
    chunk_num = (x.shape[1] - x.shape[1]%chunk_size)//chunk_size
    chunks = x[:,:x.shape[1]-x.shape[1]%chunk_size].chunk(chunk_num,dim=1) 
    chunks_torch = torch.stack(chunks)
    chunks = chunks_torch.split(batch_size)
    # print([chunk.shape for chunk in chunks])
    
    sources = [[] for i in range(model.source_num)]
    masks = []
    model = model.to(device)
    with torch.no_grad():
        
        for chunk in chunks:
            out, m = model(chunk.to(device))
            for i,source in enumerate(out):
                sources[i].append(source)
    outs = [torch.cat(sources[i],axis=0).view(1,1,-1)[0].cpu() for i in range(model.source_num)]
    return outs