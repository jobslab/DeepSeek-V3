from model import MLA, ModelArgs, precompute_freqs_cis
import torch
import json

def test_MLA():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    with open("inference\\configs\\config_671B.json") as f:
        config = f.read()
        config = json.loads(config)
    args = ModelArgs(**config)
    x = torch.randn((2, 3, args.dim))
    start_pos = 0
    seqlen = x.size(1)
    freqs_cis = precompute_freqs_cis(args)

    freqs_cis = freqs_cis[start_pos:start_pos+seqlen]
    mask = torch.full((seqlen,seqlen), float("-inf")).triu_(1)
    model = MLA(args)
    latent = model.forward(x, start_pos, freqs_cis, mask)
    print(latent.shape)

if __name__ == '__main__':
    test_MLA()