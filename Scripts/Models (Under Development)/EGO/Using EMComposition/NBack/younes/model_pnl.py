import torch
import torch.nn as nn

## feed-forward WM
class FFWM(nn.Module):
    """Model used for Sternberg and N-back"""

    def __init__(self, indim, hiddim, outdim=2, bias=False):
        super().__init__()
        self.indim = indim
        self.hiddim = hiddim
        self.hid1_layer = nn.Linear(indim, indim, bias=bias)
        self.hid2_layer = nn.Linear(indim, hiddim, bias=bias)
        self.out_layer = nn.Linear(hiddim, outdim, bias=bias)
        self.drop2 = nn.Dropout(p=0.05, inplace=False)
        bias_dim = indim
        max_num_bias_modes = 10
        self.embed_bias = nn.Embedding(max_num_bias_modes, bias_dim)


    def forward(self, inputL, control_bias_int=0):
        """inputL is list of tensors"""
        hid1_in = torch.cat(inputL, -1)
        hid1_act = self.hid1_layer(hid1_in).relu()
        control_bias = self.embed_bias(torch.tensor(control_bias_int))
        hid2_in = hid1_act + control_bias
        hid2_in = self.drop2(hid2_in)
        hid2_act = self.hid2_layer(hid2_in).relu()
        yhat_t = self.out_layer(hid2_act)
        return yhat_t

def construct_model(
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
):
