import torch
import torch.nn as nn
import numpy as np
import sys
import time
from einops import rearrange

BLOCK_SIZE = torch.tensor(1024, dtype=torch.int32, device='cuda')
NEG_INF = torch.tensor(-1e10, dtype=torch.float32, device='cuda')# -infinity
EPSILON = torch.tensor(1e10, dtype=torch.float32, device='cuda')

def normal_attention(Q, K, V, mask=None):
    scale = 1 / np.sqrt(Q.shape[-1])
    Q = Q * scale
    QKt = torch.einsum('... i d, ... j d -> ... i j', Q, K)

    key_mask = rearrange(mask, 'b j -> b 1 1 j')
    QKt = torch.where(key_mask > 0, QKt, NEG_INF)

    attn = nn.functional.softmax(QKt, dim=-1)
    return attn @ V

def flash_attention_forward(Q, K, V, mask=None):
    O = torch.zeros_like(Q, requires_grad=True).to(device='cuda')
    l = torch.zeros(Q.shape[:-1])[...,None].to(device='cuda')
    m = torch.ones(Q.shape[:-1])[...,None].to(device='cuda') * NEG_INF

    # O = O.to(device='cuda')
    # l = l.to(device='cuda')
    # m = m.to(device='cuda')

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

    Tr = len(Q_BLOCKS)
    Tc = len(K_BLOCKS)

    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    for j in range(Tc):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        maskj = mask_BLOCKS[j]

        for i in range(Tr):
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled  = Qi * scale

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
            
            # Masking
            maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
            S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_ij = torch.exp(S_ij - m_block_ij)
            # Masking
            P_ij = torch.where(maskj_temp > 0, P_ij, torch.tensor(0., dtype=P_ij.dtype, device=P_ij.device))

            l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

            mi_new = torch.maximum(m_block_ij, mi)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij
            
            O_BLOCKS[i] = (li/li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new
        
    O = torch.cat(O_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)
    return O, l, m

def flash_attention_backward(Q, K, V, mask, O, l, m, dO):
    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

    Tr = len(Q_BLOCKS)
    Tc = len(K_BLOCKS)

    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    dO_BLOCKS = list(torch.split(dO, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    dQ = torch.zeros_like(Q, requires_grad=True).to(device='cuda')
    dK = torch.zeros_like(K, requires_grad=True).to(device='cuda')
    dV = torch.zeros_like(V, requires_grad=True).to(device='cuda')

    dQ_BLOCKS = list(torch.split(dQ, Q_BLOCK_SIZE, dim=2))
    dK_BLOCKS = list(torch.split(dK, KV_BLOCK_SIZE, dim=2))
    dV_BLOCKS = list(torch.split(dV, KV_BLOCK_SIZE, dim=2))

    for j in range(Tc):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        maskj = mask_BLOCKS[j]

        dKj_block = torch.zeros_like(dK_BLOCKS[j], requires_grad=True).to(device='cuda')
        dVj_block = torch.zeros_like(dV_BLOCKS[j], requires_grad=True).to(device='cuda')

        for i in range(Tr):
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            dOi = dO_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled  = Qi * scale

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
            
            # Masking
            maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
            S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

            P_ij = (1/li) * torch.exp(S_ij - mi)
            # Masking
            P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

            dVj_block = dVj_block + torch.einsum('... r c, ... r d -> ... c d', P_ij, dOi)
            dP_ij = torch.einsum('... r d, ... c d -> ... r c', dOi, Vj)

            Di = torch.sum(dOi * Oi, dim=-1, keepdims=True)
            dS_ij = P_ij * (dP_ij - Di)

            dQ_BLOCKS[i] = dQ_BLOCKS[i] + scale * torch.einsum('... r c, ... c d -> ... r d', dS_ij, Kj)

            dKj_block = dKj_block + scale * torch.einsum('... r c, ... r d -> ... c d', dS_ij, Qi)
        
        dK_BLOCKS[j] = dKj_block
        dV_BLOCKS[j] = dVj_block
    
    dQ = torch.cat(dQ_BLOCKS, dim=2)
    dK = torch.cat(dK_BLOCKS, dim=2)
    dV = torch.cat(dV_BLOCKS, dim=2)
    return dQ, dK, dV

def flash_attention(Q, K, V, mask):
    out = flash_attention_forward(Q, K, V, mask)
    return out[0]

if __name__ == "__main__":
    Q = torch.randn(1, 2, 4096, 1024, requires_grad=True).to(device='cuda')
    K = torch.randn(1, 2, 4096, 1024, requires_grad=True).to(device='cuda')
    V = torch.randn(1, 2, 4096, 1024, requires_grad=True).to(device='cuda')
    mask = torch.randint(0, 2, (1, 4096)).to(device='cuda')

    for i in range(10):
        start1 = time.time_ns()
        out1 = flash_attention(Q, K, V, mask)
        end1 = time.time_ns()

        start2 = time.time_ns()
        out2 = normal_attention(Q, K, V, mask)
        end2 = time.time_ns()

        t1 = (end1 - start1) / 1000000
        t2 = (end2 - start2) / 1000000

        print(f'{t1}ms, {t2}ms')
        print(torch.allclose(out1, out2, atol=1e-5))