import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)
    div_term_even = np.power(base, np.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = np.sin(positions / div_term_even)
    div_term_odd = np.power(base, (np.arange(1, d_model, 2) - 1) / d_model)
    pe[:, 1::2] = np.cos(positions / div_term_odd)
    return pe
