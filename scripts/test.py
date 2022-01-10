import os, sys
sys.path.append('.')
import numpy as np
from src.model import SubwordEmbedding

EXPECTED_EMBEDDING = np.array([[-0.0045,  0.0097,  0.0500,  0.0337, -0.0330],
                                [ 0.0011,  0.0044,  0.0108,  0.0488, -0.0035]])

if __name__=="__main__":
    print(f"expected embedding: \n{EXPECTED_EMBEDDING}\n")
    subword = SubwordEmbedding()
    subword.from_pretrained( 'resources/converted_cc.en.300.bin' )

    # see something vec
    something_vec = subword(['something', 'something is right'])
    model_emebdding = something_vec[:,:5]
    print(f"predicted embedding: \n{np.array(model_emebdding)}\n")

    subword.save("test")
    subword.restore("test")

    # see something vec
    something_vec = subword(['something', 'something is right'])
    model_emebdding = something_vec[:,:5]
    print(f"predicted embedding after loading and saving: \n{np.array(model_emebdding)}\n")
