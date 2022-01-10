## Fasttext Subword Embeddings in PyTorch
[FastText](https://github.com/facebookresearch/fastText) is an incredible word embedding with a decent partial solution to handle OOV words and incorporate lexical similarity.     
<img src='img/model_summary.png' width="400" height="200">  
but what if we need to pass gradients through our fasttext embeddings?  

## Usage
Code snippet to demonstrate that it will replicate the original fasttext embeddings
```
# Implemented model gives the same emebddings
from src.model import Subword_Embedding
subword = Subword_Embedding()
subword.from_pretrained( pretraining_folder = 'resources/cc.en.300.bin' )

# see something vec
something_vec = subword(['something', 'something is right'])
something_vec[:,:5]
tensor([[-0.0045,  0.0097,  0.0500,  0.0337, -0.0330],
        [ 0.0011,  0.0044,  0.0108,  0.0488, -0.0035]])
```

### Converting original `fasttext` embeddings into ingestible formats
Download model.  
`python scripts/download_model.py "en"`  
Convert the weights of the downloaded model.   
`python scripts/convert_pretrained_weights.py`  

### Modularity
To copy paste into your models, copy and paste 'src/model.py` simply as it is self-contained. 

## References
https://vimeo.com/234958672  
https://www.aclweb.org/anthology/Q17-1010/  
http://christopher5106.github.io/deep/learning/2020/04/02/fasttext_pretrained_embeddings_subword_word_representations.html 

