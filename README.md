## Fasttext Subword Embeddings in PyTorch
[FastText](https://github.com/facebookresearch/fastText) is great for things like handling OOV words using subwords...   
<img src='img/model_summary.png' width="400" height="200">  
but what if we need to pass gradients through our fasttext embeddings?  

## Usage
Code snippet to demonstrate that it will replicate the original fasttext embeddings
```
# Imeplemented model gives the same emebddings
from src.model import Subword_Embedding
subword = Subword_Embedding()
subword.from_pretrained( pretraining_folder = 'resources/cc.en.300.bin' )

# see something vec
something_vec = subword(['something', 'something is right'])
something_vec[:,:5]
tensor([[-0.0045,  0.0097,  0.0500,  0.0337, -0.0330],
        [ 0.0011,  0.0044,  0.0108,  0.0488, -0.0035]])
```
Thr original fasttext package will give the same embeddings:
```
import fasttext
ft = fasttext.load_model('resources/cc.en.300.bin/cc.en.300.bin')
orignal_something_word_ft = ft['something']
display(orignal_something_word_ft[:5])
orignal_something_word_ft = ft['something is right']
display(orignal_something_word_ft[:5])
array([-0.00450556,  0.00967976,  0.05004459,  0.03372731, -0.03298175],
      dtype=float32)
array([ 0.00113235,  0.00438947,  0.01076363,  0.04882365, -0.00348948],
      dtype=float32)
```
We can also save and restore in the following snippet.
Keep in mind that it saves 3 things: embedding weights, embedding size, vocab
```
# saving and restoring function works
subword.save('test')
subword2 = Subword_Embedding()
subword2.restore('test')

# see something vec
something_vec = subword2(['something', 'something is right'])
something_vec[:,:5]
tensor([[-0.0045,  0.0097,  0.0500,  0.0337, -0.0330],
        [ 0.0011,  0.0044,  0.0108,  0.0488, -0.0035]],
       grad_fn=<SliceBackward>)
```

Similarly, we can train models of the same architecture on a new corpus.  
See `src.model_from_scratch.py`
The train file demonstrate a general use case for the model.  
`python train.py`   
Generally, the results make sense.    
<img src='img/word_sim.png'>

### Converting original `fasttext` embeddings into ingestible formats
`python scripts/convert_pretrained_weights.py`  
`python scripts/download_model.py`

### Modularity
To copy paste into your models, copy and paste 'src/model.py` simply as it is self-contained. 

## References
https://vimeo.com/234958672  
https://www.aclweb.org/anthology/Q17-1010/  
http://christopher5106.github.io/deep/learning/2020/04/02/fasttext_pretrained_embeddings_subword_word_representations.html 

