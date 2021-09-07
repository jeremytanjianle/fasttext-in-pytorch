import argparse
import numpy as np
from tqdm import tqdm
import fasttext

def save_embeddings(model, output_dir):
    """
    save pretrained fasttext embeddings to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings"), model.get_input_matrix())
    with open(os.path.join(output_dir, "vocabulary.txt"), "w", encoding='utf-8') as f:
        for word in tqdm(model.get_words(), desc='saving words'):
            f.write(word+"\n")
            
def convert_embeddings(pretraining_path='cc.en.300.bin', output_dir='resources/converted_cc.en.300.bin'):
    ft = fasttext.load_model(pretraining_path)
    save_embeddings(ft, output_dir)

if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description='convert fasttext pretrained embeddings')
    parser.add_argument("--pretraining_path", type=str, default="resources/converted_cc.en.300.bin/cc.en.300.bin",
                        help="path_of_pretraining_file")
    parser.add_argument("--output_dir", default="resources/converted_cc.en.300.bin",
                        help="write to resources")
    args = parser.parse_args()    

    convert_embeddings(pretraining_path=args.pretraining_path, output_dir=args.output_dir)