def word2subword(word_token_string, N_GRAMS=3, pad_token = '<pad>', max_len=0):
    """
    word2subword('hello')
    >>> ['<hello>', '<he', 'hel', 'ell', 'llo', 'lo>']
    """
    full_word_w_sos_eos = '<'+word_token_string+'>'
    list_of_subword_tokens = [full_word_w_sos_eos]
    
    if len(word_token_string) > N_GRAMS:
        list_of_subword_tokens += [full_word_w_sos_eos[i:i+N_GRAMS] for i in range(len(full_word_w_sos_eos)+1-N_GRAMS)]
    
    if max_len>0:
        list_of_subword_tokens = list_of_subword_tokens[:max_len] + [pad_token]*max(max_len-len(list_of_subword_tokens),0)
    return list_of_subword_tokens