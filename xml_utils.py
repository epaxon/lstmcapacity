import numpy as np

def generate_xml_sequence(num_chars_per_tag, tag_depth, num_tags):
    """
    
    """
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    rand_seq = np.random.randint(len(alphabet), size=num_chars_per_tag*tag_depth*num_tags)
    
    return rand_seq, alphabet
    
    