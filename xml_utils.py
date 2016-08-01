import numpy as np

def generate_rand_xml_sequence(num_chars_per_tag=[10], tag_depth=[4], num_tag_blocks=25):
    """
    
    """
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    tag_open_symbol = '<'
    tag_close_symbol = '-'
    tag_end_symbol = '>'
    
    aa = np.array(list(alphabet))
    
    # first decide the large-scale structure of the tags
    tag_structure = []
    
    for tblock in range(num_tag_blocks):
        # randomly decide how many tags we are going to open
        num_open = tag_depth[np.random.randint(len(tag_depth))]
        
        tag_structure.extend(num_open*[True])
        tag_structure.extend(num_open*[False])
    
    tag_seq = np.array([])
    tag_stack = []
    for itag, isopen in enumerate(tag_structure):
        
        if isopen:
            # randomly decide the length and letters of the tag
            taglen = num_chars_per_tag[np.random.randint(len(num_chars_per_tag))]
            tagchars = aa[np.random.randint(len(alphabet), size=taglen)]
            tag_stack.append(tagchars)
            
            tag_seq = np.hstack((tag_seq, tag_open_symbol, tagchars, tag_end_symbol))            
        else:
            # pop the stack to get the last open tag
            lasttagchars = tag_stack.pop()
            
            tag_seq = np.hstack((tag_seq, tag_close_symbol, lasttagchars, tag_end_symbol))
        
    
    return tag_seq
