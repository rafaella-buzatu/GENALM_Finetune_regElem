import torch
import matplotlib.pyplot as plt
import xmltodict
import numpy as np
import os
import urllib

def format_attention(attention):
    '''Convert the attention matrix from tuple to tensor.
    '''
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def getSequence(chromosome, start, end, refGenome = 'hg38'):
    """
    Returns the DNA sequence 

    Parameters
    ----------
    chromosome : string
        The chromosome number.
    start : string
        The starting position of the DNA sequence.
    end : string
        The end position of the DNA sequence.
    refGenome: string
        The reference genome to use to extract the DNA sequence.
        { 'hg38' - default; 'hg19'}
        

    Returns
    -------
    sequence : string
        The DNA sequence corresponding to the input location.

    """
    base = 'http://genome.ucsc.edu/cgi-bin/das/' + refGenome + '/dna?segment='
    url = base + chromosome + ':' + str(start) + ',' + str(end)
    response = urllib.request.urlopen(url).read()
    data = xmltodict.parse(response)
    sequence = data['DASDNA']['SEQUENCE']['DNA']['#text'].upper().replace('\n','')
    
    return sequence

def get_attention(model, tokenizer, device, sentence_a):
    ''' Get an attention score for each token of a sentence.
    '''
    #Encode the input
    inputs = tokenizer.encode_plus(sentence_a, max_length=4096, return_tensors="pt", 
                                   padding=True, truncation=True)
    inputs = inputs.to(device)
    input_ids = inputs['input_ids']
    #Get the attention matrix from the ouputs of the model
    attention = model(input_ids,  output_attentions=True)[-1]
    input_id_list = input_ids[0].tolist() # Batch index 0
    #Get the tokens from the tokenizer
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
    #Format the attention matrix into tensor
    attn = format_attention(attention)

    #Get the attention score for each token
    attn_score = []
    for i in range(1, len(tokens)-1):
        #for each token, sum all attention scores going into CLS for all layers and heads
        attn_score.append(float(attn[:,:,0, i].sum()))
    
    return tokens, attn_score



def add_scores_to_string(long_string, scores_dict):
    '''Generate a list of scores for each character in a given string by adding up all the 
    characters in the string that are part of a substring in the scores_dict.
    '''
    scores = [0] * len(long_string)
    for substring, score in scores_dict.items():
        start = 0
        while True:
            index = long_string.find(substring, start)
            if index == -1:
                break
            for i in range(index, index + len(substring)):
                scores[i] += score
            start = index + 1
    return scores



def get_indices_of_high_value_subsets(lst, threshold):

    #Subset the heatmap to only get the indices of the regions where the attention score is > threshold.
    subsets_indices = []
    subset = []
    
    for index, value in enumerate(lst):
        if value > threshold:
            subset.append(index)
        elif len(subset) >= 3:
            subsets_indices.append(subset)
            subset = []
        else:
            subset = []
    
    if len(subset) >= 3:
        subsets_indices.append(subset)
    
    return subsets_indices

    
def plot_heatmap(heatmap, sequence, location, pathToPlots, fileName):
    '''Plot the heatmap of the attention scores for a given sequence.
    '''
    #for i in range(len(sequencesToPlot)):
    plt.rcParams["figure.figsize"] = 20,2

    x = np.linspace(0, len(heatmap), len(heatmap))
        
    y = np.asarray(heatmap)
        
    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
    extent = [-0.1,  len(heatmap)-0.9, 0, 1]
    im = ax.imshow(y[np.newaxis,:], aspect="auto", extent=extent)
    fig.colorbar(im, ax= ax, orientation='horizontal', location = 'top')

    #Set ticks and labels
    ax.set_yticks([])
    ax2.set_xticks(ticks = [i for i in range( len(heatmap))], labels = [sequence[i] for i in range(len(heatmap))])
    #ax2.set_xticklabels([input_text[i]+str(i) for i in seqToPlot])
    ax.set_xlim(extent[0], extent[1])
    ax2.plot(x,y)

    #Add the location to the plot
    if ('random' in fileName):
        location = location + ' shuffled ' + fileName.split('random')[1]
    ax.text(0, 1.5, location, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(pathToPlots, fileName +'.png'))
    plt.show()


def getFeatures (input_text, DNAregions, model, tokenizer, device, threshold, 
                 returnHeatmap = False):
        '''Get the attention scores for each character in the input_text.
        '''

        #Get the attention scores for each character in the input_text
        #Calculate attention scores and tokens for a given input
        tokens, attentionScores = get_attention (model, tokenizer, device, input_text)
        #Remove CLS and SEP tokens
        tokens = tokens[1:-1]
        #Create dictionary of tokens and scores 
        attentionDict = {tokens[i]: attentionScores[i] for i in range(len(tokens))}

        #Generate a heatmap of attention scores for each character in the input_text 
        heatmap = add_scores_to_string(input_text, attentionDict)

        if (returnHeatmap):
            return heatmap
        else:
            #Get the indices of the regions>2 where the attention score is > threshold percentile.
            sequencesAttention = get_indices_of_high_value_subsets(heatmap, np.percentile(heatmap,threshold))
            #Empty list to store important tokens
            features = []

            #Iterate over the selected high attention regions
            for i in range(len(sequencesAttention)):

                seq = sequencesAttention[i]
                #Extract motif of at least 5 bases
                if(len(seq)<4):
                    
                    motif =''.join( [input_text[i] for i in list(range(seq[0] -1, seq[-1] + 1))])
                else:
                    motif = ''.join ([input_text[i] for i in seq])
                
                features.append(motif)
    
            return features
    
def writeListToFile(pathToDir, filename, lst):
    with open(os.path.join(pathToDir, filename), 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')
 
def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = []
    for letter in sequence:
        if letter in encoding:
            encoded_sequence.append(encoding[letter])
    return encoded_sequence
            
def writeListToMEME (pathToDir, filename, tokenList):
    with open(os.path.join(pathToDir, filename), 'w') as meme:
        meme.write("MEME version 4\n\nALPHABET= ACGT\n\n")
        meme.write("Background letter frequencies\nA 0.2699592 C 0.2297283 G 0.2299934 T 0.2703191\n")

        for token in tokenList:
            ppm = one_hot_encode(token)
            ### one_hot_encode of sequences
            meme.write("\nMOTIF " + token + "\nletter-probability matrix: alength= 4 w= " + str(len(token)) + "\n")
            for p in ppm:
                meme.write(" ".join(map(str, p)))
                meme.write("\n")
                
def writeListToBed (pathToDir, filename, positions, chromosome):
    with open(os.path.join(pathToDir, filename), 'w') as bed:
        for i in range(len(positions)):
            line = '\t'.join([chromosome, str(positions[i]), str(positions[i])])
            bed.write(line)
            bed.write("\n")
                