# lets make the data 400 in lenghth to match autoencoder imlementation
#lets have 10 different sections each 40 in lenght -- ideal conditions

import numpy as np

def create_synthetic_data(size=10000):
    print("beat patter dictionary ")
    ## basic patters
    downbeat    = np.array([1 if i % 4 == 0 else 0 for i in range(0,40)])
    downbeat2x  = np.array([1 if i % 2 == 0 else 0 for i in range(0,40)])
    high_hats   = np.array([2 if i % 4 == 1 else 0 for i in range(0,40)])
    high_hats2x = np.array([2 if i % 2 == 1 else 0 for i in range(0,40)])
    tom_drum    = np.array([3 if i % 4 == 2 else 0 for i in range(0,40)])

    ## combine basic patters
    comb1 = downbeat   + high_hats
    comb2 = downbeat   + high_hats2x
    comb3 = downbeat2x + high_hats2x
    comb4 = downbeat   + high_hats2x  + tom_drum
    comb5 = downbeat2x + high_hats


    print(list(downbeat))
    print(list(downbeat2x))
    print(list(high_hats))
    print(list(high_hats2x))
    print(list(tom_drum), "\n")

    print(list(comb1))
    print(list(comb2))
    print(list(comb3))
    print(list(comb4))
    print(list(comb5), "\n")


    ##here is the list we will permute
    musical_sections = [downbeat, downbeat2x, high_hats, high_hats2x, tom_drum, comb1, comb2, comb3, comb4, comb5]

    from itertools import permutations

    synth_data = np.array(list(permutations(musical_sections))[0:size]) ##okay but eventually you might need to trucate the permutations

    #reshape and unravel
    (a,b,c) = synth_data.shape
    synth_data = synth_data.reshape((a, b*c))
    # synth_data = synth_data[:,None,:] #add channel dim
    print("size of dataset ", synth_data.shape)

    print("Example datapoint:\n ", synth_data[0])

    return synth_data
