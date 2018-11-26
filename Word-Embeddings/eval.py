import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='../../vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='../../vectors_5_200_15_6.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    evaluate_vectors(W_norm, vocab, ivocab, words)

def evaluate_vectors(W, vocab, ivocab, words):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'Semantic_Relatedness.txt', 'synonym_detection.txt', 'analogy.txt',
        ]
    # filenames = [
    #     'v2.txt',
    #     ]
    prefix = '../question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size

    # Semantic Relatedness
    with open('%s/%s' % (prefix, filenames[0]), 'r', encoding='utf8') as f:  

        print("\n\nSemantic relatedness checking.......\n\n")

        full_data = [line.rstrip().split('\t') for line in f]
        print("Total number of test cases : {}".format(len(full_data)))
        
        data = [x for x in full_data if all(word in vocab for word in x)]
        print("Number of test cases ignored due to the absence of words in the vocabulary : {}".format(len(full_data)-len(data)))
        
        indices = np.array([[vocab[word] for word in row] for row in data])
        print("Number of test cases considered : {}".format(len(indices)))
        ind1, ind2 = indices.T

        num_iter = len(ind1)
        dist = np.zeros(num_iter)

        for j in range(num_iter):
            #cosine similarity if input W has been normalized
            dist[j] = np.dot(W[ind1[j], :], W[ind2[j], :].T)

    with open('cosine_val.txt','w',encoding='utf8') as f:

        print("\n\nWriting output into cosine_val.txt ....")

        for j in range(len(data)):
            f.write(data[j][0] + '\t' + data[j][1] + '\t' + str(dist[j]).encode("utf-8").decode('utf8') + '\n')
    print("\nDone.")
    
    # Synonymn Detection
    with open('%s/%s' % (prefix, filenames[1]), 'r', encoding='utf8') as f:  

        print("\n\n\n\nSynonymn detection.......\n\n")

        full_data = [line.rstrip().split('\t') for line in f]
        print("Total number of test cases : {}".format(len(full_data)))
        
        data = [x for x in full_data if all(word in vocab for word in x)]
        print("Number of test cases ignored due to the absence of words in the vocabulary : {}".format(len(full_data)-len(data)))
        
        indices = np.array([[vocab[word] for word in row] for row in data])
        print("Number of test cases considered : {}".format(len(indices)))

        indices2 = np.zeros((len(indices),5))
        for j in range(len(indices)):
            indices2[j,:] = np.array(indices[j])[:5]

        indices2 = indices2.astype("int")
        ind1, ind2, ind3, ind4, ind5 = indices2.T

        num_iter = len(ind1)
        dist = np.zeros((num_iter,4))

        for j in range(num_iter):
            #cosine similarity if input W has been normalized
            dist[j,0] = np.dot(W[ind1[j], :], W[ind2[j], :].T)
            dist[j,1] = np.dot(W[ind1[j], :], W[ind3[j], :].T)
            dist[j,2] = np.dot(W[ind1[j], :], W[ind4[j], :].T)
            dist[j,3] = np.dot(W[ind1[j], :], W[ind5[j], :].T)

    with open('synonymn.txt','w',encoding='utf8') as f:
        
        print("\n\nWriting output into synonymn.txt .....")

        for j in range(len(data)):
            f.write(data[j][0] + '\t' + "is most similar to" + '\t' + data[j][np.argmax(dist[j])+1] + '\n')
    print("Done.")

    # Analogy detection
    with open('%s/%s' % (prefix, filenames[2]), 'r', encoding='utf8') as f: 

        print("\n\n\n\nWord Analogy.......\n\n")

        full_data = [line.rstrip().split('\t') for line in f]
        print("Total number of test cases : {}".format(len(full_data)))
        
        data = [x for x in full_data if all(word in vocab for word in x)]
        print(data)
        print("Number of test cases ignored due to the absence of words in the vocabulary : {}".format(len(full_data)-len(data)))
        
        indices = np.array([[vocab[word] for word in row] for row in data])
        print("Number of test cases considered : {}".format(len(indices)))
        ind1, ind2, ind3, ind4 = indices.T

        num_iter = len(ind1)
        dist = np.zeros(num_iter)

        for j in range(num_iter):
            pred_vec = (W[ind2[j], :] - W[ind1[j], :] +  W[ind3[j], :])
            dist_all = np.dot(W, pred_vec.T)
            prediction = np.argmax(dist_all, 0).flatten()
            dist[j] = prediction

        dist = dist.astype("int")

    with open('analogy.txt','w',encoding='utf8') as f:

        print("\n\nWriting output into analogy.txt .....")

        for j in range(len(data)):
            f.write(data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\t'+ "Prediction : " + words[dist[j]] + '\n')

    print("Done.")


if __name__ == "__main__":
    main()
