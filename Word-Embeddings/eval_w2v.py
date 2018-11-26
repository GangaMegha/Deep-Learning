import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', default=100, type=int)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vectors_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    # vector_dim = len(vectors[ivocab[0]])
    vector_dim = args.dimension
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    eps=1e-8
    d = (np.sum(W ** 2, 1) ** (0.5))+eps
    W_norm = (W.T / d).T
    evaluate_vectors(W_norm, vocab, ivocab, words, args)

def evaluate_vectors(W, vocab, ivocab, words, args):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'Semantic_Relatedness.txt', 'synonym_detection.txt', 'analogy.txt', 'odd_one.txt', 'fill_in.txt',
        ]

    prefix = ''

    print("\n\n\n{}".format(args.vectors_file))

    # Semantic Relatedness
    with open('%s%s' % (prefix, filenames[0]), 'r', encoding='utf8') as f:  

        print("\nSemantic relatedness checking.......\n\n")

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

    with open('output/cosine_val.txt','a',encoding='utf8') as f2:

        f2.write("\n\n\n{}\n".format(args.vectors_file))

        print("\nWriting output into cosine_val.txt ....")

        for j in range(len(data)):
            f2.write(data[j][0] + '\t' + data[j][1] + '\t' + str(dist[j]).encode("utf-8").decode('utf8') + '\n')
    print("\nDone.")



    
    # Synonymn Detection
    count_correct = 0

    with open('%s%s' % (prefix, filenames[1]), 'r', encoding='utf8') as f:  

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

            if(np.argmax(dist[j])+1 == 1):
                count_correct = count_correct + 1

    with open('output/synonymn.txt','a',encoding='utf8') as f2:
        
        f2.write("\n\n\n{}\n".format(args.vectors_file))

        print("\n\nWriting output into synonymn.txt .....")

        for j in range(len(data)):
            f2.write(data[j][0] + '\t' + "is most similar to" + '\t' + data[j][np.argmax(dist[j])+1] + '\n')

    print("\nAccuracy : {}".format(count_correct*100.0/num_iter))
    print("\nDone.")





    # Analogy detection
    with open('%s%s' % (prefix, filenames[2]), 'r', encoding='utf8') as f: 

        print("\n\n\n\nWord Analogy.......\n\n")

        full_data = [line.rstrip().split('\t') for line in f]
        print("Total number of test cases : {}".format(len(full_data)))
        
        data = [x for x in full_data if all(word in vocab for word in x)]
        #print(data)
        print("Number of test cases ignored due to the absence of words in the vocabulary : {}".format(len(full_data)-len(data)))
        
        indices = np.array([[vocab[word] for word in row] for row in data])
        print("Number of test cases considered : {}".format(len(indices)))
        ind1, ind2, ind3, ind4 = indices.T

        num_iter = len(ind1)
        dist = np.zeros((num_iter,5))

        for j in range(num_iter):
            pred_vec = (W[ind2[j], :] - W[ind1[j], :] +  W[ind3[j], :])
            dist_all = np.dot(W, pred_vec.T)

            dist_all[ind1[j]] = -np.Inf
            dist_all[ind2[j]] = -np.Inf
            dist_all[ind3[j]] = -np.Inf

            # prediction = np.argmax(dist_all, 0).flatten()
            prediction = dist_all.argsort()[-5:][::-1]
            dist[j] = prediction

        dist = dist.astype("int")

    with open('output/analogy.txt','a',encoding='utf8') as f2:

        f2.write("\n\n\n{}\n".format(args.vectors_file))

        print("\n\nWriting output into analogy.txt .....")

        for j in range(len(data)):
            f2.write(data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\t'+ \
                "Prediction : " + words[dist[j][0]] + ', ' + words[dist[j][1]] + ', ' + words[dist[j][2]] \
                + ', ' + words[dist[j][3]] + ', ' + words[dist[j][4]] + '\n')

    print("Done.")





    # Odd one out
    count_correct = 0

    with open('%s%s' % (prefix, filenames[3]), 'r', encoding='utf8') as f: 

        print("\n\n\n\nOdd one out.......\n\n")

        full_data = [line.rstrip().split('\t') for line in f]
        print("Total number of test cases : {}".format(len(full_data)))
        
        data = [x for x in full_data if all(word in vocab for word in x)]
        print("Number of test cases ignored due to the absence of words in the vocabulary : {}".format(len(full_data)-len(data)))
        
        indices = np.array([[vocab[word] for word in row] for row in data])
        print("Number of test cases considered : {}".format(len(indices)))
        ind1, ind2, ind3, ind4 = indices.T

        num_iter = len(ind1)
        dist = np.zeros((num_iter,4))

        for j in range(num_iter):
            #cosine similarity if input W has been normalized
            dist[j,0] = np.dot(W[ind1[j], :], W[ind2[j], :].T) + np.dot(W[ind1[j], :], W[ind3[j], :].T) + np.dot(W[ind1[j], :], W[ind4[j], :].T)
            dist[j,1] = np.dot(W[ind2[j], :], W[ind1[j], :].T) + np.dot(W[ind2[j], :], W[ind3[j], :].T) + np.dot(W[ind2[j], :], W[ind4[j], :].T)
            dist[j,2] = np.dot(W[ind3[j], :], W[ind1[j], :].T) + np.dot(W[ind3[j], :], W[ind2[j], :].T) + np.dot(W[ind3[j], :], W[ind4[j], :].T)
            dist[j,3] = np.dot(W[ind4[j], :], W[ind1[j], :].T) + np.dot(W[ind4[j], :], W[ind2[j], :].T) + np.dot(W[ind4[j], :], W[ind3[j], :].T)

            if(np.argmin(dist[j]) ==3):
                count_correct = count_correct + 1

    with open('output/odd_one.txt','a',encoding='utf8') as f2:

        f2.write("\n\n\n{}\n".format(args.vectors_file))

        print("\n\nWriting output into odd_one.txt .....")

        for j in range(len(data)):
            f2.write(data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\t'+ \
                "Odd one : " + data[j][np.argmin(dist[j])] + '\n')

    print("\nAccuracy : {}".format(count_correct*100.0/num_iter))
    print("\nDone.")





    # Sentence Completion
    count_correct = 0
    num_iter = 0

    with open('%s%s' % (prefix, filenames[4]), 'r', encoding='utf8') as f: 

        print("\n\n\n\nFill in the blanks.......\n\n")

        full_data = [x for x in f.readlines()]
        print("Total number of test cases : {}".format(len(full_data)))
        
        with open('output/sentence_fill.txt','a',encoding='utf8') as f2:

            f2.write("\n\n\n{}\n".format(args.vectors_file))

            print("\n\nWriting output into sentence_fill.txt .....")

            for line in full_data:
                sentence = np.array([x for x in line.split(' :')[0].split(' ')])
                options = np.array([x for x in line.split(': ')[1].split(', ')])

                sen = np.array([word for word in sentence if word in vocab])
                op = np.array([word for word in options if word in vocab])

                sen_indices = np.array([vocab[word] for word in sentence if word in vocab])
                op_indices = np.array([vocab[word] for word in options if word in vocab])

                if len(op_indices)==0 or len(sen_indices)==0 :
                    continue

                num_iter = num_iter + 1

                dist = np.zeros(len(op_indices))

                for i in range(len(dist)):
                    for j in range(len(sen_indices)):
                        dist[i] = dist[i] + np.dot(W[op_indices[i], :], W[sen_indices[j], :].T)

                if np.argmax(dist) == 0:
                        count_correct = count_correct + 1

                f2.write(line + '\t'+ "Prediction : " + op[np.argmax(dist)] + '\n')

    print("\nAccuracy : {}".format(count_correct*100.0/num_iter))
    print("\nDone.")

if __name__ == "__main__":
    main()
