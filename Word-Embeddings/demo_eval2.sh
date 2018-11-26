
#skiptrial
python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_100_4.txt  --dimension 100


# Bow
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_05_100_4.txt  --dimension 100
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_05_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_05_300_4.txt  --dimension 300


python3 eval_w2v.py  --vectors_file cbow_models/vector_0_025_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_075_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_090_200_4.txt  --dimension 200


python3 eval_w2v.py  --vectors_file cbow_models/vector_0_05_200_3.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_05_200_5.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_0_05_200_6.txt  --dimension 200

python3 eval_w2v.py  --vectors_file cbow_models/vector_2_0_05_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_4_0_05_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_8_0_05_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file cbow_models/vector_12_0_05_200_4.txt  --dimension 200


#skipgram



python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_100_4.txt  --dimension 100
python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_300_4.txt  --dimension 300


python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_050_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_075_200_4.txt  --dimension 200
python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_090_200_4.txt  --dimension 200

#ganga below
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_200_3.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_200_5.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_025_200_6.txt.txt  --dimension 200

# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_2_0_025_200_4.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_4_0_025_200_4.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_8_0_025_200_4.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_12_0_025_200_4.txt.txt  --dimension 200


# python3 eval_w2v.py  --vectors_file skipgram/vector_neg3_0_025_200_4.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg4_0_025_200_4.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg6_0_025_200_4.txt.txt  --dimension 200
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg7_0_025_200_4.txt.txt  --dimension 200

# #very small learning rate for last
# python3 eval_w2v.py  --vectors_file skipgram/vector_neg5_0_010_200_4.txt.txt  --dimension 200
