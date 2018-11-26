
#skiptrial
#python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_100_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 100


# Bow
#python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_05_100_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 100
#python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_05_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 200
#python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_05_300_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 300


#python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_025_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_075_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.075 -dim 200
python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_090_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.090 -dim 200


python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_05_200_3.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 3 -alpha 0.05 -dim 200
python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_05_200_5.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 5 -alpha 0.05 -dim 200
python word2vec.py -train corp_v2.txt -model cbow_models/vector_0_05_200_6.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 6 -alpha 0.05 -dim 200

python word2vec.py -train corp_v2_2.txt -model cbow_models/vector_2_0_05_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 200
python word2vec.py -train corp_v2_4.txt -model cbow_models/vector_4_0_05_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 200
python word2vec.py -train corp_v2_8.txt -model cbow_models/vector_8_0_05_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 200
python word2vec.py -train corp_v2_12.txt -model cbow_models/vector_12_0_05_200_4.txt -binary 0 -processes 20 -min-count 5 -cbow 1 -window 4 -alpha 0.05 -dim 200


#skipgram



python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_100_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 100
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_300_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 300


python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_050_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.05 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_075_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.075 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_090_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.090 -dim 200

#ganga below
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_200_3.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 3 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_200_5.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 5 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_025_200_6.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 6 -alpha 0.025 -dim 200

python word2vec.py -train corp_v2_2.txt -model skipgram/vector_neg5_2_0_025_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2_4.txt -model skipgram/vector_neg5_4_0_025_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2_8.txt -model skipgram/vector_neg5_8_0_025_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2_12.txt -model skipgram/vector_neg5_12_0_025_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200


python word2vec.py -train corp_v2.txt -model skipgram/vector_neg3_0_025_200_4.txt -binary 0 -negative 3 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg4_0_025_200_4.txt -binary 0 -negative 4 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg6_0_025_200_4.txt -binary 0 -negative 6 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg7_0_025_200_4.txt -binary 0 -negative 7 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.025 -dim 200

#very small learning rate for last
python word2vec.py -train corp_v2.txt -model skipgram/vector_neg5_0_010_200_4.txt -binary 0 -negative 5 -processes 20 -min-count 5 -cbow 0 -window 4 -alpha 0.01 -dim 200
