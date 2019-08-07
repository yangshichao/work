import jieba.posseg as psg
import jieba
import random
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import  to_categorical
from keras.layers import LSTM,Embedding
from keras.models import Sequential

import gensim
from gensim import corpora,models
from collections import Counter
import  pandas as pd
dir="F://work//torch//"
data="".join([dir,"car.csv"])
stop_words="".join([dir,"stopwords.txt"])
stopwords=pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
laogong = "".join([dir,'beilaogongda.csv'])  #被老公打
laopo = "".join([dir,'beilaopoda.csv'])  #被老婆打
erzi = "".join([dir,'beierzida.csv'])   #被儿子打
nver = "".join([dir,'beinverda.csv'])    #被女儿打
laogong_df = pd.read_csv(laogong, encoding='utf-8', sep=',')
laopo_df = pd.read_csv(laopo, encoding='utf-8', sep=',')
erzi_df = pd.read_csv(erzi, encoding='utf-8', sep=',')
nver_df = pd.read_csv(nver, encoding='utf-8', sep=',')

#删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
            segs = list(filter(lambda x:len(x)>1, segs)) #长度为1的字符
            segs = list(filter(lambda x:x not in stopwords, segs)) #去掉停用词
            sentences.append((" ".join(segs), category))# 打标签
        except Exception:
            print(line)
            continue
#调用函数、生成训练数据
sentences = []
preprocess_text(laogong, sentences, 0)
preprocess_text(laopo,sentences,1)
preprocess_text(erzi,sentences,2)
preprocess_text(nver,sentences,3)
random.shuffle(sentences)
all_texts=[sentence[0] for sentence in sentences]
all_labe1s=[sentence[1] for sentence in sentences]
MAX_SEQUENCE_LENGTH=100
EMBEDDING_DIM=200
VALIDATION_SPLIT=0.16
TEST_SPLIT=0.2
tokenizer=Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences=tokenizer.texts_to_sequences(all_texts)
word_index=tokenizer.word_index
data=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
labels=to_categorical(np.asarray(all_labe1s))
p1=int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2=int(len(data)*(1-TEST_SPLIT))
x_train=data[:p1]
y_tain=labels[:p1]
x_val=data[p1:p2]
y_val=labels[p1:p2]
x_test=data[p2:]
y_test=labels[p2:]
model=Sequential()
model.add(Embedding(len(word_index)))




# stopwords=pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# print(stopwords)
# stopwords=stopwords['stopword'].values
# df=pd.read_csv(data,encoding='gbk')
# df.dropna(inplace=True)
# lines=df.content.tolist()
# sentences=[]
# for line in lines:
#     try:
#         segs=jieba.lcut(line)
#         segs=[v for v in segs if not  str(v).isdigit()]
#         segs=list(filter(lambda x:x.strip(),segs))
#         segs=list(filter(lambda x:x not in stopwords),segs)
#     except Exception:
#         print(line)
#         continue
# dictionary=corpora.Dictionary(sentences)
# corpus=[dictionary.doc2bow(sentence) for sentence in sentences]
# lda=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=10)
# # print(lda.print_topic(1,topn=5))
# for topic in lda.print_topics(num_topics=10,num_words=8):
#     print(topic[1])


# print(type(lines))


