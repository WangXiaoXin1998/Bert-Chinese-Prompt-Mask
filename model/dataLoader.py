from config.config import path, templete, maxlength
import os
from pytorch_transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained(path["pretrained"])

def get_X_Data(newsType):
    data_path = path['data']
    train_X = []
    test_X = []
    for root, dirs, files in os.walk(data_path+'Train_'+newsType):
        for f in files:
            with open(os.path.join(root, f), "r", encoding='utf-8') as news:  # 打开文件
                data = news.read().replace('\n','').replace('　','')  # 读取文件
                train_X.append('[CLS] '+data[0:maxlength]+' [SEP] [SEP] ' + templete + ' [SEP]')
    for root, dirs, files in os.walk(data_path+'Test_'+newsType):
        for f in files:
            with open(os.path.join(root, f), "r", encoding='utf-8') as news:  # 打开文件
                data = news.read().replace('\n','').replace('　','')  # 读取文件
                test_X.append('[CLS] '+data[0:maxlength]+' [SEP] [SEP] ' + templete + ' [SEP]')
    return X_data2id(train_X), X_data2id(test_X)

def get_Y_Data(newsType,len_train=0,len_test=0):
    if newsType == 'PE':
        templete = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('体育'))
    elif newsType == 'Social':
        templete = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('社会'))
    elif newsType == 'Finance':
        templete = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('财经'))
    train_Y = []
    test_Y = []
    for i in range(len_train):
        train_Y.append(templete)
    for i in range(len_test):
        test_Y.append(templete)
    return train_Y, test_Y

def X_data2id(X_data):
    '''
    将整个数据集转换成id
    :param X_data_text:
    :return:
    '''
    X_data_id = []
    for i in range(len(X_data)):
        X_data_tokens = tokenizer.tokenize(X_data[i])
        X_data_id.append(sentence2id(X_data_tokens, tokenizer))
    return np.array(X_data_id)

def sentence2id(tokens_a, tokenizer):
    '''
    将text转为 id
    :param sentence:
    :return:
    '''
    tt = np.ones(maxlength+20)
    tt0 = tokenizer.convert_tokens_to_ids(tokens_a)
    tt[0:min(len(tt0), maxlength+20)] = tt0[0:min(len(tt0), maxlength+20)]
    return tt