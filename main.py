from config.config import device, epoch
import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from model.dataLoader import get_X_Data,get_Y_Data
from model.PromptMask import PromptMask
import time
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


def run_bert(seed, device):
    train_X_1, test_X_1 = get_X_Data('体育')
    train_Y_1, test_Y_1 = get_Y_Data('体育',len(train_X_1),len(test_X_1))
    train_X_2, test_X_2 = get_X_Data('社会')
    train_Y_2, test_Y_2 = get_Y_Data('社会',len(train_X_2),len(test_X_2))
    train_X_3, test_X_3 = get_X_Data('财经')
    train_Y_3, test_Y_3 = get_Y_Data('财经',len(train_X_3),len(test_X_3))
    train_X_4, test_X_4 = get_X_Data('房产')
    train_Y_4, test_Y_4 = get_Y_Data('房产',len(train_X_4),len(test_X_4))
    train_X_5, test_X_5 = get_X_Data('股票')
    train_Y_5, test_Y_5 = get_Y_Data('股票',len(train_X_5),len(test_X_5))
    train_X_6, test_X_6 = get_X_Data('彩票')
    train_Y_6, test_Y_6 = get_Y_Data('彩票',len(train_X_6),len(test_X_6))
    train_X_7, test_X_7 = get_X_Data('家居')
    train_Y_7, test_Y_7 = get_Y_Data('家居',len(train_X_7),len(test_X_7))
    train_X_8, test_X_8 = get_X_Data('教育')
    train_Y_8, test_Y_8 = get_Y_Data('教育',len(train_X_8),len(test_X_8))
    train_X_9, test_X_9 = get_X_Data('科技')
    train_Y_9, test_Y_9 = get_Y_Data('科技',len(train_X_9),len(test_X_9))
    train_X_10, test_X_10 = get_X_Data('时尚')
    train_Y_10, test_Y_10 = get_Y_Data('时尚',len(train_X_10),len(test_X_10))
    train_X_11, test_X_11 = get_X_Data('时政')
    train_Y_11, test_Y_11 = get_Y_Data('时政',len(train_X_11),len(test_X_11))
    train_X_12, test_X_12 = get_X_Data('游戏')
    train_Y_12, test_Y_12 = get_Y_Data('游戏',len(train_X_12),len(test_X_12))
    train_X_13, test_X_13 = get_X_Data('娱乐')
    train_Y_13, test_Y_13 = get_Y_Data('娱乐',len(train_X_13),len(test_X_13))

    train_X = torch.tensor(np.vstack([train_X_1, train_X_2, train_X_3, train_X_4, train_X_5, train_X_6, train_X_7, train_X_8, train_X_9, train_X_10, train_X_11, train_X_12, train_X_13]))
    train_Y = torch.tensor(np.vstack([train_Y_1, train_Y_2, train_Y_3, train_Y_4, train_Y_5, train_Y_6, train_Y_7, train_Y_8, train_Y_9, train_Y_10, train_Y_11, train_Y_12, train_Y_13]))
    test_X = torch.tensor(np.vstack([test_X_1, test_X_2, test_X_3, test_X_4, test_X_5, test_X_6, test_X_7, test_X_8, test_X_9, test_X_10, test_X_11, test_X_12, test_X_13]))
    test_Y = torch.tensor(np.vstack([test_Y_1, test_Y_2, test_Y_3, test_Y_4, test_Y_5, test_Y_6, test_Y_7, test_Y_8, test_Y_9, test_Y_10, test_Y_11, test_Y_12, test_Y_13]))

    train_data = TensorDataset(train_X, train_Y)
    test_data = TensorDataset(test_X, test_Y)

    loader_train = DataLoader(
        dataset=train_data,
        batch_size=20,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    loader_test = DataLoader(
        dataset=test_data,
        batch_size=20,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    net = PromptMask()
    net = net.to(device)
    acc = 0
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    
    for i in range(epoch):
        # if i > 5:
        #     current_lr *= 0.95
        #     change_lr(optimizer, current_lr)

        print('-------------------------   training   ------------------------------')
        time0 = time.time()
        batch = 0
        ave_loss, sum_acc = 0, 0
        for batch_x, batch_y in loader_train:
            net.train()
            batch_x, batch_y = Variable(batch_x).long(), Variable(batch_y).long()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = net(batch_x)
            criterion = nn.CrossEntropyLoss()
            batch_y = batch_y.reshape([40,1])
            batch_y = torch.squeeze(batch_y)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度缓存
            ave_loss += loss
            batch += 1

            if batch % 2 == 0:
                print('epoch:{}/{},batch:{}/{},time:{}, loss:{},learning_rate:{}'.format(i + 1, epoch, batch,len(loader_train),round(time.time() - time0, 4),loss,optimizer.param_groups[0]['lr']))
        # scheduler.step(ave_loss)
        print('------------------ epoch:{} ----------------'.format(i + 1))
        print('train_average_loss{}'.format(ave_loss / len(loader_train)))
        print('============================================'.format(i + 1))

        time0 = time.time()
        if (i + 1) % epoch == 0:
            label_out, label_y = [], []
            print('-------------------------   test   ------------------------------')
            # torch.save(net.state_dict(), 'save_model/params' + str(i + 1) + '.pkl')
            for batch_x, batch_y in loader_test:
                net.eval()
                batch_x, batch_y = Variable(batch_x).long(), Variable(batch_y).long()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                batch_y = batch_y.reshape([40,1])
                batch_y = torch.squeeze(batch_y)
                with torch.no_grad():
                     output = net(batch_x)
                _, pred = torch.max(output, dim=1)

                pred = pred.cpu().detach().numpy()
                batch_y = batch_y.cpu().detach().numpy()
                for j in range(pred.shape[0]):
                    label_out.append(pred[j])
                    label_y.append(batch_y[j])

            label_out = np.array(label_out)
            label_y = np.array(label_y)
            label_out = label_out.reshape([int(len(label_out)/2),2])
            label_y = label_y.reshape([int(len(label_y)/2),2])
            for i in range(len(label_out)):
                acc += 1 if all(label_out[i] == label_y[i]) else 0
            acc = acc / len(label_out)
            print('------------------ epoch:{} ----------------'.format(i + 1))
            print('test_acc:{}, time:{}'.format( round(acc, 4), time.time()-time0))
            print('============================================'.format(i + 1))
            try:
                with open('output/pre_out' + seed + '.txt', 'w', encoding='utf-8') as file:
                    for j in range(len(label_out)):
                        file.write(str(label_out[j]))
                        file.write('\n')
            except:
                print("文件写入异常")
            
    return acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    average_acc = 0
    seeds = [10, 100, 1000, 2000, 4000]
    for seed in seeds:
        setup_seed(seed)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        device = torch.device(device)
        average_acc += run_bert(seed,device)
    average_acc /= 5
    print('average_acc:{}'.format(round(average_acc, 4),))