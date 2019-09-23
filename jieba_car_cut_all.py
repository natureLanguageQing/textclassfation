import kashgari
import numpy as np
from tensorflow import keras
import pandas as pd
import jieba as jieba
# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel, CNNModel

tf_board_callback = keras.callbacks.TensorBoard(log_dir='logs', update_freq=10000)

bert = BERTEmbedding('roberta', task="classification", sequence_length=300)

kashgari.config.use_cudnn_cell = True


def random(data):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    # 模型数据分配是 训练集 ：验证集：测试集 8:1:1
    train_data = [data[j] for i, j in enumerate(random_order) if i % 16 != 0]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 16 == 0]
    print(len(train_data), len(valid_data))
    return train_data, valid_data


class Classification:
    @staticmethod
    def read_message(filename):
        file = pd.read_csv(filename, encoding='utf-8').values.tolist()
        train_data, valid_data = random(file)
        # 原文
        x_items = []
        # 标签
        train_y = []
        #
        valid_x = []
        valid_y = []
        for i in valid_data:
            if len(i) > 2:
                valid_x.append(list(str(i[2]) + str(i[3])))
                # 标签放入训练y集合 不切字
                valid_y.append(str(i[1]))
        for i in train_data:
            if len(i) > 2:
                x_items.append(list(str(i[2]) + str(i[3])))
                # 标签放入训练y集合 不切字
                train_y.append(str(i[1]))
        return x_items, train_y, valid_x, valid_y

    # 训练模型
    def train(self):
        x_items, train_y, valid_x, valid_y = self.read_message('car/train.csv')
        # 获取bert字向量

        model = BLSTMModel(bert)
        # 输入模型训练数据 标签 步数
        model.fit(x_items,
                  train_y,
                  valid_x,
                  valid_y,
                  batch_size=64,
                  epochs=12,
                  callbacks=[tf_board_callback])
        # 保存模型
        file = pd.read_csv("car/test.csv", encoding='utf-8').values.tolist()
        test_data = []
        id_list = []
        for i in file:
            test_data.append(list(str(i[1]) + str(i[2])))
            id_list.append(i[0])
        predict_answers = model.predict(x_data=test_data)
        file = open("data/test_predict_bert_car.csv", 'w', encoding='utf-8')
        file.write("id,flag\n")
        for i, j in zip(id_list, predict_answers):
            i = i.strip()
            file.write(str(i) + "," + str(j) + "\n")
        model.save("../model/news-classification-bert-model")


if __name__ == '__main__':
    train = Classification()
    train.train()
