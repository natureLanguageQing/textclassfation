import kashgari
from tensorflow import keras
import pandas as pd
import jieba as jieba
# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel, CNNModel

tf_board_callback = keras.callbacks.TensorBoard(log_dir='logs', update_freq=10000)

bert = BERTEmbedding('wwm', task="classification", sequence_length=300)

kashgari.config.use_cudnn_cell = True


class Classification:
    @staticmethod
    def read_message(filename):
        file = pd.read_csv(filename, encoding='utf-8').values.tolist()
        # 原文
        x_items = []
        # 标签
        train_y = []
        for i in file:
            if len(i) > 2:
                # 直接切分成字
                x_items.append(jieba.lcut(i[1]))
                # 标签放入训练y集合 不切字
                train_y.append(str(i[2]))
        return x_items, train_y

    # 训练模型
    def train(self):
        x_items, train_y = self.read_message('data/train.csv')
        # 获取bert字向量

        model = CNNModel(bert)
        # 输入模型训练数据 标签 步数
        model.fit(x_items,
                  train_y,
                  batch_size=64,
                  epochs=1,
                  callbacks=[tf_board_callback])
        # 保存模型
        file = pd.read_csv("data/test_stage1.csv", encoding='utf-8').values.tolist()
        test_data = []
        id_list = []
        for i in file:
            test_data.append(list(i[1]))
            id_list.append(i[0])
        predict_answers = model.predict(x_data=test_data)
        file = open("data/test_predict_bert.csv", 'w', encoding='utf-8')
        for i, j in zip(id_list, predict_answers):
            i = i.strip()
            file.write(str(i) + "," + str(j))
        model.save("../model/news-classification-bert-model")


if __name__ == '__main__':
    train = Classification()
    train.train()
