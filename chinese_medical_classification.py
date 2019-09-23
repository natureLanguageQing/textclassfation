import keras
import pandas as pd

# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)
bert = BERTEmbedding('bert-base-chinese', sequence_length=200)
var = bert.model_key_map
print(var)


class Classification:
    @staticmethod
    def read_message(filename):
        # filename = '../data/中医执业药师考试/中医执业药师训练集（修正版本）.txt'
        file = open(filename, 'r', encoding='utf-8')

        # 原文
        x_items = []
        # 标签
        train_y = []
        for i in file:
            a, b, c, d = i.split(sep='\t')
            # 直接切分成字
            x_items.append(list(str(a + b + c)))
            # 标签放入训练y集合 不切字
            train_y.append(str(d))
        return x_items, train_y

    # 训练模型
    def train(self):
        x_items, train_y = self.read_message('../data/Chinese medicine licensed pharmacist/train.txt')
        x_dev, dev_y = self.read_message('../data/Chinese medicine licensed pharmacist/dev.txt')
        # 获取bert字向量

        model = BLSTMModel()
        # 输入模型训练数据 标签 步数
        model.fit(x_items,
                  train_y,
                  x_dev,
                  dev_y,
                  batch_size=32,
                  epochs=20,
                  fit_kwargs={'callbacks': [tf_board_callback]})
        # 保存模型
        model.save("../model/中医执业药师char-model")

    def pre_train(self):
        model = BLSTMModel.load_model("../model/中医执业药师classification-model")
        x_items, train_y = self.read_message('../data/Chinese medicine licensed pharmacist/test.txt')
        model.evaluate(x_items, train_y)


if __name__ == '__main__':
    train = Classification()
    train.train()
    train.pre_train()
