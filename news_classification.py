# 读取文件数据 返回 训练数据 以及标签
import kashgari
import pandas as pd
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel
from tensorflow import keras

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

bert = BERTEmbedding('wwm', task="classification", sequence_length=200)

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
                x_items.append(list(str(i[1])))
                # 标签放入训练y集合 不切字
                train_y.append(str(i[2]))
        return x_items, train_y

    # 训练模型
    def train(self):
        x_items, train_y = self.read_message('data/train.csv')
        # 获取bert字向量

        model = BLSTMModel()
        # 输入模型训练数据 标签 步数
        model.fit(x_items,
                  train_y,
                  batch_size=32,
                  epochs=20,
                  callbacks=[tf_board_callback])
        # 保存模型
        file = open("data/test_stage1.csv", 'r', encoding='utf-8')
        test_data = []

        for i in file.readlines()[1:]:
            test_data.append(i[1])
        predict_answers = model.predict(x_data=test_data)
        file = open("data/test_predict.csv", 'w', encoding='utf-8')
        for i, j in zip(test_data, predict_answers):
            file.write(i[0] + "," + j[0])
        model.save("../model/news-classification-model")


if __name__ == '__main__':
    train = Classification()
    train.train()
