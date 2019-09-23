import keras
import pandas as pd

# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel, CNNModel

tf_board_callback = keras.callbacks.TensorBoard(log_dir='/logs', update_freq=1000)


def read_message():
    excel = pd.read_csv(r'input1/二分类训练数据.csv', encoding='gbk', low_memory=False).values.tolist()
    # 原文
    x_items = []
    # 标签
    train_y = []
    for i in excel:
        # 直接切分成字
        if i[0] == i[0]:
            x_items.append(i[0].split())
            # 标签放入训练y集合
            train_y.append(i[13])
        else:
            break
    return x_items, train_y


def predict_message():
    excel = pd.read_csv(r'input1/二分类训练数据.csv', encoding='gbk').values.tolist()
    # 原文
    x_items = []
    for i in excel:
        # 直接切分成字
        if i[0] == i[0]:
            x_items.append((i[0] + i[1] + i[2] + i[3] + i[4] + i[5]).split())
        else:
            break
    return x_items


class Classification:

    def __init__(self):
        self.bert_place = 'input0/chinese_L-12_H-768_A-12'

    # 训练模型
    def train(self):
        x_items, train_y = read_message()
        # 获取bert字向量
        bert = BERTEmbedding(self.bert_place, sequence_length=256)
        model = CNNModel(bert)
        # 输入模型训练数据 标签 步数
        model.fit(x_items
                  , train_y
                  , epochs=200, batch_size=32
                  , fit_kwargs={'callbacks': [tf_board_callback]})
        # 保存模型
        model.save("output/classification-model")
        model.evaluate(x_items, train_y)

    def pre_train(self):
        model = CNNModel.load_model("output/classification-model")
        x_items, train_y = read_message()
        model.evaluate(self, x_items, train_y)


if __name__ == '__main__':
    train = Classification()
    train.train()
