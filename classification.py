import keras
import pandas as pd

# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel, CNNModel

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./tf_dir', update_freq=10)

bert = BERTEmbedding('bert-base-chinese', sequence_length=128)


def read_message():
    excel = pd.read_csv('input1/all_message_thanks2019-07-01.csv', encoding="gbk").values.tolist()
    # 原文
    x_items_right = []
    # 标签
    train_y = []
    i_count = 0
    for i in excel:
        if str(i[13]) == "回答正确" and i_count <= 300:
            # 直接切分成字

            x_items_right.append((i[0] + i[1] + i[2] + i[3] + i[4]).split())
            # 标签放入训练y集合 不切字

            train_y.append("回答正确")
            i_count += 1
        else:
            x_items_right.append((i[0] + i[1] + i[2] + i[3] + i[4]).split())
            # 标签放入训练y集合 不切字

            train_y.append("回答错误")

    return x_items_right, train_y


def train():
    x_items, train_y = read_message()
    # 获取bert字向量
    model = CNNModel(bert)
    # 输入模型训练数据 标签 步数
    model.fit(x_items
              , train_y
              , epochs=20
              , class_weight=True
              , fit_kwargs={'callbacks': [tf_board_callback]})
    # 保存模型
    model.save("../classification-model")
    for i in x_items:
        result = model.predict(i)
        print("\n" + result)


if __name__ == '__main__':
    train()
