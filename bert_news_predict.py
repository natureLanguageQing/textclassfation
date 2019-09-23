import jieba
import kashgari
from tensorflow import keras
import pandas as pd

# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding

bert = BERTEmbedding('wwm', task="classification", sequence_length=300)

kashgari.config.use_cudnn_cell = True


class Classification:
    @staticmethod
    def read_message(filename):
        file = open(filename, 'r', encoding='utf-8')

        # 原文
        x_items = []
        ids = []
        # 标签
        for i in file.readlines()[1:]:
            a = i.split(sep=',')
            if len(a) > 1:
                # 直接切分成字
                x_items.append(jieba.lcut(a[1]))
                ids.append(a[0])

        return x_items, ids

    # 训练模型
    def train(self):
        x_items, ids = self.read_message('data/test_stage1.csv')
        # 获取bert字向量

        model = kashgari.utils.load_model("../model/news-classification-bert-model")
        # 输入模型训练数据 标签 步数
        predict_answer = model.predict(x_items,
                                       batch_size=64)
        print(predict_answer)
        file = open("data/test_predict_bert.csv", 'w', encoding='utf-8')
        for i, j in zip(ids, predict_answer):
            file.write(str(i) + "," + str(j)+"\n")


if __name__ == '__main__':
    train = Classification()
    train.train()
