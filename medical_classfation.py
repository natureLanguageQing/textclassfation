import keras
import pandas as pd

# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding, WordEmbeddings
from kashgari.tasks.classification import BLSTMModel, CNNModel
import thulac

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

thu = thulac.thulac(user_dict="../dictionary/THUOCL_medical.txt", seg_only=True)


class MedicalClassification:
    def f(self):
        pass

    @staticmethod
    def read_message(filename):
        # 原文
        x_items = []
        # 标签
        train_y = []
        file = open(filename, 'r', encoding='utf-8')
        for line in file:
            message = line.split('\t')
            # 已经分词文件 空格切分
            q_a = str(message[0] + ' ' + message[1] + ' ' + message[2])
            # 移除所有空格
            q_a = q_a.replace(" ", '')
            # 按字进行分割
            waiting_q_a = list(q_a)
            x_items.append(waiting_q_a)
            answer = message[3]
            # 移除标签中的\n
            answer = answer.replace("\n", "")

            train_y.append(answer)
        return x_items, train_y

    # 训练模型
    def train(self):
        x_train, train_y = self.read_message('../data/西药执业药师/train.txt')
        x_dev, dev_y = self.read_message('../data/西药执业药师/test.txt')
        x_test, test_y = self.read_message('../data/西药执业药师/dev.txt')
        # 获取bert字向量
        bert = BERTEmbedding('bert-base-chinese', sequence_length=100)
        # 获取词向量
        # embedding = WordEmbeddings('sgns.weibo.bigram.bz2', 50)

        long_model = CNNModel(bert)
        # 输入模型训练数据 标签 步数
        long_model.fit(x_train,
                       train_y,
                       x_dev,
                       dev_y,
                       epochs=20,
                       batch_size=128,
                       fit_kwargs={'callbacks': [tf_board_callback]})
        # 保存模型
        long_model.save("../classification-model")
        result = long_model.evaluate(x_test, test_y)
        return result

    def pre_train(self):
        bilstm_model = BLSTMModel.load_model('../classification-model')
        x_items, _ = self.read_message('../data/西药执业药师/dev.txt')
        for i in x_items:
            result = bilstm_model.predict(i)
            print("\n" + result)


if __name__ == '__main__':
    model = MedicalClassification()
    model.train()
