import csv

import keras
# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel
from keras.callbacks import ModelCheckpoint
from thulac import thulac


class Classification:

    def __init__(self):
        self.cut = thulac(user_dict='../dictionary/THUOCL_medical.txt', seg_only=True)
        self.tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

    def read_message(self, filename):
        file = open(filename, 'r', encoding='utf-8')
        # 原文
        x_items = []
        # 标签
        train_y = []
        for i in file:
            a, b, c, d = i.split(sep='\t')
            # 直接切分成字
            e = str(a + b + c)
            f = self.cut.cut(e)
            h = ''
            for word in f:
                for words in word:
                    h += words + ' '
            x_items.append(list(e))
            # 标签放入训练y集合 不切字
            d = d.replace('\n', '')
            train_y.append(str(d))
        return x_items, train_y

    # 训练模型
    def train(self):
        # filepath = "saved-model-{epoch:02d}-{acc:.2f}.hdf5"
        # checkpoint_callback = ModelCheckpoint(filepath,
        #                                       monitor='acc',
        #                                       verbose=1)
        x_items, train_y = self.read_message('../data/yingyangshi/train.txt')
        x_dev, dev_y = self.read_message('../data/yingyangshi/dev.txt')
        # 获取bert字向量
        bert = BERTEmbedding('textclassfation/input0/chinese_L-12_H-768_A-12')
        model = BLSTMModel(bert)
        # model.build_multi_gpu_model(gpus=2)
        model.fit(x_items,
                  train_y,
                  x_dev,
                  dev_y,
                  epochs=2,
                  batch_size=64)
        # 保存模型
        model.save("../健康管理师单选分字BERT-model")

    def interview(self):
        model = BLSTMModel.load_model("../健康管理师单选分字BERT-model")
        x_items, train_y = self.read_message('../data/yingyangshi/test.txt')
        x_full = self.full_message('../data/yingyangshi/test.txt')
        model.evaluate(x_items, train_y)
        results_string: str = ''
        train_string: str = ''
        right_predict: list = []
        wrong_predict: list = []
        for i in x_items:
            results_string += model.predict(i)
        for j in train_y:
            train_string += j
        if len(results_string) == len(train_string):
            print('预测结果', results_string, '正确结果', train_string)
            print('五个五个去判断 全等就是做对了不全等就是做错了')
            a = len(train_string)
            b: int = int(a / 5)
            print('验证数据集长度', b)
            right: int = 0
            for i in range(b):
                start_single: int = i * 5
                end_single: int = (i + 1) * 5
                single = x_full[start_single:end_single]
                var = results_string[start_single:end_single]
                result = train_string[start_single:end_single]
                if var == result:
                    print('做对了')
                    right_predict.append(single)
                    for varey in single:
                        print(varey)

                    right += 1
                else:
                    print('做错了', var, result)
                    wrong_predict.append(single)
                    for varey in single:
                        print(varey)
            acc = b - right
            with open('wrong single.csv', 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                for wrong_list in wrong_predict:
                    for message in wrong_list:
                        wrong_list = message.split('\t')
                        csv_writer.writerow(wrong_list)
            with open('right single.csv', 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                for right_list in right_predict:
                    for message in right_list:
                        message = message.split('\t')
                        csv_writer.writerow(message)
            print('正确答案', right, '错误答案', acc)
            print('准确率', right / b)

    def pre_evaluate(self):
        model = BLSTMModel.load_model("../健康管理师分字-model")
        x_items, train_y = self.read_message('../data/health_manager_v4/test.txt')
        model.evaluate(x_items, train_y)

    @staticmethod
    def full_message(param):

        file = open(param, 'r', encoding='utf-8')
        # 原文
        x_items = []
        for i in file:
            x_items.append(i)
            # 标签放入训练y集合 不切字
        return x_items


if __name__ == '__main__':
    train = Classification()
    train.train()
