import keras
import pandas as pd

# 读取文件数据 返回 训练数据 以及标签
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel, CNNModel
from thulac import thulac

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

cut = thulac(user_dict='../dictionary/THUOCL_medical.txt', seg_only=True)


class Classification:

    def f(self):
        pass

    @staticmethod
    def read_message(filename):
        file = open(filename, 'r', encoding='utf-8')
        # 原文
        x_items = []
        # 标签
        train_y = []
        for i in file:
            a, b, c, d = i.split(sep='\t')
            # 直接切分成字
            e = str(a + b + c)
            f = cut.cut(e)
            g = []
            for word in f:
                g.append(word)
            x_items.append(list(e))
            # 标签放入训练y集合 不切字
            d = d.replace('\n', '')
            train_y.append(str(d))
        return x_items, train_y

    # 训练模型
    def train(self):
        x_items, train_y = self.read_message('../data/健康管理师分类数据集/train.txt')
        x_xiyao, xiyao_y = self.read_message('../data/西药执业药师/train.txt')
        x_yingyangshi, yingyangshi_y = self.read_message('../data/yingyangshi/train.txt')
        x_items.extend(x_xiyao)
        train_y.extend(xiyao_y)
        x_items.extend(x_yingyangshi)
        train_y.extend(yingyangshi_y)

        x_dev, dev_y = self.read_message('../data/健康管理师分类数据集/valid.txt')
        # 获取bert字向量
        bert = BERTEmbedding('bert-base-chinese', sequence_length=200)
        model = BLSTMModel(bert)
        # 输入模型训练数据 标签 步数
        model.fit(x_items,
                  train_y,
                  x_dev,
                  dev_y,
                  epochs=8,
                  batch_size=128
                  , fit_kwargs={'callbacks': [tf_board_callback]}
                  )
        # 保存模型
        model.save("../健康管理师分字BERT-model")

    def interview(self):
        model = BLSTMModel.load_model("../健康管理师分字BERT-model")
        x_items, train_y = self.read_message('../data/健康管理师分类数据集/test.txt')
        model.evaluate(x_items, train_y)
        results_string: str = ''
        train_string: str = ''
        for i in x_items:
            results_string += model.predict(i)
        for j in train_y:
            train_string += j
        if len(results_string) == len(train_string):
            print('预测结果', results_string, '正确结果', train_string)
            print('五个五个去判断 全等就是做对了不全等就是做错了')
            a = len(train_string)
            b: float = a / 5
            right: int = 0
            for i in range(int(b)):
                var = results_string[5 * i:5 * (i + 1)]
                result = train_string[5 * i:5 * (i + 1)]
                if var == result:
                    print('做对了')
                    right += 1
                else:
                    print('做错了', var, result)
            acc = b - right
            print('正确答案', right, '错误答案', acc)
            print('准确率', right / b)

    def pre_evaluate(self):
        model = BLSTMModel.load_model("../健康管理师分字-model")
        x_items, train_y = self.read_message('../data/健康管理师分类数据集/test.txt')
        model.evaluate(x_items, train_y)

    def clean_message(self):

        exam_singles: pd.DataFrame = pd.read_csv('../data/健康管理师/健康管理师选择题去重.csv', encoding='UTF-8')
        exam_singles = exam_singles.drop_duplicates(

            subset=['问题', '选项A', '选项B', '选项C', '选项D', '选项E'],  # 去重列，按这些列进行去重

            keep='first'  # 保存第一条重复数据

        )
        exam_singles.to_csv('../data/健康管理师/健康管理师选择题去重再次去重.csv')


if __name__ == '__main__':
    # single_list: pd.DataFrame = pd.read_csv('../data/健康管理师/健康管理师单项选择题去重.csv', encoding='gbk')
    train = Classification()
    train.train()
    train.interview()
