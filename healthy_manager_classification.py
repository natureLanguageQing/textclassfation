import csv

# 读取文件数据 返回 训练数据 以及标签
import pandas as pd
from kashgari.tasks.classification import BLSTMModel
from thulac import thulac

thulac = thulac(seg_only=True)


def read_message(param):
    mulity_single = pd.read_csv(param).values.tolist()
    fenlei: list = []
    fenleo_label: list = []
    for multiple in mulity_single:
        question = multiple[0].replace("<MASK>", "")
        for single in multiple[1:6]:
            fenlei.append(thulac.cut(question + single, text=True).split())
        if "A" in multiple[6]:
            fenleo_label.append("对")
        else:
            fenleo_label.append("错")
        if "B" in multiple[6]:
            fenleo_label.append("对")
        else:
            fenleo_label.append("错")
        if "C" in multiple[6]:
            fenleo_label.append("对")
        else:
            fenleo_label.append("错")
        if "D" in multiple[6]:
            fenleo_label.append("对")
        else:
            fenleo_label.append("错")
        if "E" in multiple[6]:
            fenleo_label.append("对")
        else:
            fenleo_label.append("错")
    return fenlei, fenleo_label


class Classification:

    def interview(self):
        model = BLSTMModel.load_model("../model/health_manager_multi_bert-model")
        x_items, train_y = read_message('../data/health_manager_v2/multiple-choice.csv')
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
                var = results_string[start_single:end_single]
                result = train_string[start_single:end_single]
                if var == result:
                    print('做对了')
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
        model = BLSTMModel.load_model("../model/health_manager_multi_bert-model")
        result = model.predict("")


if __name__ == '__main__':
    train = Classification()
    train.interview()
