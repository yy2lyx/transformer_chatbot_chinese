import json
import re
import random

if __name__ == '__main__':
    corpus = []
    with open('data/ori_data/12万对话语料青云库.csv', 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            cont_1,cont_2 = line.split(' | ')
            corpus.append([cont_1, cont_2])

    start = []
    end = []
    re_lines = []
    with open('data/ori_data/xiaohuangji50w_nofenci.conv', 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            if re.match('^M', line):
               re_lines.append(line.strip())

    err_1 = []
    for i in range(len(re_lines)):
        if re_lines[i] == 'M':
            err_1.append(i)
    err_2 = [_ + 1 for _ in err_1]
    err = err_1 + err_2
    re_lines = [re_lines[i] for i in range(len(re_lines)) if i not in err]

    for i in range(len(re_lines)):
        if i % 2 == 0:
            start.append(re_lines[i].strip().split('M ')[1])
        else:
            end.append(re_lines[i].strip().split('M ')[1])
    for i in range(len(start)):
        corpus.append([start[i], end[i]])

    # 打乱预料
    random.shuffle(corpus)

    with open('data/gen_data/corpus.json', 'w',encoding='utf8') as p:
        json.dump(corpus, p,ensure_ascii=False)


