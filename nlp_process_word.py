from utils.get_config import get_conf
import json
import pkuseg
from collections import Counter


def encode_question(words, word_map, max_len):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    if len(words) > max_len:
        # 提问句子的处理：填未知词+pad
        enc_c = [word_map.get(word, word_map['<unk>']) for word in words][:max_len]
    return enc_c


def encode_reply(words, word_map, max_len):
    # 回答句子的处理：填未知词+pad + <start> + <end>
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + \
            [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(words))
    if len(words) > max_len:
        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words][:max_len] + [
            word_map['<end>']]
    return enc_c


if __name__ == '__main__':
    vars = get_conf(conf_path='conf.ini')

    # 设置最小单词数
    min_word_freq = vars['min_word_freq']
    max_len = vars['max_len']
    corpus_encoded_path = vars['corpus_encoded_path']
    vocab_path = vars['vocab_path']

    with open('data/gen_data/corpus.json', 'r', encoding='utf8') as corpus_file:
        corpus = json.load(corpus_file)

    # 对语料集进行处理（分字 + 填充标志符 + 限制长度）
    new_corpus = []
    word_freq = Counter()
    for i in range(len(corpus)):
        start_words = [_ for _ in corpus[i][0]]
        end_words = [_ for _ in corpus[i][1]]
        new_corpus.append([start_words, end_words])
        word_freq.update(start_words)
        word_freq.update(end_words)

    # 构建词汇表
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map = {k: v for k, v in sorted(word_map.items(), key=lambda item: item[1])}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    print("Total words are: {}".format(len(word_map)))

    with open(vocab_path, 'w', encoding='utf8') as j:
        json.dump(word_map, j, ensure_ascii=False)

    # 语料集的处理
    corpus_num = []
    for pair in new_corpus:
        question = encode_question(pair[0], word_map, max_len)
        answer = encode_reply(pair[1], word_map, max_len)
        corpus_num.append([question, answer])

    with open(corpus_encoded_path, 'w') as p:
        json.dump(corpus_num, p)

