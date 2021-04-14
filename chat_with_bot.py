from net.nets import Transformer
from utils.get_config import get_conf
import json
import torch
import os
import pkuseg

class ChatWithBot:

    def __init__(self,vars):
        # 定义超参
        self.d_model = vars['d_model']
        self.heads = vars['heads']
        self.num_layers = vars['num_layers']
        self.device = torch.device(vars['device'])
        self.max_len = vars['max_len']
        self.vocab = self.load_vocab(vars['vocab_path'])
        self.model_weights_path = vars['model_weights_path']

    @staticmethod
    def load_vocab(vocab_path):
        """
        加载字典
        :param vocab_path:
        :return:
        """
        with open(vocab_path, 'r', encoding='utf8') as voc_file:
            vocab = json.load(voc_file)
        return vocab

    def load_model(self):
        transformer = Transformer(d_model=self.d_model, heads=self.heads, num_layers=self.num_layers,
                                       word_map=self.vocab).to(self.device)
        if (os.path.exists(self.model_weights_path)):
            state = torch.load(self.model_weights_path)
            transformer.load_state_dict(state['model_state_dict'])
            print('加载模型成功！')
        else:
            print('没有模型权重文件！')
        return transformer


    def respond(self,transformer, question, question_mask):
        """
        结果输出
        """
        rev_word_map = {v: k for k, v in self.vocab.items()}
        transformer.eval()
        start_token = self.vocab['<start>']
        encoded = transformer.encode(question, question_mask)
        words = torch.LongTensor([[start_token]]).to(self.device)

        for step in range(self.max_len - 1):
            size = words.shape[1]
            target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
            target_mask = target_mask.to(self.device).unsqueeze(0).unsqueeze(0)
            decoded = transformer.decode(words, target_mask, encoded, question_mask)
            predictions = transformer.logit(decoded[:, -1])
            _, next_word = torch.max(predictions, dim=1)
            next_word = next_word.item()
            if next_word == self.vocab['<end>']:
                break
            words = torch.cat([words, torch.LongTensor([[next_word]]).to(self.device)], dim=1)  # (1,step+2)

        # Construct Sentence
        if words.dim() == 2:
            words = words.squeeze(0)
            words = words.tolist()

        sen_idx = [w for w in words if w not in {self.vocab['<start>']}]
        sentence = ''.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])
        return sentence

    def chat(self):
        # 1. 加载模型
        transformer = self.load_model()
        transformer.eval()

        # 2. 加载分词
        seg = pkuseg.pkuseg(model_name="web", user_dict="default")

        # 3. 聊天输出
        print("机器人: 你好", flush=True)
        break_next = False
        while (True):
            if not break_next:
                print("你: ", end='', flush=True)
            user_input = input()

            if break_next:
                break

            if (user_input == "拜拜"):
                break_next = True

            # 分词
            words = seg.cut(user_input)
            # 替换掉没在词库中的词
            enc_qus = [self.vocab.get(word, self.vocab['<unk>']) for word in words]

            user_input = torch.LongTensor(enc_qus).to(self.device).unsqueeze(0)
            user_input_mask = (user_input != 0).to(self.device).unsqueeze(1).unsqueeze(1)
            bot_response = self.respond(transformer, user_input, user_input_mask);
            print("机器人: " + bot_response, flush=True)



if __name__ == '__main__':
    vars = get_conf('conf.ini')
    chat_bot = ChatWithBot(vars)
    chat_bot.chat()











