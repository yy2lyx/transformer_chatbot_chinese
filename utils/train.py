from net.nets import Transformer
from net.loss_opt import AdamWarmup,LossWithLS
from utils.dataset import create_masks
import json
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os

torch.random.seed()


class Trainer:
    def __init__(self,vars):
        # 定义超参
        self.d_model = vars['d_model']
        self.heads = vars['heads']
        self.num_layers = vars['num_layers']
        self.device = torch.device(vars['device'])
        self.epoch = vars['epoch']
        self.writer = SummaryWriter(vars['logs_path'])
        self.vocab = self.load_vocab(vars['vocab_path'])
        self.model_weights_path = vars['model_weights_path']
        self.learning_rate = vars['learning_rate']


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

    def init_model(self):
        """
        初始化模型
        :return:
        """
        self.transformer = Transformer(d_model=self.d_model, heads=self.heads, num_layers=self.num_layers, word_map=self.vocab).to(self.device)
        adam_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = AdamWarmup(model_size=self.d_model, warmup_steps=4000, optimizer=adam_optimizer)
        self.criterion = LossWithLS(len(self.vocab), 0.1)


    def train_process(self,train_loader,val_loader):
        self.init_model()

        # def checkpoint
        if os.path.exists(self.model_weights_path):
            checkpoint = torch.load(self.model_weights_path)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
        else:
            epoch_start = 0
        print("Training Starting from Epoch :{}".format(epoch_start))

        for epoch in range(epoch_start, self.epoch):
            epoch_loss = 0
            epoch_loss_val = 0
            count = 0
            count_val = 0

            # 训练集
            with tqdm(total=len(train_loader), desc=f'Epoch:{epoch}(train)') as t:
                for (question,reply) in train_loader:
                    self.transformer.train()

                    samples = question.shape[0]
                    question = question.to(self.device)
                    reply = reply.to(self.device)

                    # Prepare Data
                    reply_input = reply[:, :-1]
                    reply_target = reply[:, 1:]

                    # Create mask and add dimensions
                    question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input,
                                                                                      reply_target,self.device)

                    out = self.transformer(question, question_mask, reply_input, reply_input_mask)
                    # Loss is directly related to perplexity, but a more simple metric to calculate
                    loss = self.criterion(out, reply_target, reply_target_mask)
                    loss.backward()
                    epoch_loss += loss.item() * samples
                    count += samples

                    self.optimizer.step()
                    self.optimizer.optimizer.zero_grad()

                    t.set_postfix_str(
                        "Batch Loss: {:.4f}".format(loss.item()))
                    t.update()
                t.set_postfix_str(
                    "Training Loss: {:.4f}".format(epoch_loss / count))
                t.update()
            # 验证集
            with torch.no_grad():
                with tqdm(total=len(val_loader), desc=f'Epoch:{epoch}(val)') as t:
                    for (question,reply) in val_loader:
                        self.transformer.eval()

                        val_samples = question.shape[0]
                        question = question.to(self.device)
                        reply = reply.to(self.device)

                        # Prepare Data
                        reply_input = reply[:, :-1]
                        reply_target = reply[:, 1:]

                        # Create mask and add dimensions
                        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input,
                                                                                          reply_target, self.device)

                        out = self.transformer(question, question_mask, reply_input, reply_input_mask)
                        # Loss is directly related to perplexity, but a more simple metric to calculate
                        val_loss = self.criterion(out, reply_target, reply_target_mask)
                        epoch_loss_val += val_loss.item() * val_samples
                        count_val += val_samples

                        t.set_postfix_str(
                            "Batch Loss: {:.4f}".format(val_loss.item()))
                        t.update()
                    t.set_postfix_str(
                        "Val Loss: {:.4f}".format(epoch_loss_val / count_val))
                    t.update()
            self.writer.add_scalars('epoch_loss', {'train': epoch_loss / count,
                                                   'val': epoch_loss_val / count_val}, epoch)
            model_dict = {
                'epoch': epoch,
                'model_state_dict': self.transformer.state_dict(),
            }
            torch.save(model_dict, self.model_weights_path.split('.pkl')[0] + '_' + str(epoch) + '.pkl')
        self.writer.close()

        model_dict = {
            'epoch': epoch,
            'model_state_dict': self.transformer.state_dict(),
        }
        torch.save(model_dict, self.model_weights_path)







