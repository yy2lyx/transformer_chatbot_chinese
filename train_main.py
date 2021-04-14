from utils.dataset import TrainValLoader,ChatDataset
from utils.train import Trainer
from utils.get_config import get_conf
import torch


torch.random.seed()

if __name__ == '__main__':
    vars = get_conf('conf.ini')
    loader = TrainValLoader(vars)
    corpus = loader.load_corpus('data/gen_data/corpus_encoded_word.json')
    dataset = ChatDataset(corpus).corpus

    train_loader,val_loader = loader.get_train_val_loader()
    print('Loading dataset completed!')
    trainer = Trainer(vars)
    trainer.train_process(train_loader,val_loader)