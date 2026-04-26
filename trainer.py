from logging import Logger


class Trainer:
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.logger = Logger()

    # def train(self, data_processor, context_builder=None):
    #     self.model.train()
        
        # for epoch in range(self.config.num_epochs):
                   