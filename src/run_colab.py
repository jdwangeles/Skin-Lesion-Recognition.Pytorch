import trainer

def exp_035():
    runner = trainer.trainer()
    runner.cuda_ids = "0,1"
    runner.n_epochs = 1
    runner.batch_size = 14
    runner.server = "lab_center"
    runner.eval_frequency = 10
    runner.backbone = "PNASNet5Large"
    runner.learning_rate = 1e-4
    runner.optimizer = "SGD"
    runner.initialization = "pretrained"
    runner.num_classes = 7
    runner.num_workers = 12  # Due to warning from utils
    runner.input_channel = 3
    runner.iter_fold = 1
    runner.seed = 47
    runner.disable_save = True
    # runner.resume = 0
    runner.train()

if __name__ == "__main__":
    exp_035()