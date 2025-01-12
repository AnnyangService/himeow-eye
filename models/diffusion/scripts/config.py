from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    output_dir = "./output"
    save_image_epochs = 10
    save_model_epochs = 30
    seed = 42
