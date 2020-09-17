#hyperparameter
hidden_size = 256 # hidden size of model 
layer_size = 3 # number of layers of model
dropout = 0.2 # dropout rate in training
bidirectional = True # use bidirectional RNN for encoder
use_attention = True # use attention between encoder-decoder
batch_size = 8 # batch size in training
workers = 4 # number of workers in dataset loader
max_epochs = 10 # number of max epochs in training
lr = 1e-04 # learning rate
teacher_forcing = 0.9 # teacher forcing ratio in decoder
max_len = 80 # maximum characters of sentence
seed = 1 # random seed
mode = 'train'

data_csv_path = './dataset/train/train_data/data_list.csv'

DATASET_PATH = './dataset/train'




# audio params
SAMPLE_RATE = 16000
WINDOW_SIZE = 0.02
WINDOW_STRIDE = 0.01
WINDOW = 'hamming'

# audio loader params 
SR = 22050
NUM_WORKERS = 4
BATCH_SIZE = 100 #600

#NUM_SAMPLES = 59049

# optimizer
LR = 0.0001
WEIGHT_DECAY = 1e-5
EPS = 1e-8

# epoch
MAX_EPOCH = 500
SEED = 123456
DEVICE_IDS=[0,1,2,3]

# train params 
DROPOUT = 0.5
NUM_EPOCHS = 300