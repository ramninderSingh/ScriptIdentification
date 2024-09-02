
common_config = {
    # 'data_dir': 'data/mnt/ramdisk/max/90kDICT32px/',
    # 'img_width': 100,
    # 'img_height': 32,
    'map_to_seq_hidden': 32,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 50,
    'train_batch_size': 512,
    'eval_batch_size': 512,
    'lr': 0.0005,
    'max_images_syn':200000,
    'max_images_real':9000,
    'patience': 10,
    'show_interval': 100,
    'classes':3,
    # 'valid_interval': 500,
    # 'hindi_path_syn': '/DATA1/ocrteam/ScriptDataset/TrainDataset/SynData/hindi',
    # 'english_path_syn':'/DATA1/ocrteam/ScriptDataset/TrainDataset/SynData/english',
    # 'gujarati_path_syn':'/DATA1/ocrteam/ScriptDataset/TrainDataset/SynData/gujarati/train',
    'hindi_path_real': '/DATA1/ocrteam/recognition/train/hindi/',
    'english_path_real':'/DATA1/ocrteam/recognition/train/english/',
    'punjabi_path_real':'/DATA1/ocrteam/recognition/train/punjabi/',
    # 'gujarati_path_real':'/DATA1/ocrteam/recognition/train/gujarati',
    'cpu_workers': 4,
    'checkpoints_dir': '/DATA1/ocrteam/CRNN/savedModels/HEP/'
    
}
train_config.update(common_config)

test_config = {
    'test_batch_size': 512,
    'cpu_workers': 4,
    'reload_model': '/DATA1/ocrteam/CRNN/savedModels/HEP/crnn_real_t2.pt',
    'max_images':9000,
    'classes':3,
    'hindi_path_real_train': '/DATA1/ocrteam/recognition/train/hindi/',
    'english_path_real_train':'/DATA1/ocrteam/recognition/train/english/',
    # 'gujarati_path_real_train':'/DATA1/ocrteam/recognition/train/gujarati'
    'punjabi_path_real_train':'/DATA1/ocrteam/recognition/train/punjabi/',
    'hindi_path': '/DATA1/ocrteam/recognition/test/hindi',
    'english_path':'/DATA1/ocrteam/recognition/test/english',
    'punjabi_path':'/DATA1/ocrteam/recognition/test/punjabi',
    # 'gujarati_path':'/DATA1/ocrteam/recognition/test/gujarati',
}
test_config.update(common_config)
