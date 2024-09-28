
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
    # 'hindi_path_syn': path for hindi_syn(folder),
    # 'english_path_syn': path for english_syn,
    # 'gujarati_path_syn': path for guj_syn,
    # 'hindi_path_real': path for hindi_real,
    # 'english_path_real': path for eng_real,
    # 'punjabi_path_real': path for pun_real,
    # 'gujarati_path_real': path for guj_real,
    'cpu_workers': 4,
    # 'checkpoints_dir': model saving path
    
}
train_config.update(common_config)

test_config = {
    'test_batch_size': 512,
    'cpu_workers': 4,
    # 'reload_model': model_path,
    'max_images':9000,
    'classes':3,
    # 'hindi_path_real_train': hin_real_train_path,
    # 'english_path_real_train':eng_real_train_path,
    # 'gujarati_path_real_train':guj_real_train_path
    # 'punjabi_path_real_train':pun_real_train_path,
    # 'hindi_path': hin_test_path,
    # 'english_path':eng_test_path,
    # 'punjabi_path': pun_test_path,
    # 'gujarati_path': guj_test_path,
}
test_config.update(common_config)



infer_config = {
    'model_path':'trained_model_path',
    'img_path': 'image_path'
    # 'num':2

}

infer_config.update(common_config)