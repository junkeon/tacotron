import os
import random
import numpy as np
import tensorflow as tf

from text import text_to_sequence

def get_dataset(metadata, data_dir, hparams): 
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    
    def _round_up(x):
        multiple = hparams.outputs_per_step
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder
    
    def _pad_target(t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=0)
    
    def transform_data():
        for lin_name, mel_name, _, text in metadata:
            input_data = np.asarray(text_to_sequence(text, cleaner_names), dtype=np.int32)
            
            linear_target = np.load(os.path.join(data_dir, lin_name))
            mel_target = np.load(os.path.join(data_dir, mel_name))
            
            linear_target = _pad_target(linear_target, _round_up(len(linear_target)))
            mel_target = _pad_target(mel_target, _round_up(len(mel_target)))
                        
            data = {}
            data['input'] = input_data
            data['input_lengths'] = len(input_data)
            data['mel_targets'] = mel_target
            data['linear_targets'] = linear_target
            
            yield data
    
    dataset_element = tf.data.Dataset.from_generator(transform_data, 
                                                     
                                                     {'input' : tf.int32, 
                                                      'input_lengths' : tf.int32, 
                                                      'mel_targets' : tf.float32, 
                                                      'linear_targets' : tf.float32
                                                     },
                                                     
                                                     {'input' : [None], 
                                                      'input_lengths' : [], 
                                                      'mel_targets' : [None, hparams.num_mels], 
                                                      'linear_targets' : [None, hparams.num_freq]
                                                     }
                                                    )    
    dataset_element = dataset_element.padded_batch(batch_size = hparams.batch_size, 
                                                   padded_shapes = {'input' : [None], 
                                                      'input_lengths' : [], 
                                                      'mel_targets' : [None, hparams.num_mels], 
                                                      'linear_targets' : [None, hparams.num_freq]}).prefetch(1).repeat()
    next_element = dataset_element.make_one_shot_iterator().get_next()  
    
    return next_element
