
# coding: utf-8

# In[1]:


import argparse
from datetime import datetime
import math
import os
import subprocess
import time
import tensorflow as tf
import traceback

from datasets.new_datafeeder import get_dataset
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
log = infolog.log



def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def train(log_dir, args):
    run_name = args.name or args.model
    
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    with open(args.input, encoding='utf-8') as f:
        metadata = [row.strip().split('|') for row in f]
    metadata = sorted(metadata, key=lambda x:x[2])

    data_element = get_dataset(metadata, args.data_dir, hparams)
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('model') as scope:
        model = create_model(args.model, hparams)
        model.initialize(data_element['input'], 
                         data_element['input_lengths'], 
                         data_element['mel_targets'], 
                         data_element['linear_targets'])
        model.add_loss()
        model.add_optimizer(global_step)

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for _ in range(int(args.max_iter)):

        start_time = time.time()
        step, mel_loss, lin_loss, loss, opt = sess.run([global_step, model.mel_loss, model.linear_loss, model.loss, model.optimize])
        end_time = time.time()

        message = 'Step %7d [%.03f sec/step, loss = %.05f (mel : %.05f + lin : %.05f)]'%(step, end_time - start_time, loss, mel_loss, lin_loss)

        log(message)

        if loss > 100 or math.isnan(loss):
            log('Loss exploded to %.05f at step %d!' % (loss, step))
            raise Exception('Loss Exploded')

        if step % args.checkpoint_interval == 0:
            log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
            saver.save(sess, checkpoint_path, global_step=step)

            log('Saving audio and alignment...')
            input_seq, spectrogram, alignment = sess.run([model.inputs[0], model.linear_outputs[0], model.alignments[0]])
            waveform = audio.inv_spectrogram(spectrogram.T)
            audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
            plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                info='%s, %s, step=%d, loss=%.5f' % (args.model, time_string(), step, loss))

            log('Input: %s' % sequence_to_text(input_seq))
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('./'))
    parser.add_argument('--input', default='training/train.txt')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=100,
        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
        help='Steps between writing checkpoints.')
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
    parser.add_argument('--max_iter', type=int, default=1e5)
    parser.add_argument('--data_dir', default='training/')
    
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    
    run_name = args.name or args.model    
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)    
    infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
    
    hparams.parse(args.hparams)
    
    train(log_dir, args)


if __name__ == '__main__':
    main()
