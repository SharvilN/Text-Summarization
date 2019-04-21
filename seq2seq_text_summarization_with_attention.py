import os
import time
import utils as hp
import tensorflow as tf

MAX_LENGTH_ARTICLE = 400
MAX_LENGTH_SUMMARY = 100

tf.enable_eager_execution()
print(tf.__version__)

vocab = hp.load_saved_tensor('./vocab')
training_articles_tensor = hp.load_saved_tensor('./training_articles_tensor')
training_summaries_tensor = hp.load_saved_tensor('./training_summaries_tensor')
val_articles_tensor = hp.load_saved_tensor('./val_articles_tensor')
val_summaries_tensor = hp.load_saved_tensor('./val_summaries_tensor')

print("Tensors loaded")
print(training_articles_tensor.shape)
print(training_summaries_tensor.shape)
print(val_articles_tensor.shape)
print(val_summaries_tensor.shape)
print(len(vocab.word2idx))


# HPM
BUFFER_SIZE = len(training_articles_tensor)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_size = len(vocab.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((training_articles_tensor, training_summaries_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = hp.Encoder(units, embedding_dim, vocab_size, BATCH_SIZE)
decoder = hp.Decoder(units, embedding_dim, vocab_size, BATCH_SIZE)

# Optimizer
optimiser = tf.train.AdamOptimizer()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimiser,
                                 encoder=encoder,
                                 decoder=decoder)

#latest_ckp = tf.train.latest_checkpoint('./training_checkpoints')
#checkpoint.restore(latest_ckp)

## Training
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for batch, (inp, targ) in enumerate(dataset):
#        print("Batch: " + str(batch))
#        print("Input: " + inp)
#        print("Output: " + targ)
#        print("****************************************")
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)
#            print("enc_output:" + str(enc_output))
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([vocab.word2idx['<start>']] * BATCH_SIZE, 1)
#            print("dec_input:" + str(dec_input))

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
                loss += hp.loss_function(targ[:, t], predictions)
                print("Loss: " + str(loss))
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss / int(targ.shape[1])
        total_loss += batch_loss
        print("Batch Loss: " + str(batch_loss))
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(loss, variables)
        optimiser.apply_gradients(zip(gradients, variables))

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# Testing
#sentence = ""
#with open('002a083c3893b1fde734280b9eec28d428a02d2b.sent') as f:
#    sentence = f.read()
#hp.summarize(sentence, units, vocab, encoder, decoder, 400, 100)

