import utils as hp


training_articles, training_summaries = hp.load_data('./cnn/sample_training')
val_articles, val_summaries = hp.load_data('./cnn/validation')
test_articles, test_summaries = hp.load_data('./cnn/test')

training_articles_preprocessed, training_summaries_preprocessed = hp.get_preprocessed_articles(training_articles,
                                                                                            training_summaries)
print("Preprocessed Training articles")
val_articles_preprocessed, val_summaries_preprocessed = hp.get_preprocessed_articles(val_articles, val_summaries)
print("Preprocessed Val articles")

vocab = hp.build_vocab(training_articles_preprocessed + val_articles_preprocessed,
                    training_summaries_preprocessed + val_summaries_preprocessed)

print("Built Vocab")

training_articles_tensor, training_summaries_tensor = hp.build_tensors(vocab, training_articles_preprocessed, training_summaries_preprocessed)
training_articles_tensor, training_summaries_tensor = hp.pad_tensors(training_articles_tensor, training_summaries_tensor)

val_articles_tensor, val_summaries_tensor = hp.build_tensors(vocab, val_articles_preprocessed, val_summaries_preprocessed)
val_articles_tensor, val_summaries_tensor = hp.pad_tensors(val_articles_tensor, val_summaries_tensor)

hp.save_tensor(training_articles_tensor, './saved_tensors/training_articles_tensor')
hp.save_tensor(training_summaries_tensor, './saved_tensors/training_summaries_tensor')
hp.save_tensor(val_articles_tensor, './saved_tensors/val_articles_tensor')
hp.save_tensor(val_summaries_tensor, './saved_tensors/val_summaries_tensor')
hp.save_tensor(vocab, './saved_tensors/vocab')

print("Tensors Saved")
