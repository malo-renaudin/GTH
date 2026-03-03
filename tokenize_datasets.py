from src.language_models.dictionary_corpus import Dictionary, tokenize
import os
import pickle 

vocab = "data/english_data/vocab.txt"

main_train = "data/english_data/train.txt"
main_test = "data/english_data/test.txt"
main_valid = "data/english_data/valid.txt"
test_orc = "test_orc.txt"
train_freq_8 = "train_freq_8.txt"
train_freq_16 = "train_freq_16.txt"
train_freq_32 = "train_freq_32.txt"

dict = Dictionary("data/english_data")


tokenized_train = tokenize(dict, main_train)
with open('train_tokenized.pkl', 'wb') as f:
	pickle.dump(tokenized_train, f)
print("Tokenized train set")
tokenized_test = tokenize(dict, main_test)
with open('test_tokenized.pkl', 'wb') as f:
    pickle.dump(tokenized_test, f)
print("Tokenized test set")
tokenized_valid = tokenize(dict, main_valid)
with open('valid_tokenized.pkl', 'wb') as f:
	pickle.dump(tokenized_valid, f)
print("Tokenized valid set")
# tokenized_test_orc = tokenize(dict, test_orc)
# with open('test_orc_tokenized.pkl', 'wb') as f:
#     pickle.dump(tokenized_test_orc, f)
tokenized_train_freq_8 = tokenize(dict, train_freq_8)
with open('train_freq_8_tokenized.pkl', 'wb') as f:
    pickle.dump(tokenized_train_freq_8, f)
print("Tokenized train_freq_8 set")
tokenized_train_freq_16 = tokenize(dict, train_freq_16)
with open('train_freq_16_tokenized.pkl', 'wb') as f:
    pickle.dump(tokenized_train_freq_16, f)
print("Tokenized train_freq_16 set")
tokenized_train_freq_32 = tokenize(dict, train_freq_32)
with open('train_freq_32_tokenized.pkl', 'wb') as f:
    pickle.dump(tokenized_train_freq_32, f)
print("Tokenized train_freq_32 set")