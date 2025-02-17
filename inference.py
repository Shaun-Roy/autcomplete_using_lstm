def predict_top_five_words(model, tokenizer, input_text):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    top_five_indexes = np.argsort(predicted[0])[::-1][:5]
    top_five_words = []
    for index in top_five_indexes:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_five_words.append(word)
                break
    return top_five_words

#test 1
input_text = input("enter your input here")
predict_top_five_words(model,tokenizer,input_text)

#test 2
input_text = input("enter your input here")
predict_top_five_words(model,tokenizer,input_text)

#test 3
input_text = input("enter your input here")
predict_top_five_words(model,tokenizer,input_text)