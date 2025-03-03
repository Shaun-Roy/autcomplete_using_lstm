model = Sequential()
model.add(Embedding(total_words, 100))
model.add(Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(total_words, activation='softmax'))

adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor (could also be 'val_accuracy')
    patience=3,                 # Number of epochs with no improvement before stopping
    verbose=1,                  # Print messages when early stopping is triggered
    restore_best_weights=True   # Restore the best weights after training stops
)
# Train the model
history = model.fit(X_train_temp, y_train_temp, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping],verbose=1)


# Save model architecture as JSON file
from tensorflow.keras.models import model_from_json
model_json = model.to_json()
with open("lstm_model_autocomplete.json", "w") as json_file:
     json_file.write(model_json)
# Save weights after training
model.save_weights('model_weights.h5')


# Load the architecture from the JSON file
with open('lstm_model_autocomplete.json', 'r') as json_file:
    model_json = json_file.read()

# Recreate the model from the JSON
model = model_from_json(model_json)

# Load weights into the model
model.load_weights('model_weights.h5')



