# Define your model
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model
history = model.fit(X_train_temp, y_train_temp, epochs=30, validation_data=(X_val, y_val), verbose=1)


# Save model architecture as JSON file
from tensorflow.keras.models import model_from_json
model_json = model.to_json()
with open("lstm_model_autocomplete.json", "w") as json_file:
     json_file.write(model_json)


# Load model architecture from JSON file
# from tensorflow.keras.models import model_from_json

# with open("lstm_model_autocomplete.json", "r") as json_file:
#     loaded_model_json = json_file.read()

# # Create model from loaded architecture
# model = model_from_json(loaded_model_json)

# print("Model architecture loaded successfully from JSON file.")


