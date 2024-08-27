from tensorflow import keras
import data_reader

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 50입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader(12) #윈도우의 크기 12개를 이용하여 다음 12개를 예측한다.

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Dense(32),
    keras.layers.Dense(14),
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", metrics=["mae"], loss="mse")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y), 
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(model(dr.test_X[:200]), dr.test_Y[:200], history)
