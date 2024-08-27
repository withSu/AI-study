from tensorflow import keras
import data_reader

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 50입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
model = keras.Sequential([
    #8983: 단어 집합의 크기(모델이 처리할 수 있는 고유한 단어의 수).
    #128: 각 단어를 나타내는 임베딩 벡터의 차원 (단어의 의미를 표현할 벡터의 길이).
    keras.layers.Embedding(8983, 128), #문장들을 목록으로 임베딩해준다.
    keras.layers.GlobalAveragePooling1D(), #1차원으로 압축한다.
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", metrics=['accuracy'],
              loss="binary_crossentropy")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)
