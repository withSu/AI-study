from tensorflow import keras  # 텐서플로우를 더 쉽게 사용할 수 있게 해줍니다.
import data_reader

# 에포크는 시간을 재는 단위입니다.
# AI에서는 학습을 몇 바퀴 돌릴 것인가를 결정합니다.
# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 20  # 예제 기본값은 20입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Dense(3),  # 퍼셉트론 세 개짜리 한 층입니다.
    keras.layers.Dense(128, activation="relu"),  # 퍼셉트론 128개짜리 한 층입니다.
    # activation은 활성화 함수입니다.
    keras.layers.Dense(3, activation='softmax')  # 퍼셉트론 3개짜리 한 층입니다.
])

# 인공신경망을 컴파일합니다.
# 모델을 압축하고 메모리에 올려서, 당장 컴퓨터가 사용할 수 있는 상태로 만듭니다.
# GPU의 VRAM에 올라가서 명령어를 기다립니다.

# metrics는 인공지능의 성능을 평가할 기준을 설정합니다.
# metrics=["accuracy"]는 정확도를 기준으로 평가할 것입니다.
# loss는 인공지능의 학습 방향을 설정하는 함수입니다.
model.compile(optimizer="adam", metrics=["accuracy"],
              loss="sparse_categorical_crossentropy")

""" 모델의 손실 함수(loss function)를 설정하는 코드입니다. 
 손실 함수는 신경망이 예측한 값과 실제 값 간의 차이를 계산하여, 
 모델이 얼마나 잘 학습하고 있는지 평가하는 데 사용됩니다. 

 sparse_categorical_crossentropy는 다중 클래스 분류 문제에서 사용되는 손실 함수로, 
 실제 값이 정수 레이블로 주어졌을 때 사용합니다. 이 손실 함수는 각 클래스에 대한 
 예측 확률을 계산하고, 실제 레이블과 비교하여 손실을 계산합니다.
 """

# 인공신경망을 학습시킵니다.
print("************ TRAINING START ************")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# train_X는 학습에 사용할 훈련 데이터(문제)입니다.
# train_Y는 학습에 사용할 레이블(정답)입니다.
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,

                    # validation_data는 채점에 사용할 데이터입니다.
                    validation_data=(dr.test_X, dr.test_Y),
                    # 학습이 완료된 것 같으면 조기 종료합니다.
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)
