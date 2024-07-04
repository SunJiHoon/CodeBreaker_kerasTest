import pandas as pd
import numpy as np
import tensorflow as tf

def create_student_score_prediction_model():
    # 학생들 이름
    students = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    # 각 학생의 성적 데이터
    data = {
        'student': students,
        'midterm_1st': [85, 78, 92, 65, 88, 76, 95, 83, 71, 90],
        'final_1st': [87, 80, 94, 67, 85, 79, 96, 82, 74, 91],
        'midterm_2nd': [88, 81, 90, 70, 87, 78, 94, 84, 73, 89],
        'final_2nd': [90, 82, 93, 72, 86, 80, 97, 85, 75, 92],
    }

    df = pd.DataFrame(data)

    # 데이터프레임에서 입력과 출력 분리
    X = df[['midterm_1st', 'final_1st', 'midterm_2nd']].values
    y = df['final_2nd'].values

    # 학습용 데이터와 테스트용 데이터로 분리
    test_size = 0.2
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    # 텐서플로우 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X_train, y_train, epochs=100, validation_split=0.2)

    # 모델 저장
    model.save('student_score_prediction_model.h5')

    # 예측
    predictions = model.predict(X_test)

    # 결과 출력
    print("Predictions:", predictions)
    print("Actual:", y_test)

    # 예측 결과와 실제 값을 데이터프레임으로 저장
    results = pd.DataFrame({
        'Predictions': predictions.flatten(),
        'Actual': y_test
    })
    results.to_csv('predictions_vs_actual.csv', index=False)

def predict_student_score():
    # 저장된 모델 로드
    model = tf.keras.models.load_model('student_score_prediction_model.h5')

    # 사용자로부터 입력 받기
    midterm_1st = float(input("1학기 중간고사 성적을 입력하세요: "))
    final_1st = float(input("1학기 기말고사 성적을 입력하세요: "))
    midterm_2nd = float(input("2학기 중간고사 성적을 입력하세요: "))

    # 입력 데이터를 numpy 배열로 변환
    X_new = np.array([[midterm_1st, final_1st, midterm_2nd]])

    # 예측 수행
    prediction = model.predict(X_new)

    # 예측 결과 출력
    print(f"예측된 2학기 기말고사 성적: {prediction[0][0]}")

if __name__ == '__main__':
    print("1번: 모델 생성하기, 2번 모델을 이용하여 예측하기")
    num = int(input())
    if num == 1:
        create_student_score_prediction_model()
    elif num == 2:
        predict_student_score()
    else:
        print("잘못된 숫자가 입력되었습니다.")
