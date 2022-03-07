# 딥러닝에서 쓰이는 logit은 매우 간단합니다. 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부입니다.
# 예를 들어, 10개의 이미지를 분류하는 문제에서는 주로 softmax 함수를 사용하는데요.
# 이때, 모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, 이를 logit=False라고 표현할 수 있습니다.
# logit이 아니라 확률값이니까요. (이건 저만의 표현인 점을 참고해서 읽어주세요).
# 반대로 모델의 출력값이 sigmoid 또는 linear를 거쳐서 확률이 아닌 값이 나오게 된다면, logit=True라고 표현할 수 있습니다.
# 말 그대로 확률이 아니라 logit이니까요. 다시 코드로 돌아가보죠. 먼저 코드를 해석하려면 두 가지 가정이 필요합니다.
# (1) Loss Function이 CategoricalCrossEntropy이기 때문에 클래스 분류인 것을 알 수 있다.
# (2) output 배열은 모델의 출력값을 나타내며, softmax 함수를 거쳐서 나온 확률값이다.
# 이제 우리는 왜 2번 코드에서 from_logits=False를 사용했는지 알 수 있습니다. 문제에 알맞게 normalize된 상태이기 때문입니다
# (값을 전부 더해보면 1입니다, 확률을 예로 든거에요). 반대로 from_logits=True일 때는 output 배열의 값이 logit 상태가 아니기 때문에
# 우리가 생각한 값과 다른 값이 나오게 된 것입니다.

import tensorflow as tf

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
print(scce(y_true, y_pred).numpy())
