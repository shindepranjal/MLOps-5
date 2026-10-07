from ml_model import X, y, model


def test_dataset():
    assert len(X) > 0
    assert len(y) > 0


def test_features():
    assert X.shape[1] == 4


def test_model():
    assert model is not None


def test_prediction():
    prediction = model.predict([X[0]])

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]