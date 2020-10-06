from loss_model import create_model
from generator import gen
import itertools
from utils import plot
from tensorflow.keras.models import Model


if __name__ == "__main__":
    loss_model = create_model()
    loss_model.load_weights('weights/weights.h5')
    model = Model(inputs=loss_model.inputs, outputs=loss_model.get_layer('decoded').output)
    model.load_weights('weights/weights.h5')

    g = gen('images/validation', batch_size=30)
    (C, S), _ = next(g)
    pred = model.predict([C, S])

    plot(itertools.chain(*zip(C, S, pred)), 6, 6, (24,24))
