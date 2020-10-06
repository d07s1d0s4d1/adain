import tensorflow as tf
import itertools
from tensorflow.keras.optimizers import Adam
from loss_model import create_model
from generator import gen
from utils import plot


train_dir_name = 'images/train'
style_dir_name = 'images/validation'

N = 4
input_shape=(256, 256, 3)
lr = 1e-4
decay = 5e-5
style_loss_weight = 10
mode = 'simple'
in_memory = False
val_batch_size = 100

epochs = 160
batch_size = 8
steps_per_epoch = 1000

if __name__ == '__main__':
    loss_model = create_model(input_shape, N, coef=style_loss_weight)
    opt = Adam(learning_rate=lr, decay=decay)
    loss_model.compile(optimizer=opt)

    train_gen = gen(train_dir_name, input_shape[:2], batch_size, mode, in_memory)
    val_gen = gen(style_dir_name, input_shape[:2], val_batch_size, 'simple', in_memory)
    val_data = next(val_gen)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=loss_model)
    manager = tf.train.CheckpointManager(ckpt, '/tf_ckpts', max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for epoch in range(epochs):
        print('Epoch:', int(ckpt.step))
        
        history = loss_model.fit(train_gen, steps_per_epoch=steps_per_epoch, validation_data=val_data, verbose=1)

        if int(ckpt.step) % 1 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        ckpt.step.assign_add(1)

    (C, S), _ = val_data
    model = Model(inputs=loss_model.inputs, outputs=loss_model.get_layer('decoded').output)

    pred = model.predict([C, S])
    plot(itertools.chain(*zip(C, S, pred)), 6, 6, (24,24))

    model.save_weights('/weights/weights.h5')