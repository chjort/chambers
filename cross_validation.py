import glob
import gc
import os
import json
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from chambers.data.dataset import Dataset, load_train_test_val, parse_class_dict_csv
from chambers.augmentations import augmentations_to_function
from chambers.callbacks import Logger
from chambers.losses import cross_entropy_with_dice_coef_no_bg
from chambers.metrics import precision, recall, f1, mean_class_iou, particle_iou
from utils.utils import get_timestamp, onehot_to_rgb
from chambers.models import get_model


def kfold_cv_generalization(model_name, cv_dir, input_shape, epochs, batch_size, train_aug, predict_test=True):
    # K-Fold Cross-validation (Generalization performance)

    pred_graph = tf.Graph()
    with pred_graph.as_default():
        pred_input_placeholder = tf.placeholder(tf.string, shape=None)
        img_bytes = tf.read_file(pred_input_placeholder)
        img = tf.image.decode_image(img_bytes, channels=3)
        img = tf.cast(img / 255, dtype=tf.float32)
        pred_input_img = tf.expand_dims(img, axis=0)
    sess = tf.Session(graph=pred_graph)

    aug_fn = augmentations_to_function(train_aug)
    class_names, class_labels = parse_class_dict_csv(os.path.join(cv_dir, "class_dict.csv"))
    fold_dirs = sorted(glob.glob(os.path.join(cv_dir, "*x*")))
    cv_dict = {"folds": {},
               "cv": {"metrics": {}, "N": 0}}

    timestamp = get_timestamp()
    results_dir = os.path.join(os.getcwd(), "results")
    cv_res_dir = os.path.join(results_dir, *[os.path.basename(cv_dir), "gen_" + timestamp])
    for k, fold_dir in enumerate(fold_dirs):
        print("Fold {}: {}".format(k, fold_dir))
        sets = load_train_test_val(fold_dir)
        train_set = Dataset(x=sets["train"][0],
                            y=sets["train"][1],
                            class_labels=class_labels,
                            augmentation=aug_fn,
                            one_hot=True,
                            n_threads=4
                            )

        test_set = Dataset(x=sets["test"][0],
                           y=sets["test"][1],
                           class_labels=class_labels,
                           augmentation=aug_fn,
                           one_hot=True,
                           n_threads=4
                           )
        train_set.set_batch(batch_size)
        test_set.set_batch(batch_size)

        # create model
        model = get_model(model_name,
                          input_shape=input_shape,
                          num_classes=len(class_labels)
                          )

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=0.000001, amsgrad=True)
        model.compile(optimizer=optimizer,
                      loss=cross_entropy_with_dice_coef_no_bg,
                      metrics=[precision, recall, f1, mean_class_iou, particle_iou]
                      )

        # setup directories and paths
        fold_result_dir = os.path.join(cv_res_dir,
                                       *[os.path.basename(fold_dir),
                                         model_name])
        config_dir = os.path.join(fold_result_dir, "configs")
        checkpoint_dir = os.path.join(fold_result_dir, "checkpoints")
        log_dir = os.path.join(fold_result_dir, "logs")
        os.makedirs(config_dir)
        os.makedirs(checkpoint_dir)
        os.makedirs(log_dir)

        # save argument settings
        attributes = {"model_name": model_name,
                      "dataset": cv_dir,
                      "augmentations": train_aug,
                      }
        with open(os.path.join(config_dir, "settings.json"), "w") as f:
            f.write(json.dumps(attributes, sort_keys=True, indent=4))

        # save initial weights
        model.save_weights(os.path.join(checkpoint_dir, "initial_weights.h5"))

        # callbacks
        logger_cb = Logger(log_dir, config_dir, plot=False)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, )

        patience = 10
        lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_particle_iou",
                                                     factor=0.2,
                                                     patience=patience,
                                                     min_lr=1e-7,
                                                     mode="max",
                                                     verbose=2
                                                     )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_weights.h5"),
            monitor="val_particle_iou",
            mode="max",
            period=1,
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )

        callbacks = [checkpoint_cb, logger_cb, tensorboard_cb, lr_cb]

        # train
        train_hist = model.fit(train_set.dataset,
                               validation_data=test_set.dataset,
                               epochs=epochs,
                               steps_per_epoch=train_set.n // batch_size,
                               validation_steps=test_set.n,
                               callbacks=callbacks,
                               verbose=2
                               )

        # save metrics
        cv_dict["folds"]["fold" + str(k)] = {"metrics": train_hist.history,
                                             "N_train": train_set.n,
                                             "N_test": test_set.n
                                             }
        cv_dict["folds"]["fold" + str(k)]["metrics"]["lr"] = [float(lr) for lr in train_hist.history["lr"]]
        cv_dict["cv"]["N"] = train_set.n + test_set.n

        # clear
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        if predict_test:
            # load best model
            model = get_model(model_name,
                              input_shape=(960, 1290, 3),
                              num_classes=len(class_labels)
                              )
            model.load_weights(os.path.join(checkpoint_dir, "best_weights.h5"))
            optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=0.000001, amsgrad=True)
            model.compile(optimizer=optimizer,
                          loss=cross_entropy_with_dice_coef_no_bg,
                          metrics=[precision, recall, f1, mean_class_iou, particle_iou]
                          )

            # predict
            full_img_dir = os.path.join(cv_dir, *["fold"+str(k), "test"])
            full_label_dir = os.path.join(cv_dir, *["fold" + str(k), "test_labels"])
            save_dir = os.path.join(fold_result_dir, "test_predictions")
            os.makedirs(save_dir)
            for i, fname in enumerate(os.listdir(full_img_dir)):
                x = sess.run(pred_input_img, feed_dict={pred_input_placeholder: os.path.join(full_img_dir, fname)})
                pred = model.predict(x, steps=1)[0]
                pred_rgb = onehot_to_rgb(pred, class_labels)

                im = Image.fromarray(pred_rgb)
                file_name_without_ext = ".".join(fname.split(".")[:-1])
                im.save(os.path.join(save_dir, file_name_without_ext + "_pred.png"))
                shutil.copyfile(os.path.join(full_img_dir, fname),
                                os.path.join(save_dir, fname))
                shutil.copyfile(os.path.join(full_label_dir, fname),
                                os.path.join(save_dir, file_name_without_ext + "_label.png"))

            # clear
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    sess.close()

    for metric in cv_dict["folds"]["fold0"]["metrics"].keys():
        if "val_" in metric:
            mean_metric = np.zeros(len(cv_dict["folds"]["fold0"]["metrics"][metric]))
            for fold in cv_dict["folds"].keys():
                weight = cv_dict["folds"][fold]["N_test"] / cv_dict["cv"]["N"]
                value = cv_dict["folds"][fold]["metrics"][metric]
                mean_metric = np.add(mean_metric, weight * np.array(value))
            cv_dict["cv"]["metrics"][metric] = mean_metric.tolist()

    with open(os.path.join(cv_res_dir, "train_history.json"), "w") as f:
        f.write(json.dumps(cv_dict, sort_keys=True, indent=4))

    # plot metrics
    fig = plt.figure()
    for metric in cv_dict["cv"]["metrics"].keys():
        if "val" in metric:
            plt.plot(cv_dict["cv"]["metrics"][metric], label=model_name)
            plt.title("{}-fold cross-validation: {}}".format(len(fold_dirs), metric))
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(os.path.join(cv_res_dir, "{}.png".format(metric)))
            fig.clf()


def kfold_cv_selection(model_names, cv_dir, input_shape, epochs, batch_size, train_aug, predict_test=True):
    # K-Fold Cross-validation (Model selection)

    pred_graph = tf.Graph()
    with pred_graph.as_default():
        pred_input_placeholder = tf.placeholder(tf.string, shape=None)
        img_bytes = tf.read_file(pred_input_placeholder)
        img = tf.image.decode_image(img_bytes, channels=3)
        img = tf.cast(img / 255, dtype=tf.float32)
        pred_input_img = tf.expand_dims(img, axis=0)
    sess = tf.Session(graph=pred_graph)

    aug_fn = augmentations_to_function(train_aug)
    class_names, class_labels = parse_class_dict_csv(os.path.join(cv_dir, "class_dict.csv"))
    fold_dirs = sorted(glob.glob(os.path.join(cv_dir, "*x*")))
    cv_dict = {"folds": {},
               "cv": {"N": 0, "models": {}}}

    timestamp = get_timestamp()
    results_dir = os.path.join(os.getcwd(), "results")
    cv_res_dir = os.path.join(results_dir, *[os.path.basename(cv_dir), "sel_" + timestamp])
    for k, fold_dir in enumerate(fold_dirs):
        print("Fold {}: {}".format(k, fold_dir), flush=True)
        sets = load_train_test_val(fold_dir)
        cv_dict["folds"]["fold" + str(k)] = {"N_train": len(sets["train"][0]),
                                             "N_test": len(sets["test"][0])}
        cv_dict["cv"]["N"] = len(sets["train"][0]) + len(sets["test"][0])
        for model_name in model_names:
            print(model_name, flush=True)
            train_set = Dataset(x=sets["train"][0],
                                y=sets["train"][1],
                                class_labels=class_labels,
                                augmentation=aug_fn,
                                one_hot=True,
                                n_threads=4
                                )

            test_set = Dataset(x=sets["test"][0],
                               y=sets["test"][1],
                               class_labels=class_labels,
                               augmentation=aug_fn,
                               one_hot=True,
                               n_threads=4
                               )
            train_set.set_batch(batch_size)
            test_set.set_batch(batch_size)

            # create model
            model = get_model(model_name,
                              input_shape=input_shape,
                              num_classes=len(class_labels)
                              )
            if model_name == "ResNet50_FPN":
                wpath = "/home/ch/Dropbox/DTU/Masters/code/project/results/updated_cv4/gen_2019-06-13_17-49-46/{}_256x256/ResNet50_FPN/checkpoints/initial_weights.h5".format("fold"+str(k))
                print("Loading", wpath, flush=True)
                model.load_weights(wpath)
            elif model_name == "DeepLabV3+ Mobile":
                wpath = "/home/ch/Dropbox/DTU/Masters/code/project/results/updated_cv4/gen_2019-06-14_16-53-25/{}_256x256/DeepLabV3+ Mobile/checkpoints/initial_weights.h5".format("fold"+str(k))
                print("Loading", wpath, flush=True)
                model.load_weights(wpath)
            else:
                wpath = "/home/ch/Dropbox/DTU/Masters/code/project/results/updated_cv4/sel_2019-06-12_18-27-15/{}_256x256/{}/checkpoints/initial_weights.h5".format("fold"+str(k), model_name)
                print("Loading", wpath, flush=True)
                model.load_weights(wpath)

            optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=0.000001, amsgrad=True)
            model.compile(optimizer=optimizer,
                          loss=cross_entropy_with_dice_coef_no_bg,
                          metrics=[precision, recall, f1, mean_class_iou, particle_iou]
                          )

            # setup directories and paths
            fold_result_dir = os.path.join(cv_res_dir,
                                           *[os.path.basename(fold_dir),
                                             model_name])
            config_dir = os.path.join(fold_result_dir, "configs")
            checkpoint_dir = os.path.join(fold_result_dir, "checkpoints")
            log_dir = os.path.join(fold_result_dir, "logs")
            os.makedirs(config_dir)
            os.makedirs(checkpoint_dir)
            os.makedirs(log_dir)

            # save argument settings
            attributes = {"model_name": model_name,
                          "dataset": cv_dir,
                          "augmentations": train_aug,
                          "fold": k
                          }
            with open(os.path.join(config_dir, "settings.json"), "w") as f:
                f.write(json.dumps(attributes, sort_keys=True, indent=4))

            # save initial weights
            model.save_weights(os.path.join(checkpoint_dir, "initial_weights.h5"))

            # callbacks
            logger_cb = Logger(log_dir, config_dir, plot=False)
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, )

            patience = 10
            lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_particle_iou",
                                                         factor=0.2,
                                                         patience=patience,
                                                         min_lr=1e-7,
                                                         mode="max",
                                                         verbose=2
                                                         )
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "best_weights.h5"),
                monitor="val_particle_iou",
                mode="max",
                period=1,
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )

            callbacks = [checkpoint_cb, logger_cb, tensorboard_cb, lr_cb]

            train_hist = model.fit(train_set.dataset,
                                   validation_data=test_set.dataset,
                                   epochs=epochs,
                                   steps_per_epoch=train_set.n // batch_size,
                                   validation_steps=test_set.n,
                                   callbacks=callbacks,
                                   verbose=2
                                   )

            cv_dict["folds"]["fold" + str(k)][model_name] = train_hist.history
            cv_dict["folds"]["fold" + str(k)][model_name]["lr"] = [float(lr) for lr in train_hist.history["lr"]]

            del model
            tf.keras.backend.clear_session()
            gc.collect()

            if predict_test:
                # load best model
                model = get_model(model_name,
                                  input_shape=(960, 1290, 3),
                                  num_classes=len(class_labels)
                                  )
                model.load_weights(os.path.join(checkpoint_dir, "best_weights.h5"))
                optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=0.000001, amsgrad=True)
                model.compile(optimizer=optimizer,
                              loss=cross_entropy_with_dice_coef_no_bg,
                              metrics=[precision, recall, f1, mean_class_iou, particle_iou]
                              )

                # predict
                full_img_dir = os.path.join(cv_dir, *["fold" + str(k), "test"])
                full_label_dir = os.path.join(cv_dir, *["fold" + str(k), "test_labels"])
                save_dir = os.path.join(fold_result_dir, "test_predictions")
                os.makedirs(save_dir)
                for i, fname in enumerate(os.listdir(full_img_dir)):
                    x = sess.run(pred_input_img, feed_dict={pred_input_placeholder: os.path.join(full_img_dir, fname)})
                    pred = model.predict(x, steps=1)[0]
                    pred_rgb = onehot_to_rgb(pred, class_labels)

                    im = Image.fromarray(pred_rgb)
                    file_name_without_ext = ".".join(fname.split(".")[:-1])
                    im.save(os.path.join(save_dir, file_name_without_ext + "_pred.png"))
                    shutil.copyfile(os.path.join(full_img_dir, fname),
                                    os.path.join(save_dir, fname))
                    shutil.copyfile(os.path.join(full_label_dir, fname),
                                    os.path.join(save_dir, file_name_without_ext + "_label.png"))

                # clear
                del model
                tf.keras.backend.clear_session()
                gc.collect()
    sess.close()

    # compute cross-validation metrics across folds
    for model_name in model_names:
        cv_dict["cv"]["models"][model_name] = {}
        for metric in cv_dict["folds"]["fold0"][model_name].keys():
            if "val_" in metric:
                metric_folds = np.zeros(len(cv_dict["folds"]["fold0"][model_name][metric]))
                for fold in cv_dict["folds"].keys():
                    weight = cv_dict["folds"][fold]["N_test"] / cv_dict["cv"]["N"]
                    values = cv_dict["folds"][fold][model_name][metric]
                    metric_folds = np.add(metric_folds, weight * np.array(values))
                cv_dict["cv"]["models"][model_name][metric] = metric_folds.tolist()

    with open(os.path.join(cv_res_dir, "train_history.json"), "w") as f:
        f.write(json.dumps(cv_dict, sort_keys=True, indent=4))

    # plot metrics
    first_model = list(cv_dict["cv"]["models"].keys())[0]
    fig = plt.figure()
    for metric in cv_dict["cv"]["models"][first_model].keys():
        if "val" in metric:
            for model in sorted(cv_dict["cv"]["models"].keys()):
                plt.plot(cv_dict["cv"]["models"][model][metric], label=model)

            #plt.title("{}-fold cross-validation: {}".format(len(fold_dirs), metric))
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(os.path.join(cv_res_dir, "{}.png".format(metric)))
            fig.clf()
