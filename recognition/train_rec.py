import os
import time
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import (
    CTCLabelConverter,
    AttnLabelConverter,
    Averager,
)
from dataset import AlignCollate, RecognitionDataset
from model import Model
from test import validation

import configuration as cf
import yaml
from yaml.loader import SafeLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg):
    """ dataset preparation """
    global_cfg = cfg["Global"]
    optimizer_cfg = cfg["Optimizer"]
    architecture_cfg = cfg["Architecture"]
    train_cfg = cfg["Train"]
    val_cfg = cfg["Eval"]
    train_dataset_cfg = train_cfg["dataset"]
    transforms_cfg = train_dataset_cfg["transforms"]

    for operator in transforms_cfg:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if op_name == "Resize":
            for key, value in param.items():
                if key == "img_H":
                    imgH = int(value)
                if key == "img_W":
                    imgW = int(value)
                if key == "keep_ratio_with_pad":
                    pad = value
    alignCollate = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=pad)
    train_dataset = RecognitionDataset(train_df, train_dataset_cfg)

    train_loader_cfg = train_cfg["loader"]
    train_batch_size = train_loader_cfg["batch_size"]
    train_shuffle = train_loader_cfg["shuffle"]
    train_drop_last = train_loader_cfg["drop_last"]
    train_workers = train_loader_cfg["workers"]

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        drop_last=train_drop_last,
        num_workers=train_workers,
        collate_fn=alignCollate,
        pin_memory=True,
    )

    validation_dataset = RecognitionDataset(validation_df, train_dataset_cfg)

    val_loader_cfg = val_cfg["loader"]
    val_batch_size = val_loader_cfg["batch_size"]
    val_shuffle = val_loader_cfg["shuffle"]
    val_drop_last = val_loader_cfg["drop_last"]
    val_workers = val_loader_cfg["workers"]

    validation_data_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        drop_last=val_drop_last,
        num_workers=val_workers,
        collate_fn=alignCollate,
        pin_memory=True,
    )

    """ model configuration """
    prediction_cfg = architecture_cfg["Prediction"]

    if prediction_cfg["name"] == "CTC":
        converter = CTCLabelConverter(prediction_cfg["character"])
    else:
        converter = AttnLabelConverter(prediction_cfg["character"])
    architecture_cfg["Prediction"]["num_class"] = len(converter.character)
    model = Model(architecture_cfg, imgH, imgW)

    # weight initialization
    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.kaiming_normal_(param)
        except Exception:  # for batchnorm.
            if "weight" in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU

    model = torch.nn.DataParallel(model).to(device)

    model.train()
    saved_epoch = 0
    start_epoch = 0
    epochs = global_cfg["epochs"]
    use_gpu = global_cfg["use_gpu"]
    pretrained_model_path = global_cfg["pretrained_model"]
    if pretrained_model_path is not None:
        print(f"loading pretrained model from {pretrained_model_path}")
        if global_cfg["FT"]:
            if use_gpu:
                checkpoint = torch.load(pretrained_model_path)
            else:
                checkpoint = torch.load(
                    pretrained_model_path, map_location=torch.device("cpu")
                )
            checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if (k in model.state_dict().keys())
                and (model.state_dict()[k].shape == checkpoint[k].shape)
            }
            for name in model.state_dict().keys():
                if name in checkpoint.keys():
                    model.state_dict()[name].copy_(checkpoint[name])
        else:
            if use_gpu:
                model.load_state_dict(torch.load(pretrained_model_path))
            else:
                model.load_state_dict(
                    torch.load(pretrained_model_path, map_location=torch.device("cpu"))
                )

    checkpoint_path = global_cfg["checkpoint_path"]
    if checkpoint_path is not None:

        try:
            if use_gpu:
                model.load_state_dict(torch.load(checkpoint_path))
            else:
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=torch.device("cpu"))
                )
            saved_epoch = int(checkpoint_path.split("_")[-1].split(".")[0])
            start_epoch = saved_epoch
            print(f"resuming training from epoch {saved_epoch}...")
        except Exception:
            pass

    """ setup loss """
    pred_cfg = architecture_cfg["Prediction"]
    if pred_cfg["name"] == "CTC":
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
            device
        )  # ignore [GO] token = ignore index 0
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print("Trainable params num : ", sum(params_num))

    """ setup optimizer """
    optimizers_list = ["Adam", "Adadelta"]
    assert (
        optimizer_cfg["to_use"] in optimizers_list
    ), "Optimizer provided in to_use is not a valid optimizer"
    if optimizer_cfg["to_use"] == "Adam":
        beta1 = optimizer_cfg["Adam"]["beta1"]
        beta2 = optimizer_cfg["Adam"]["beta2"]
        lr = optimizer_cfg["Adam"]["learning_rate"]
        optimizer = optim.Adam(filtered_parameters, lr=lr, betas=(beta1, beta2))
    elif optimizer_cfg["to_use"] == "Adadelta":
        rho = float(optimizer_cfg["Adadelta"]["rho"])
        eps = float(optimizer_cfg["Adadelta"]["eps"])
        lr = optimizer_cfg["Adadelta"]["learning_rate"]
        optimizer = optim.Adadelta(filtered_parameters, lr=lr, rho=rho, eps=eps)
    batch_max_length = pred_cfg["max_text_length"]
    grad_clip = global_cfg["grad_clip"]
    valInterval = global_cfg["validation_interval"]
    save_model_dir = global_cfg["save_model_dir"]
    sensitive = pred_cfg["case_sensitive"]
    data_filtering_off = pred_cfg["data_filtering_off"]
    save_interval = global_cfg["save_interval"]

    """ start training """
    best_accuracy = -1
    best_norm_ED = -1  # normalised edit distance
    df_len = len(train_df)
    iterations = int(df_len / train_batch_size)
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        print(f"Running epoch {epoch+1} / {epochs}")
        print("-" * 100)
        for index, (image_tensor, label) in enumerate(train_data_loader):
            image_tensors = image_tensor
            labels = label
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=batch_max_length)
            batch_size = image.size(0)

            if pred_cfg["name"] == "CTC":
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

            else:
                preds = model(image, text[:, :-1])
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)
            elapsed_time = time.time() - start_time
            print(
                f"Iteration {index+1} / {iterations} \t Training loss: {loss_avg.val():0.5f} \t \
                Elapsed_time: {elapsed_time:0.5f}"
            )
        print("-" * 100)
        print("-" * 100)
        if (epoch + 1) % valInterval == 0 or epoch == 0:
            elapsed_time = time.time() - start_time
            model.eval()
            with torch.no_grad():
                (
                    validation_loss,
                    current_accuracy,
                    current_norm_ED,
                    preds,
                    confidence_score,
                    labels,
                    infer_time,
                    length_of_data,
                ) = validation(
                    model,
                    criterion,
                    validation_data_loader,
                    converter,
                    architecture_cfg,
                    sensitive,
                    data_filtering_off,
                )
            model.train()
            loss_log = f"[{epoch+1}/{epochs}] Train loss: {loss_avg.val():0.5f}, Valid loss: {validation_loss:0.5f}, \
                        Elapsed_time: {elapsed_time:0.5f}"
            loss_avg.reset()
            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, \
                {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                print(f"Saving model with new best accuracy of {best_accuracy}")
                save_name = save_model_dir + "/best_accuracy.pth"
                torch.save(model.state_dict(), save_name)
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
                save_name = save_model_dir + "/best_norm_ED.pth"
                torch.save(model.state_dict(), save_name)
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

            loss_model_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
            print(loss_model_log)

            # show some predicted results
            dashed_line = "-" * 100
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
            for gt, pred, confidence in zip(
                labels[:20], preds[:20], confidence_score[:20]
            ):
                if pred_cfg["name"] == "Attn":
                    gt = gt[: gt.find("[s]")]
                    pred = pred[: pred.find("[s]")]

                predicted_result_log += (
                    f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                )
            predicted_result_log += f"{dashed_line}"
            print(predicted_result_log)
            print("-" * 100)

        if (epoch + 1) % (save_interval) == 0:
            print(f"Saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), f"./{save_model_dir}/epoch_{epoch+1}.pth")

        if (epoch + 1) == epochs:
            print("=" * 100)
            print("end of training!")
            print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration_file",
        default="./configs/fpc_finetune.yml",
        help="configuration file path",
    )
    opt = parser.parse_args()
    print("loading configuration file")
    path = opt.configuration_file

    with open(path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)

    beeyard_cfg = cfg["Train"]["dataset"]["beeyard"]
    train_split = cfg["Train"]["dataset"]["train_split"]
    df_random_state = cfg["Train"]["dataset"]["df_random_state"]
    save_df_as_csv = cfg["Train"]["dataset"]["save_df_as_csv"]
    train_df, validation_df = get_df(
         beeyard_cfg, train_split, df_random_state, save_df_as_csv
    )
    print("Total numbe of Training images :", (len(train_df)))
    print("Total numbe of validation images :", (len(validation_df)))
    print("-" * 80)

    os.makedirs(f'./{cfg["Global"]["save_model_dir"]}', exist_ok=True)

    """ Seed and GPU setting """

    random.seed(cfg["Global"]["manualSeed"])
    np.random.seed(cfg["Global"]["manualSeed"])
    torch.manual_seed(cfg["Global"]["manualSeed"])
    if device != "cpu":
        torch.cuda.manual_seed(cfg["Global"]["manualSeed"])
        cudnn.benchmark = True
        cudnn.deterministic = True
        if cfg["Global"]["num_gpu"] is None:
            cfg["Global"]["num_gpu"] = torch.cuda.device_count()

    if cfg["Global"]["num_gpu"] > 1:
        print("------ Use multi-GPU setting ------")
        print(
            "if you stuck too long time with multi-GPU setting, try to set --workers 0"
        )
        cfg["Train"]["loader"]["workers"] = (
            cfg["Train"]["loader"]["workers"] * cfg["Global"]["num_gpu"]
        )
        cfg["Train"]["loader"]["batch_size"] = (
            cfg["Train"]["loader"]["batch_size"] * cfg["Global"]["num_gpu"]
        )

    train(cfg)
