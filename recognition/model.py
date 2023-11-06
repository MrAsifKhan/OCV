import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):
    def __init__(self, cfg, imgH=64, imgW=200):
        super(Model, self).__init__()
        tns_cfg = cfg["Transform"]
        featureExt_cfg = cfg["FeatureExtraction"]
        seqMod_cfg = cfg["SequenceModeling"]
        pred_cfg = cfg["Prediction"]
        input_channel = featureExt_cfg["input_channel"]
        output_channel = featureExt_cfg["output_channel"]
        self.stages = {
            "Trans": tns_cfg["name"],
            "Feat": featureExt_cfg["name"],
            "Seq": seqMod_cfg["name"],
            "Pred": pred_cfg["name"],
        }

        """ Transformation """

        if tns_cfg["name"] == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=tns_cfg["num_fiducial"],
                I_size=(imgH, imgW),
                I_r_size=(imgH, imgW),
                I_channel_num=input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if featureExt_cfg["name"] == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif featureExt_cfg["name"] == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(
                input_channel, output_channel
            )
        elif featureExt_cfg["name"] == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                input_channel, output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        hidden_size = seqMod_cfg["hidden_size"]
        if seqMod_cfg["name"] == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output, hidden_size, hidden_size
                ),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
            )
            self.SequenceModeling_output = hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        self.batch_max_length = pred_cfg["max_text_length"]
        num_class = pred_cfg["num_class"]
        if pred_cfg["name"] == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        elif pred_cfg["name"] == "Attn":
            self.Prediction = Attention(
                self.SequenceModeling_output, hidden_size, num_class
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages["Trans"] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2)
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.batch_max_length,
            )

        return prediction
