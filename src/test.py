import os
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader
from src.model import MLP
from src.datasets import Dataset, DefectViews
from src.config_parser import Config
from src.tools import Utils, Logger, TBWriter


class Tester:

    def __init__(self, testset: Dataset, model: MLP, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.testset = testset
        self.model = model.to(self.device)

        try:
            self.model_path = Utils.validate_path(model_path)
            self.model.load_state_dict(torch.load(self.model_path))
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf} Quitting")
            sys.exit(-1)

        # tensorboard
        self.writer = TBWriter.instance().get_writer()

        Logger.instance().debug(f"device: {self.device.type}, model at: {os.path.abspath(os.path.realpath(self.model_path))}")

    def test(self, config: Config):
        self.model.eval()
        testloader = DataLoader(self.testset, batch_size=config.batch_size)

        prcurve_labels = []
        prcurve_predic = []
        
        with torch.no_grad():
            tot_samples = 0
            tot_correct = 0
            for images, labels in testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                y_pred = self.model(images)
                
                # max returns (value, index)
                top_pred_val, top_pred_idx = torch.max(y_pred.data, 1)
                n_correct = top_pred_idx.eq(labels.view_as(top_pred_idx)).sum()
                
                # accuracy
                tot_samples += labels.size(0)
                tot_correct += n_correct

                # precision-recall curve
                prcurve_labels.extend(labels)
                prcurve_predic.append(y_pred)

            acc = tot_correct / tot_samples
            Logger.instance().debug(f"Test accuracy on {len(self.testset)} images: {acc:.3f}")

        # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve
        # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#assessing-trained-models-with-tensorboard
        test_probs = torch.cat([torch.stack(tuple(batch)) for batch in prcurve_predic])
        for defect_class in config.defect_class:
            truth = list(map(lambda x: x == DefectViews.label_to_idx[defect_class], prcurve_labels))
            probs = test_probs[:, DefectViews.label_to_idx[defect_class]]
            self.writer.add_pr_curve(defect_class, torch.Tensor(truth), probs.cpu(), global_step=0, num_thresholds=1000)
            self.writer.close()

    def one_shot(self, image):
        pass