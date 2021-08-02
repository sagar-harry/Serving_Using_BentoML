import torch
import bentoml
from bentoml.handlers import ImageHandler

from loading_datasets import transforms
import bentoml
import bentoml.service.artifacts

from bentoml.frameworks.pytorch import PytorchModelArtifact

classes = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

device = torch.device('cpu')   

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact("mnist_pred")])

class numberClassifier(bentoml.BentoService):
    @bentoml.api(ImageHandler)
    def predict(self, img):
        img = transforms(img)
        self.artifacts.model.eval()
        outputs = self.artifacts.model.predict(img)
        _, idxs = outputs.topk(1)
        idx = idxs.squeeze().item()
        return classes[idx]
        