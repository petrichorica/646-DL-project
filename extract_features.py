from transformers import AutoProcessor, AutoModel, SiglipVisionModel
import torch
import torch.nn as nn

class mySigLipModel(nn.Module):
    def __init__(self):
        super(mySigLipModel, self).__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.shape = 768

    def get_image_embedding(self, frame):
        image = frame.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy()
    
class mySigLipVisionModel(nn.Module):
    def __init__(self):
        super(mySigLipVisionModel, self).__init__()
        self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def get_image_embedding(self, frame):
        image = frame.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        features = torch.flatten(features, start_dim=1)
        return features.detach().numpy()