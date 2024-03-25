from transformers import AutoProcessor, AutoModel, SiglipVisionModel
import torch
import torch.nn as nn
from torchvision import transforms

class mySigLipModel(nn.Module):
    def __init__(self):
        super(mySigLipModel, self).__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.shape = 768

    def get_image_embedding(self, frame):
        image = frame.convert("RGB")
        # transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                                                         std=[0.229, 0.224, 0.225])])
        # image = transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        features = torch.flatten(outputs, start_dim=1)
        return outputs.detach().numpy()
    
class mySigLipVisionModel(nn.Module):
    def __init__(self):
        super(mySigLipVisionModel, self).__init__()
        self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def get_image_embedding(self, frame):
        image = frame.convert("RGB")
        # transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                                                         std=[0.229, 0.224, 0.225])])
        # image = transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        features = torch.flatten(features, start_dim=1)
        return features.detach().numpy()