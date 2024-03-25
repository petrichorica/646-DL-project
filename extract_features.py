from transformers import AutoProcessor, AutoModel, CLIPVisionModel
import torch
import torch.nn as nn

class mySigLip(nn.Module):
    def __init__(self):
        super(mySigLip, self).__init__()
        # model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model.eval()
        self.processor.eval()
        
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

