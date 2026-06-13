from FaceMaskCNN import FaceMaskCNN
import torch
import cv2
import torchvision.transforms as transforms

class BasicInference:
    def __init__(self,model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceMaskCNN().to(self.device)
        checkpoint = torch.load(model_path,map_location=self.device)

        self.class_names = checkpoint.get("class_names",["with_mask", "without_mask"])
        self.val_acc = checkpoint.get("val_acc",None)

        #Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.299,0.224,0.225]),])

        print(f"Inference ready  |  device: {self.device}  |  classes: {self.class_names}")

inf = BasicInference("E:\\IWMI_Assesment\\models\\best_model.pth")
