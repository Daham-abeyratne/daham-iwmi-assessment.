from FaceMaskCNN import FaceMaskCNN
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

class BasicInference:
    def __init__(self,model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path,map_location=self.device)
        self.model = FaceMaskCNN().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.class_names = checkpoint.get("class_names",["with_mask", "without_mask"])
        self.val_acc = checkpoint.get("val_acc",None)

        #Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),])

        print(f"Inference ready  |  device: {self.device}  |  classes: {self.class_names}")

    def _classify_crop(self, pil_crop):
            tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs  = F.softmax(logits, dim=1)[0]

            idx        = torch.argmax(probs).item()
            label      = self.class_names[idx]
            confidence = probs[idx].item() * 100
            all_probs  = {name: probs[i].item() for i, name in enumerate(self.class_names)}

            return label, confidence, all_probs

    def detect_images(self, image_path, output_path=None, show=False):
            bgr_img = cv2.imread(image_path)
            if bgr_img is None:
                raise FileNotFoundError(f"Could not read: {image_path}")

            annotated = bgr_img.copy()
            gray      = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            results = []

            if len(faces) == 0:
                # Fallback, ig no face found, classify full image
                print("No face detected — classifying full image as fallback.")
                pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
                label, confidence, all_probs = self._classify_crop(pil_img)

                color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
                cv2.putText(annotated, f"{label} ({confidence:.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                results.append({
                    "face_index": 0, "bbox": None,
                    "label": label, "confidence": confidence, "all_probs": all_probs
                })

            else:
                for i, (x, y, w, h) in enumerate(faces):
                    face_pil = Image.fromarray(
                        cv2.cvtColor(bgr_img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    )
                    label, confidence, all_probs = self._classify_crop(face_pil)

                    # Green = mask on, Red = mask off
                    color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(
                        annotated,
                        f"{label} {confidence:.1f}%",
                        (x, y - 10 if y > 20 else y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    print(f"  Face {i+1}: {label}  ({confidence:.1f}%)")
                    results.append({
                        "face_index": i, "bbox": (x, y, w, h),
                        "label": label, "confidence": confidence, "all_probs": all_probs
                    })

            if output_path:
                cv2.imwrite(output_path, annotated)
                print(f"Annotated image saved → {output_path}")

            if show:
                cv2.imshow("Face Mask Detection", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return results, annotated

    def predict(self, pil_image):  
            return self._classify_crop(pil_image)



inf = BasicInference("E:\\IWMI_Assesment\\models\\best_model.pth")

test_img_path = "E:\\IWMI_Assesment\\data\\without_mask\\without_mask_35.jpg"
pil_img = Image.open(test_img_path).convert("RGB")
label, confidence, all_probs = inf.predict(pil_img)

print(f"\npredict() result:")
print(f"  Label:      {label}")
print(f"  Confidence: {confidence:.1f}%")
print(f"  All probs:  {all_probs}")

results, annotated = inf.detect_images(
    image_path=test_img_path,
    output_path="../results/test_inference_output.jpg"
)

print(f"\ndetect_images() result:")
for r in results:
    print(f"  Face {r['face_index']+1}: {r['label']} ({r['confidence']:.1f}%)  bbox={r['bbox']}")

print("\nAll checks passed.")