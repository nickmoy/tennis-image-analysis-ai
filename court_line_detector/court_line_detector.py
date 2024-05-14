import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                # ImageNet default means
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), cv2.FILLED)

        return image

    def draw_keypoints_to_video(self, frames, keypoints):
        output_frames = []
        for frame in frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)

        return output_frames
