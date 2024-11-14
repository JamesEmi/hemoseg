import os
import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from model import Model
from PIL import Image
from datetime import datetime

class HemorrhageAssessment:
    def __init__(self, hemobox_checkpoint_path, mask2former_model_id, zero_outside_bbox=True, results_dir="results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zero_outside_bbox = zero_outside_bbox
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Load Hemobox model (for blood mask)
        self.hemobox_model = Model("FPN", "resnet34", in_channels=3, out_classes=1)
        self.hemobox_model = Model.load_from_checkpoint(checkpoint_path=hemobox_checkpoint_path,
                                                arch="FPN",
                                                encoder_name="resnet34",
                                                in_channels=3,
                                                out_classes=1)
        self.hemobox_model.to(self.device).eval()

        # Load Mask2Former model (for person mask)
        self.processor = AutoImageProcessor.from_pretrained(mask2former_model_id)
        self.mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_model_id)
        self.mask2former_model.to(self.device).eval()
        self.threshold = 0.7
        
        # Define the colormap for person mask
        self.cmap = [(255, 0, 255)]
        self.id2color = {0: (255, 0, 255)}

    def pad_im(self, image, target_shape=(768, 768)):
        # Expecting image in CHW format
        if image.ndim == 3 and image.shape[0] == 3:  # CHW format
            c, h, w = image.shape
        else:
            raise ValueError(f"Unexpected image format: {image.shape}")

        # Calculate scaling factor to fit within target shape while maintaining aspect ratio
        scale = min(target_shape[0] / h, target_shape[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image while maintaining aspect ratio
        resized_image = np.array([cv2.resize(img, (new_w, new_h)) for img in image])

        pad_h = target_shape[0] - new_h
        pad_w = target_shape[1] - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_im = np.pad(resized_image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        return padded_im

    def get_blood_mask(self, image):
        # image is expected to be in CHW format, float32
        # No need to transpose again
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        print(f"Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
        with torch.no_grad():
            blood_logits = self.hemobox_model(image_tensor)
        print(f"blood_logits shape: {blood_logits.shape}, dtype: {blood_logits.dtype}")
        pr_masks = blood_logits.sigmoid().cpu().numpy()
        for pr_mask in pr_masks:
            np_mask = pr_mask.squeeze()
            np_mask = (np_mask * 255).astype(np.uint8)
            return np_mask

    def get_person_mask(self, image):
        # image is expected to be in HWC format, uint8
        print(f"Here in get_person_mask:")
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")

        if image.dtype != np.uint8:
            image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image
        pil_image = Image.fromarray(image_uint8)
        print(f"PIL image mode: {pil_image.mode}, size: {pil_image.size}")
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.mask2former_model(**inputs)
        results = self.processor.post_process_instance_segmentation(outputs, target_sizes=[pil_image.size[::-1]])[0]
        person_segment_ids = [seg['id'] for seg in results['segments_info'] if seg['label_id'] == 0]
        if not person_segment_ids:
            print("No person found in image")
            return None
        # Find the largest person segment
        largest_mask = None
        largest_area = 0
        
        for person_segment_id in person_segment_ids:
            mask = (results['segmentation'].numpy() == person_segment_id)
            mask_area = np.count_nonzero(mask)  # Count the non-zero pixels

            if mask_area > largest_area:
                largest_area = mask_area
                largest_mask = mask

        if largest_mask is None:
            self.get_logger().warn("No valid 'person' mask found.")
            return None
    
        visual_mask = (largest_mask * 255).astype(np.uint8)
        visual_mask = Image.fromarray(visual_mask)

        return visual_mask

    def extend_box(self, box, factor, image_shape):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Calculate the new width and height extension
        new_width = int(width * factor)
        new_height = int(height * factor)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_x1 = max(center_x - new_width // 2, 0)
        new_y1 = max(center_y - new_height // 2, 0)
        new_x2 = min(center_x + new_width // 2, image_shape[0]-1)
        new_y2 = min(center_y + new_height // 2, image_shape[1]-1)

        # Ensure that the new box is valid, i.e., x2 > x1 and y2 > y1
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            print(f"Invalid box dimensions after extension: {new_x1, new_y1, new_x2, new_y2}")
            return x1, y1, x2, y2  # Return the original box if extension fails
        
        print(f"Original box: {x1, y1, x2, y2}")
        print(f"Extended box: {new_x1, new_y1, new_x2, new_y2}")

        return new_x1, new_y1, new_x2, new_y2

    def calculate_ratios(self, blood_mask, person_mask):
        person_pixels = np.count_nonzero(person_mask)
        blood_pixels_outside = np.count_nonzero(blood_mask & ~person_mask)
        overlap_ratio = np.count_nonzero(blood_mask & person_mask) / person_pixels if person_pixels > 0 else 0
        return blood_pixels_outside / person_pixels if person_pixels > 0 else 0, overlap_ratio
    
    def create_custom_colormap(self):
        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        colormap[0] = [0, 0, 0]
        return colormap
    
    def overlay_mask(self, image, mask, alpha):
        print('MASK', mask.shape)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        elif mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)  # Squeeze to (768, 768)

        alpha = alpha    
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        cust_colormap = self.create_custom_colormap()
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask = cv2.LUT(mask, cust_colormap)
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)

        return overlay

    def save_results(self, base_name, image, blood_mask, raw_overlay, person_mask, overlay):
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_blood_mask.png"), blood_mask)
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_person_mask.png"), person_mask)
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_raw_overlay.png"), raw_overlay)
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_overlay.png"), overlay)
    
    def binarize_mask(self, tensor):
        # Apply the threshold: values below become 0, above become 255 (here I am pushing to 0 and 1 so that it works with the colormap function).
        #clean this up later so as to use just 1 viz function.
        result_tensor = torch.where(tensor < self.threshold, torch.tensor(0, dtype=tensor.dtype), torch.tensor(1, dtype=tensor.dtype))
        return result_tensor

    def visualize_map(self, image, segmentation_map):
        color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
        for label, color in self.id2color.items():
            color_seg[segmentation_map == label, :] = color
        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        return img

    def process_image(self, image_path):
        # Read and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
        image = image.astype(np.float32)  # Convert image to float32

        # Transpose to CHW format
        image_chw = np.transpose(image, (2, 0, 1))
        # Pad the image
        image_padded = self.pad_im(image_chw)

        # Get blood mask
        np_mask = self.get_blood_mask(image_padded)
        np_mask_3d = np_mask[:, :, np.newaxis]
        # Prepare image for overlay (HWC format, uint8)
        image_hwc_uint8 = np.transpose(image_padded, (1, 2, 0)).astype(np.uint8)
        print(f"Trying to overlay image of shape: {image_hwc_uint8.shape}, with np_mask of shape {np_mask_3d.shape}")
        raw_overlay = self.overlay_mask(image_hwc_uint8, np_mask_3d, alpha=0.5)
        
        # Prepare image for person mask model (HWC format, uint8)
        image_padded_hwc_uint8 = np.transpose(image_padded, (1, 2, 0)).astype(np.uint8)
        person_mask = self.get_person_mask(image_padded_hwc_uint8)

        if person_mask is None:
            print(f"No person detected in {os.path.basename(image_path)}.")
            return None
        
        person_mask = np.array(person_mask)
        person_mask_inv = 255 - np.array(person_mask)  # For visualization
        non_zero_pixels = np.where(person_mask > 0)  # Get non-zero pixel indices
        
        largest_box = None
        largest_box_area = 0


        if non_zero_pixels[0].size > 0 and non_zero_pixels[1].size > 0:
            y1, y2 = non_zero_pixels[0].min(), non_zero_pixels[0].max()  # Min and max row indices
            x1, x2 = non_zero_pixels[1].min(), non_zero_pixels[1].max()  # Min and max col indices
            x1, y1, x2, y2 = self.extend_box((x1, y1, x2, y2), factor=1.2, image_shape=image_padded_hwc_uint8.shape)
            largest_box = (x1, y1, x2, y2)
            if self.zero_outside_bbox:
                np_mask[:y1, :] = np_mask[y2:, :] = np_mask[:, :x1] = np_mask[:, x2:] = 0
        else:
            print(f"No person detected in {os.path.basename(image_path)}.")
            return None
        
        blood_mask_bbox_tensor = torch.tensor(np_mask).to(self.device)
        blood_mask_bbox_thres = self.binarize_mask(blood_mask_bbox_tensor).cpu().numpy()
        # self.get_logger().info(f"Person mask has shape {person_mask.shape} with {np.count_nonzero(person_mask)} non-zero pixels.")
        blood_pixel_ext = np.count_nonzero(blood_mask_bbox_thres & ~person_mask)
        # self.get_logger().info(f"Thresholded blood mask has shape {blood_mask_bbox_thres.shape} with {np.count_nonzero(blood_mask_bbox_thres)} non-zero pixels.")
        blood_pixel_ext = np.count_nonzero(blood_mask_bbox_thres & ~person_mask)
        # self.get_logger().info(f"External blood has a pixel count of {blood_pixel_ext}.")
        total_person_pixels = np.count_nonzero(person_mask)
        # self.get_logger().info(f"Person seg has a pixel count of {total_person_pixels}.")
        blood_ext_ratio = blood_pixel_ext / total_person_pixels if total_person_pixels > 0 else 0 #be careful with this if condition
        overlap_ratio = np.count_nonzero(blood_mask_bbox_thres & person_mask) / np.count_nonzero(person_mask)
        # blood_ext_ratio, overlap_ratio = self.calculate_ratios(np_mask, person_mask)
        severe_hemorrhage = (blood_ext_ratio > 0.01) or (overlap_ratio > 0.5)
        
        np_mask_tensor = torch.tensor(np_mask).to(self.device)
        np_mask_thres = self.binarize_mask(np_mask_tensor).cpu().numpy()
        # person_mask_vis = self.visualize_map(np_image, person_mask_inv)
        print(f"Attempting to overlay image_hwc_uint8 of shape {image_hwc_uint8.shape} ith person_mask_inv of shape {person_mask_inv.shape}")
        person_mask_vis = self.visualize_map(image_hwc_uint8, person_mask_inv)
        print(f"Attempting to overlay person_mask_vis of shape {person_mask_vis.shape} ith np_mask_thres of shape {np_mask_thres.shape}")
        np_mask_thres_3d = np_mask_thres[:, :, np.newaxis]
        print(f"Attempting to overlay person_mask_vis of shape {person_mask_vis.shape} ith np_mask_thres_3d of shape {np_mask_thres_3d.shape}")
        overlaid_image = self.overlay_mask(person_mask_vis, np_mask_thres_3d, alpha=0.5)
        
        if largest_box is not None:
                cv2.rectangle(overlaid_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"Severe Hemorrhage" if severe_hemorrhage else "Normal"
                cv2.putText(overlaid_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # overlay = self.overlay_mask(image_hwc_uint8, np_mask)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.save_results(base_name, image_hwc_uint8, np_mask, raw_overlay, person_mask, overlaid_image)

        result = {
            "image": base_name,
            "blood_ext_ratio": blood_ext_ratio,
            "overlap_ratio": overlap_ratio,
            "severe_hemorrhage": severe_hemorrhage
        }
        return result

    def process_directory(self, image_dir):
        results_log = []
        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, image_file)
                result = self.process_image(image_path)
                if result:
                    results_log.append(result)

        # Save results log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_log_path = os.path.join(self.results_dir, f"hemorrhage_assessment_log_{timestamp}.txt")
        with open(results_log_path, "w") as log_file:
            for result in results_log:
                log_file.write(f"{result}\n")
        print(f"Results saved to {self.results_dir}.")

if __name__ == "__main__":
    hemobox_checkpoint_path = "/notebooks/triage/hemofpn/lightning_logs/version_8/checkpoints/epoch=7-step=159.ckpt"
    mask2former_model_id = "facebook/mask2former-swin-large-coco-instance"
    image_dir = "/notebooks/triage/hemofpn/data_f8/train_sample"

    hemorrhage_assessment = HemorrhageAssessment(
        hemobox_checkpoint_path=hemobox_checkpoint_path,
        mask2former_model_id=mask2former_model_id,
        zero_outside_bbox=True,  # Toggle to control blood mask zeroing outside bbox
        results_dir="/notebooks/triage/hemofpn/data_f8/train_sample/results5"
    )

    hemorrhage_assessment.process_directory(image_dir)
