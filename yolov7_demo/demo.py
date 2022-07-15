import torch

from yolo_heatmap.heatmap import create_heatmap
import numpy as np
from yolov7.utils.general import scale_coords
@torch.no_grad()
def create_predictions(weights,img_path,imgsz = 640,
    half = True,
    use_cuda = torch.cuda.is_available()
    ):
    from yolov7.utils.datasets import letterbox
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.general import check_img_size, scale_coords

    device = "cuda" if use_cuda else "cpu"
    half &= device == "cuda"
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model = model.half()
    im0 = cv2.imread(img_path)
    img, ratio,_ = letterbox(im0.copy(),  stride=stride)
    # Convert
    img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    import matplotlib.pyplot as plt

    pred = model(img)[0]
    for i in range(pred.shape[0]):
        pred[i,:, :4] = scale_coords(img.shape[2:], pred[i,:, :4], im0.shape).round()
    return pred, im0,img

if __name__ == "__main__":
    import cv2
    
    import matplotlib.pyplot as plt

    pred, im0,img = create_predictions(weights="./yolov7.pt", img_path=r"./yolov7/inference/images/horses.jpg")

    horse = 37
    
    samples = {
        
        "horsiness": create_heatmap(pred,cls=horse, use_objectiveness=True, output_size = im0.shape[:2]),
        "horsiness_kde":create_heatmap(pred,cls=horse, use_objectiveness=True, output_size = im0.shape[:2],   top_n=100, use_kde=True),
        "objectiveness": create_heatmap(pred,cls=None, use_objectiveness=True, output_size = im0.shape[:2]),
        "objectiveness_kde":create_heatmap(pred,cls=None, use_objectiveness=True, output_size = im0.shape[:2],   top_n=100, use_kde=True),
    }
    fig, axs = plt.subplots(2,2)
    for i,(label, heatmap) in enumerate(samples.items()):
        ax = axs[i//2, int("kde" in label)]
        ax.imshow(im0)
        ax.set_title(label)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(heatmap.detach().cpu().numpy(),alpha=.8)
    plt.tight_layout()
    plt.savefig("horses_demo.jpg")
    
    