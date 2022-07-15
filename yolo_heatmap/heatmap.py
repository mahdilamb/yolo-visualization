

from typing import Optional
import torch

def xywh2xyxy(prediction:torch.Tensor) -> torch.Tensor:
    """Convert a tensor of xywh to x0,y0,x1,y1
    Args:
        prediction (torch.Tensor): input tensor
    Returns:output tensor
        torch.Tensor: _description_
    """    
    box_corner = prediction.new(*prediction.shape[:-1],4)

    half = prediction[..., 2:4] / 2

    box_corner[..., :2] = prediction[..., :2] - half
    box_corner[..., 2:4] = prediction[..., :2] + half

    return box_corner

def filter_predictions(
    predictions:torch.Tensor,
    scores:torch.Tensor,
        min_score:Optional[float] = None,
        top_n:Optional[int] = None,)->torch.tensor:
    """Filter predictions by minimum value, and possibly trimming to a certain number

    Args:
        predictions (torch.Tensor): the input predictions
        scores (torch.Tensor): the scores with which to filter the predictions. Assumed to be 1D and the same length as the last dimension of predictions
        min_score (Optional[float], optional): The minimum score to allow - ignored if None. Defaults to None.
        top_n (Optional[int], optional): The maximum values to all - ignored if None. Defaults to None.

    Returns:
        torch.tensor: a filtered set of predictions
    """    
    if min_score is not None:
        idx = torch.argwhere(scores >=min_score)
        predictions = predictions[idx]
        scores = scores[idx]
    if top_n is not None and scores.shape[-1] > top_n:
        predictions = predictions[scores >= scores[torch.kthvalue(scores.float(),scores.shape[-1] +1 - top_n).indices]]
    return predictions

def draw_boxes(out:torch.Tensor,scores:torch.Tensor, predictions:torch.Tensor) ->torch.Tensor:
    """Draw the bounding boxes

    Args:
        out (torch.Tensor): the output tensor
        scores (torch.Tensor): The scores for each of the bboxes
        bboxes (torch.Tensor): The predictions

    Returns:
        torch.Tensor: the out tensor, modified
    """
    bboxes = xywh2xyxy(predictions) 
    for score, box in zip(scores,bboxes):
        x0,y0,x1,y1 = box.int().tolist()
        out[y0:y1,x0:x1] += score
    return out

def draw_kdes(out:torch.Tensor,scores:torch.Tensor, predictions:torch.Tensor) ->torch.Tensor:
    """Draw approximate KDEs

    Args:
        out (torch.Tensor): the output tensor
        scores (torch.Tensor): The scores for each of the predictions
        bboxes (torch.Tensor): The predictions

    Returns:
        torch.Tensor: the out tensor, modified
    """
    x_values = torch.arange(0,out.shape[1],device=out.device)
    y_values = torch.arange(0,out.shape[0],device=out.device)
    def gauss1d(v, mu, sig):
        return torch.exp(-torch.square(v - mu) / (2 * sig*sig))

    for score, box in zip(scores,predictions[...,:4]):
        x,y,w,h = box.tolist()
        gauss = torch.outer(gauss1d(y_values,y,h/2),gauss1d(x_values,x,w/2))
        out += gauss.multiply_(score)
    return out

@torch.no_grad()
def create_heatmap(
        predictions:torch.Tensor,
        output_size:tuple[int,int],
        cls:Optional[int] = None,
        use_objectiveness:bool=True,
        use_kde:bool = False,
        min_score:Optional[float] = None,
        top_n:Optional[int] = None,
    ):
    """Create a heatmap

    Args:
        predictions (torch.Tensor): The input predictions - these are assumed to already be scaled to match the output size
        output_size (tuple[int,int]): The output size (most likely the size of the image)
        cls (Optional[int], optional): The class index. If None, will create an objectiveness map. Defaults to None.
        use_objectiveness (bool, optional): Whether to use the objectiveness score. If None, will just use the class confidence scores. Defaults to True.
        use_kde (bool, optional): Whether to use an approximate gaussian KDE. Defaults to False.
        min_score (Optional[float], optional): Whether to filter the bboxes by the minimum confidence level. Defaults to None.
        top_n (Optional[int], optional): Whehter to filter the bboxes to only use a max specified number of bboxes. Defaults to None.

    Raises:
        ValueError: If both cls is None and use_objectiveness is False 

    Returns:
        _type_: a torch tensor of the size specified in output size
    """    
    if cls is None and not use_objectiveness:
        raise ValueError("Must either use a class or objectiveness")
    
    predictions = predictions.squeeze(0).float().cpu()# TODO why do we need .float().cpu() here?
    if cls is not None:
        predictions = torch.cat((predictions[...,:5],predictions[...,(cls+5,)]),axis=1)
        scores = predictions[...,-1]
        if use_objectiveness:
            scores *= predictions[...,4]
    else:
        scores = predictions[...,4]
    
    predictions = filter_predictions(predictions, scores=scores,min_score=min_score, top_n=top_n)
    if use_kde:
        return draw_kdes(predictions.new(*output_size), scores, predictions)   
    return draw_boxes(predictions.new(*output_size), scores, predictions)

