import clip
import tqdm
import torch



def CLIP_filter(images,obj,objects,threshold=0.6,cpu=True):
    print('\nfiltering')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cpu:
        device='cpu'
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in objects])
    images_filt=[]
    model, preprocess = clip.load('ViT-B/32', device)
    for image in images:
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = text_inputs.to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        print(f'origin name:{obj}\ttop-1 name:{objects[indices[0]]}\tscore:{values[0]}')
        objname=objects[indices[0]].replace(" ","")
        if objname==obj and values[0]>threshold:
            print("saved!\n")
            images_filt.append(image)
        else:
            print('filt!\n')
    return images_filt
