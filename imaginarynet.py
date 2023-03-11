from gen_prompt import generating_prompts
from gen_image import p_generator
from gen_image import get_xml
from CLIP_filter import CLIP_filter
from tqdm import tqdm
import os
import argparse
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--clip",
        action='store_true',
        help="use CLIP as filter",
    )
    parser.add_argument(
        "--cpu",
        action='store_true',
        help="use CLIP on CPU",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["dalle-mini","stablediffusion"],
        help="choose the backend"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1250,
        help="the number of pictures per prompt"
    )
    parser.add_argument(
        "--classfile",
        type=str,
        default="COCO.txt",
        help="file which records the name of each class"
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default="./dallemini_coco_GPT_10W",
        help="output dir"
    )
    parser.add_argument(
        "--gpt",
        action="store_true",
        help="whether using GPT2 to generate descriptions or not"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="the threshold of clip in filtering the images"
    )
    opt = parser.parse_args()

    random.seed(opt.seed)

    if not os.path.exists(f"{opt.outputdir}"):
        os.mkdir(f"{opt.outputdir}")
        os.mkdir(f"{opt.outputdir}/annotation")
        os.mkdir(f"{opt.outputdir}/image")

    print("generating prompts:")
    objects=generating_prompts(opt.num,opt.classfile,f"{opt.outputdir}/prompt.txt",opt.gpt)
    print("\nstarting generating images\n")
    pg=p_generator(opt.backend)
    with open(f"{opt.outputdir}/prompt.txt") as reader:
        lines = reader.readlines()

    all_idx = 0
    for idx in tqdm(range(0, len(lines))):
        line = lines[idx]
        line = line.strip()
        n = line.count("\t")
        txt, obj= line.split("\t")
        print(txt)
        objs = [obj]
        images, detect_results, clip_score = pg.gen_group(txt, 1)
        if opt.clip:
            images=CLIP_filter(images,obj,objects,threshold=opt.threshold,cpu=opt.cpu)
        for n, img in enumerate(images):
            file_name = "{:0>6d}".format(all_idx)
            #file_name = "{:0>3d}".format(n)
            img.save(f"{opt.outputdir}/image/{file_name}.jpg")
            get_xml(all_idx, objs,opt.outputdir)
            all_idx += 1
if __name__=="__main__":
    main()