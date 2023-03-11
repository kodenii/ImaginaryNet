from transformers import pipeline, set_seed
from tqdm import tqdm
import numpy as np
import argparse

def generating_prompts(num,classfile,outfile,gpt=True):
	with open(classfile) as f:
		lines=f.readlines()
	objects=[line.replace("\n",'') for line in lines]
	generator = pipeline('text-generation', model='gpt2-large', framework="pt") if gpt else None
	set_seed(42)
	n=num
	bar = tqdm(total=len(objects)*n)
	with open(outfile, "w") as writer:
		for obj in objects:
			sent_pool = set()
			if not gpt:
				i=0
				while i<n:
					objname = obj.replace(" ", "")
					t=f"A photo of {obj}\t{objname}\n"
					writer.writelines(t)
					bar.update(1)
					i+=1
			else:
				while len(sent_pool) < n:
					for sent in generator(f"A photo of {obj}", max_length=15, num_return_sequences=n):
						sent = sent['generated_text']
						sent = sent.split(".")[0].split("\n")[0].split("\t")[0]
						if sent in sent_pool:
							continue
						bar.update()
						sent_pool |= set([sent])
						objname = obj.replace(" ", "")
						t = f"{sent}\t{objname}\n"
						writer.writelines(t)
						if len(sent_pool) == n:
							break
	return objects