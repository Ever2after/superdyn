"""Native baseline adapters. No reference answers are accepted by this API."""
import importlib.metadata
from pathlib import Path
import sys

from ..config import ROOT

class QwenBackend:
 def __init__(self,spec):
  import torch
  import transformers
  from transformers import AutoProcessor
  self.torch=torch;self.kind=spec['backend'];self.spec=spec
  path=spec['path']
  if self.kind=='vgllm':
   sys.path.insert(0,str(ROOT/'vendor/VG-LLM/src'))
   from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT as cls
  elif self.kind=='qwen25':
   from transformers import Qwen2_5_VLForConditionalGeneration as cls
  else:
   from transformers import Qwen3VLForConditionalGeneration as cls
  self.model,info=cls.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',output_loading_info=True)
  assert not info.get('missing_keys'),f'Missing checkpoint weights: {info["missing_keys"][:10]}'
  assert not info.get('mismatched_keys'),info['mismatched_keys']
  self.model.eval()
  patch_check=None
  if self.kind=='qwen3':
   # This non-overlapping, unpadded Conv3d produces exactly one output per
   # already flattened patch. Its native BF16 path rounds GEMM before bias.
   # Preserve that rounding with a bias-free linear call followed by BF16 add.
   import types,time
   patch=self.model.model.visual.patch_embed;conv=patch.proj
   assert conv.groups==1 and conv.kernel_size==conv.stride and conv.padding==(0,0,0) and conv.dilation==(1,1,1)
   def linear_patch(module,hidden_states):
    weight=module.proj.weight
    value=torch.nn.functional.linear(hidden_states.reshape(-1,weight[0].numel()).to(weight.dtype),weight.flatten(1),None)
    return value+module.proj.bias if module.proj.bias is not None else value
   with torch.inference_mode():
    probe=torch.randn(16,conv.weight[0].numel(),device=conv.weight.device,dtype=conv.weight.dtype)
    torch.cuda.synchronize();start=time.perf_counter();original=patch(probe);torch.cuda.synchronize();native_seconds=time.perf_counter()-start
    candidate=linear_patch(patch,probe);torch.cuda.synchronize()
    assert torch.equal(original,candidate),'Patch projection optimization must be bit-exact on this device/checkpoint'
   patch_check=dict(method='bias-free linear then BF16 bias; same non-overlapping Conv3d projection',probe_patches=16,bit_exact=True,max_abs_difference=float((original-candidate).abs().max()),native_probe_seconds=native_seconds)
   patch.forward=types.MethodType(linear_patch,patch)
  self.processor=AutoProcessor.from_pretrained(path,min_pixels=32*28*28,
   max_pixels=518*518 if self.kind=='vgllm' else 200704)
  self.provenance=dict(torch=torch.__version__,transformers=transformers.__version__,attention='sdpa',dtype='bfloat16',
   patch_projection_check=patch_check,model_class=cls.__name__,checkpoint=path,revision=spec['revision'],missing_keys=sorted(info.get('missing_keys',[])),
   unexpected_keys=sorted(info.get('unexpected_keys',[])),geometry_encoder_enabled=self.kind=='vgllm',
   preprocessing='official VG-LLM geometry+RGB crop' if self.kind=='vgllm' else 'native HF ordered multi-image processor')

 def generate(self,prompt,frame_paths):
  import torch
  from PIL import Image
  torch=self.torch;images=[Image.open(path).convert('RGB') for path in frame_paths]
  content=[]
  for i,im in enumerate(images):content.extend([dict(type='text',text=f'Frame F{i}:'),dict(type='image',image=im)])
  content.append(dict(type='text',text=prompt))
  messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=content)]
  text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  if self.kind=='vgllm':
   from qwen_vl.data.utils import load_and_preprocess_images
   geometry=load_and_preprocess_images(images)
   patch=self.processor.image_processor.patch_size;merge=self.processor.image_processor.merge_size
   h,w=geometry.shape[-2:];h-=((h//patch)%merge)*patch;w-=((w//patch)%merge)*patch
   inputs=self.processor(text=[text],images=[im[:,:h,:w] for im in geometry],padding=True,return_tensors='pt',do_rescale=False,do_resize=False)
   inputs['geometry_encoder_inputs']=[geometry.to('cuda')]
  else:inputs=self.processor(text=[text],images=images,return_tensors='pt',padding=True)
  inputs=inputs.to('cuda')
  with torch.inference_mode():
   output=self.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=64,use_cache=True)
  new=output[:,inputs.input_ids.shape[1]:]
  raw=self.processor.batch_decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  return raw,dict(input_tokens=int(inputs.input_ids.shape[1]),generated_tokens=int(new.shape[1]),frames=9,
   geometry_inputs_supplied=self.kind=='vgllm')

class RGPTBackend:
 def __init__(self,spec):
  import torch
  import transformers
  sys.path.insert(0,str(ROOT/'vendor/4D-RGPT'))
  import llava
  self.model=llava.load(spec['path'],device_map='cuda:0',attn_implementation='sdpa',torch_dtype=torch.bfloat16).eval().cuda()
  assert self.model.encoder_l4p is None,'Training-only teacher must be disabled for inference'
  self.provenance=dict(torch=torch.__version__,transformers=transformers.__version__,attention='sdpa',dtype='bfloat16',
   checkpoint=spec['path'],revision=spec['revision'],model_class=type(self.model).__name__,geometry_teacher_inference=False,
   compatibility='Official v4.46 stack on installed B200-compatible torch; lazy unused encoders, no training-only teacher, native SDPA.')

 def generate(self,prompt,frame_paths):
  import torch
  from PIL import Image
  images=[Image.open(path).convert('RGB') for path in frame_paths]
  content=[]
  for i,im in enumerate(images):content.extend([f'Frame F{i}:',im])
  content.append(prompt)
  config=self.model.default_generation_config
  config.do_sample=False;config.num_beams=1;config.max_new_tokens=64
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   raw=self.model.generate_content(content,generation_config=config)
  return raw,dict(frames=9,generated_tokens=len(self.model.tokenizer.encode(raw,add_special_tokens=False)))

class DSRBackend:
 def __init__(self,spec):
  import torch,transformers
  from transformers import AutoProcessor
  sys.path.insert(0,str(ROOT/'vendor/DSR_Suite/model/qwen-vl-finetune'))
  from qwenvl.train.qwen_vl_spatial import Qwen2_5_VLForConditionalGeneration_Spatial
  self.torch=torch
  self.model,info=Qwen2_5_VLForConditionalGeneration_Spatial.from_pretrained(
   spec['path'],torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',output_loading_info=True)
  assert not info.get('missing_keys'),info.get('missing_keys')
  assert not info.get('mismatched_keys'),info.get('mismatched_keys')
  assert not info.get('unexpected_keys'),info.get('unexpected_keys')
  self.model.eval()
  self.processor=AutoProcessor.from_pretrained(spec['path'],min_pixels=32*28*28,max_pixels=200704)
  self.geometry_queries=int(self.model.visual.q_former_queries.shape[0])
  assert self.geometry_queries==32
  self.provenance=dict(torch=torch.__version__,transformers=transformers.__version__,attention='sdpa',dtype='bfloat16',
   model_class=type(self.model).__name__,checkpoint=spec['path'],revision=spec['revision'],vendor_revision=spec['vendor_revision'],
   missing_keys=[],unexpected_keys=[],geometry_encoder_enabled=True,geometry_queries=self.geometry_queries,
   preprocessing='Same nine marked RGB frames in native video packaging; final frame repeated by native temporal patch padding; 32 GSM tokens appended per official inference script.',
   temporal_position_policy='fps=2 maps native temporal-patch positions to ordered frame indices; not measured physical seconds',
   external_geometry_supplied=False)

 def generate(self,prompt,frame_paths):
  import numpy as np
  from PIL import Image
  torch=self.torch
  frames=np.stack([np.asarray(Image.open(path).convert('RGB')) for path in frame_paths])
  messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=[dict(type='video',video=frame_paths),dict(type='text',text=prompt)])]
  text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=self.processor(text=[text],videos=[frames],fps=2,padding=True,return_tensors='pt')
  ids=inputs['input_ids'];token=self.model.config.video_token_id
  positions=(ids[0]==token).nonzero().flatten()
  assert len(positions)>0 and torch.all(positions[1:]==positions[:-1]+1)
  end=int(positions[-1])+1
  extra=torch.full((1,self.geometry_queries),token,dtype=ids.dtype)
  inputs['input_ids']=torch.cat([ids[:,:end],extra,ids[:,end:]],dim=1)
  inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
  inputs=inputs.to('cuda')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   output=self.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=64,use_cache=True)
  new=output[:,inputs.input_ids.shape[1]:]
  raw=self.processor.batch_decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  return raw,dict(input_tokens=int(inputs.input_ids.shape[1]),generated_tokens=int(new.shape[1]),frames=9,
   native_video_grid_thw=inputs['video_grid_thw'].tolist(),geometry_queries=self.geometry_queries,
   geometry_encoder_enabled=True,external_geometry_supplied=False)

def load_backend(spec):
 if spec['backend']=='4drgpt':return RGPTBackend(spec)
 if spec['backend']=='dsr':return DSRBackend(spec)
 return QwenBackend(spec)
