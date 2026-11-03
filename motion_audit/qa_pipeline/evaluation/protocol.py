"""Model-input and MCQ-scoring contract, independent of GPU adapters."""
import re

def display_options(options):return ['Object '+v if v in ['A','B'] else v for v in options]

def prompt(example):
 options='\n'.join(f'{chr(65+i)}. {value}' for i,value in enumerate(display_options(example['options'])))
 return ('The nine images are ordered frames F0 through F8 of one released video sequence. '
  'Frame indices indicate order, not physical seconds. Orange contours mark target object A; '
  'green contours mark target object B, when those targets are requested. '
  'Use the frame range and coordinate convention in the question.\n\n'
  f'Question: {example["question"]}\n\nOptions:\n{options}\n\n'
  'Answer directly with exactly one option LETTER (A, B, C, D, E or F as applicable). '
  'The option letter is its position in the options list, not an object ID. Do not explain.')

def parse_answer(text,options):
 """Single option letter first; semantic fallback never confuses object A with option A."""
 value=str(text).strip();tags=re.findall(r'<answer>(.*?)</answer>',value,re.S|re.I)
 if tags:value=tags[-1].strip()
 value=value.strip('`* \n\t')
 allowed=''.join(chr(65+i) for i in range(len(options)))
 exact=re.fullmatch(r'\(?(['+allowed+r'])\)?[.)\s]*',value,re.I)
 if exact:return ord(exact.group(1).upper())-65
 prefixed=re.fullmatch(r'\(?(['+allowed+r'])\)?[.)\:]\s*(.+)',value,re.I|re.S)
 if prefixed:
  index=ord(prefixed.group(1).upper())-65
  if prefixed.group(2).strip(' .\n').lower()==display_options(options)[index].lower():return index
 final=re.findall(r'(?:final\s+answer|answer|option)\s*(?:is\s*|[:=]\s*)\(?(['+allowed+r'])\)?\b',value,re.I)
 if final:
  if len(set(x.upper() for x in final))==1 and not re.search(r'\b(?:or|and)\s+['+allowed+r']\b',value,re.I):return ord(final[-1].upper())-65
  return None
 clean=re.sub(r'[.\s]+$','',value.lower())
 for index,label in enumerate(display_options(options)):
  if clean==label.lower():return index
 return None

def stats(rows,expected=None):
 total=len(rows);correct=sum(r['correct'] for r in rows)
 return dict(processed=total,expected=expected if expected is not None else total,correct=correct,
  accuracy=correct/total if total else None,invalid=sum(r['status']=='ok' and not r['parsed'] for r in rows),errors=sum(r['status']!='ok' for r in rows))

