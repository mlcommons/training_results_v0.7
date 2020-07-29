#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Lint as: python3
"""Moses detokenizer.

Moses detokenizer
"""

from __future__ import print_function

import re
import unicodedata
import six

OUTPUT = ('third_party/tensorflow_models/mlperf/models'
          '/rough/nmt/testdata/deen_output')


def is_currency(token):
  for c in token:
    if unicodedata.category(c) != 'Sc':
      return False
  return True


# operates on unicode string and returns unicode string in Python 2
# operates on string and returns string in Python 3
def detokenize_sentence(token_list):
  """Detokenize a single sentence."""
  right_shift_punctuations = re.compile(r'^[\(\[\{]+$')
  left_shift_punctuations = re.compile(r'^[,\.\?!:;\\\%\}\]\)]+$')
  contraction = re.compile(u'^[\'|’][a-zA-Z]')
  pre_contraction = re.compile('[a-zA-Z0-9]$')
  quotes = re.compile(u'^[\'\"„“”]$')
  detok_str = ''
  prepend_space = ' '
  quote_count = {'\'': 0, '\"': 0}
  for i in range(len(token_list)):
    if is_currency(token_list[i]) or \
        right_shift_punctuations.match(token_list[i]):
      detok_str += prepend_space + token_list[i]
      prepend_space = ''
    elif left_shift_punctuations.match(token_list[i]):
      detok_str += token_list[i]
      prepend_space = ' '
    elif i > 0 and contraction.match(token_list[i]) and \
        pre_contraction.search(token_list[i-1]):
      detok_str += token_list[i]
      prepend_space = ' '
    elif quotes.match(token_list[i]):
      normalized_quo = token_list[i]
      normalized_quo = '\"' if re.match(u'^[„“”]$', token_list[i]) \
          else normalized_quo
      assert normalized_quo in quote_count
      if quote_count[normalized_quo] % 2 == 0:
        if normalized_quo == '\'' and i > 0 and \
            re.search('s$', token_list[i-1]):
          detok_str += token_list[i]
          prepend_space = ' '
        else:
          detok_str += prepend_space + token_list[i]
          prepend_space = ''
          quote_count[normalized_quo] += 1
      else:
        detok_str += token_list[i]
        prepend_space = ' '
        quote_count[normalized_quo] += 1
    else:
      detok_str += prepend_space + token_list[i]
      prepend_space = ' '
  detok_str = detok_str.strip()
  if detok_str:
    detok_str += '\n'
  if six.PY2:
    detok_str = detok_str.encode('utf-8')
  return detok_str


def deescape(text):
  """De-escape text."""
  text = re.sub('&bar;', '|', text)
  text = re.sub('&#124;', '|', text)
  text = re.sub('&lt;', '<', text)
  text = re.sub('&gt;', '>', text)
  text = re.sub('&bra;', '[', text)
  text = re.sub('&ket;', ']', text)
  text = re.sub('&quot;', '\"', text)
  text = re.sub('&apos;', '\'', text)
  text = re.sub('&#91;', '[', text)
  text = re.sub('&#93;', ']', text)
  text = re.sub('&amp;', '&', text)
  return text


def detokenize(text):
  detok_list = []
  for line in text:
    if line == '\n':
      detok_list.append(line)
    detok = detokenize_sentence(deescape(line.strip()).split())
    if detok:
      detok_list.append(detok)
  return detok_list


def main():
  if six.PY3:
    with open(OUTPUT, 'r', encoding='utf-8') as fobj:
      detok_list = detokenize(fobj.readlines())
      for line in detok_list:
        print(line, end='')
  else:
    with open(OUTPUT, 'r') as fobj:
      detok_list = detokenize([x.decode('utf-8') for x in fobj.readlines()])
      for line in detok_list:
        print(line, end='')

if __name__ == '__main__':
  main()
